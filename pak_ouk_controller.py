"""Python-порт PAK_OUK_Controller (../src/pak_ouk_controller.hpp) — те же пять
модулей, тот же порядок вызовов cv2, те же константы. Существует только для
webcam_test.py: тест на реальной USB-камере без C++ тулчейна на Windows
(pip install opencv-python numpy — ничего больше). "Настоящая" реализация
для целевого железа — C++ хедер, этот файл его не заменяет и не должен
использоваться на встраиваемой системе.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ServoError:
    x_error: float = 0.0
    y_error: float = 0.0
    x_error_norm: float = 0.0
    y_error_norm: float = 0.0
    x_cmd: float = 0.0
    y_cmd: float = 0.0
    valid: bool = False


@dataclass
class MotionCommand:
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    critical_stop: bool = False


@dataclass
class FrameTelemetry:
    fps: float = 0.0
    buffer_queue_depth: int = 0
    buffer_queue_capacity: int = 1


class PAK_OUK_Controller:
    # Требуют калибровки на реальном железе/оптике — подобраны на
    # синтетическом тесте как безопасная отправная точка (см. hpp-версию).
    K_ADAPTIVE_BLOCK_SIZE = 31
    K_ADAPTIVE_C = 15.0
    K_MIN_CONTOUR_AREA = 40.0
    K_MAX_CONTOUR_AREA_FRACTION = 0.7
    K_MIN_EXTENT = 0.15

    K_KP_SERVO_X = 1.0
    K_KP_SERVO_Y = 1.0
    K_KW_YAW = 1.0
    K_MAX_APPROACH_RATE = 5000.0  # px^2/s
    K_MAX_MISSED_FRAMES = 5

    K_FILTER_WINDOW = 5
    K_MAX_JUMP_PX = 80.0
    K_BUFFER_OVERRUN_RATIO = 0.85

    def __init__(self, frame_width: int, frame_height: int,
                 critical_area_fraction: float = 0.35):
        assert frame_width > 0 and frame_height > 0
        assert 0.0 < critical_area_fraction <= 1.0

        self.frame_cx = frame_width * 0.5
        self.frame_cy = frame_height * 0.5
        self.frame_area = float(frame_width) * frame_height
        self.critical_area = self.frame_area * critical_area_fraction

        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        self._missed_frame_count = 0

        self._history = np.zeros((self.K_FILTER_WINDOW, 2), dtype=np.float32)
        self._history_idx = 0
        self._history_count = 0
        self._consecutive_rejects = 0
        self._last_filtered = np.array([0.0, 0.0], dtype=np.float32)

    # ---- 1. Адаптивная background-agnostic детекция ------------------------
    def detect_target(self, gray_src: np.ndarray):
        """Вход: 8-битный однокан. кадр. Возврат: (x, y, w, h) или None."""
        equalized = cv2.equalizeHist(gray_src)

        bright_mask = cv2.adaptiveThreshold(
            equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self.K_ADAPTIVE_BLOCK_SIZE, -self.K_ADAPTIVE_C)
        dark_mask = cv2.adaptiveThreshold(
            equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            self.K_ADAPTIVE_BLOCK_SIZE, self.K_ADAPTIVE_C)
        binary = cv2.bitwise_or(bright_mask, dark_mask)

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._open_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._close_kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        max_plausible_area = self.frame_area * self.K_MAX_CONTOUR_AREA_FRACTION
        best_area = 0.0
        best_bbox = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.K_MIN_CONTOUR_AREA or area > max_plausible_area:
                continue
            bbox = cv2.boundingRect(c)
            bbox_area = bbox[2] * bbox[3]
            if bbox_area <= 0:
                continue
            extent = area / bbox_area
            if extent < self.K_MIN_EXTENT:
                continue
            if area > best_area:
                best_area = area
                best_bbox = bbox

        return best_bbox

    # ---- 2. Visual servoing --------------------------------------------------
    def compute_servo_error(self, bbox) -> ServoError:
        e = ServoError()
        if bbox is None or bbox[2] * bbox[3] <= 0:
            return e

        x, y, w, h = bbox
        cx = x + w * 0.5
        cy = y + h * 0.5

        e.x_error = cx - self.frame_cx
        e.y_error = cy - self.frame_cy
        e.x_error_norm = e.x_error / self.frame_cx
        e.y_error_norm = e.y_error / self.frame_cy
        e.x_cmd = self.K_KP_SERVO_X * e.x_error_norm
        e.y_cmd = self.K_KP_SERVO_Y * e.y_error_norm
        e.valid = True
        return e

    # ---- 3. Контроль дистанции -----------------------------------------------
    def check_proximity(self, bbox) -> bool:
        if bbox is None:
            return False
        return (bbox[2] * bbox[3]) >= self.critical_area

    # ---- 4. Курс и коррекция --------------------------------------------------
    def compute_course_correction(self, err: ServoError, bbox,
                                   distance_rate: float) -> MotionCommand:
        cmd = MotionCommand()

        if not err.valid:
            self._missed_frame_count += 1
            cmd.critical_stop = self._missed_frame_count >= self.K_MAX_MISSED_FRAMES
            return cmd
        self._missed_frame_count = 0

        critical = self.check_proximity(bbox)
        cmd.critical_stop = critical

        cmd.angular_velocity = float(np.clip(-self.K_KW_YAW * err.x_error_norm, -1.0, 1.0))

        bbox_area = bbox[2] * bbox[3]
        area_fraction = bbox_area / self.frame_area
        critical_fraction = self.critical_area / self.frame_area
        fill_factor = float(np.clip(1.0 - area_fraction / critical_fraction, 0.0, 1.0))
        if distance_rate > 0.0:
            approach_penalty = float(np.clip(1.0 - distance_rate / self.K_MAX_APPROACH_RATE, 0.0, 1.0))
        else:
            approach_penalty = 1.0

        cmd.linear_velocity = 0.0 if critical else fill_factor * approach_penalty
        if critical:
            cmd.angular_velocity = 0.0

        return cmd

    # ---- 5. Диагностика --------------------------------------------------------
    def run_diagnostics(self, telemetry: FrameTelemetry, raw_centroid):
        """raw_centroid: (x, y) или None, если детекции не было в этом кадре.
        Возврат: (healthy: bool, filtered_centroid: np.ndarray[2])."""
        healthy = True

        if telemetry.fps <= 0.0:
            healthy = False

        if telemetry.buffer_queue_capacity > 0:
            fill_ratio = telemetry.buffer_queue_depth / telemetry.buffer_queue_capacity
            if fill_ratio >= self.K_BUFFER_OVERRUN_RATIO:
                healthy = False

        filtered = self._filter_centroid(raw_centroid)
        return healthy, filtered

    def _filter_centroid(self, raw):
        if raw is None:
            return self._last_filtered

        raw_arr = np.array(raw, dtype=np.float32)

        if self._history_count > 0:
            jump = float(np.linalg.norm(raw_arr - self._last_filtered))
            if jump > self.K_MAX_JUMP_PX:
                self._consecutive_rejects += 1
                if self._consecutive_rejects < self.K_FILTER_WINDOW:
                    return self._last_filtered
                self._history_count = 0
                self._history_idx = 0

        self._consecutive_rejects = 0

        self._history[self._history_idx] = raw_arr
        self._history_idx = (self._history_idx + 1) % self.K_FILTER_WINDOW
        if self._history_count < self.K_FILTER_WINDOW:
            self._history_count += 1

        self._last_filtered = self._history[: self._history_count].mean(axis=0)
        return self._last_filtered
