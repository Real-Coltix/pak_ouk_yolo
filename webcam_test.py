"""webcam_test.py — тест PAK_OUK_Controller на живом кадре с USB-камеры, где
детекция цели заменена на YOLO (best_5s.onnx) вместо классического
adaptive-threshold + contours из detect_target(). Остальной пайплайн
(servo-ошибка, proximity, курс, диагностика) не менялся — тот же
PAK_OUK_Controller, что и в ../ssc338q-opencv-project/windows_webcam_test.
Зависимости: opencv-python, numpy, onnxruntime (pip install -r requirements.txt).
"""

import argparse
import time

import cv2

from pak_ouk_controller import FrameTelemetry, PAK_OUK_Controller
from yolo_detector import YoloDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam-index", type=int, default=0,
                         help="индекс камеры (Device Manager -> Cameras)")
    parser.add_argument("--model", type=str, default="best_5s.onnx",
                         help="путь к ONNX-модели YOLO")
    parser.add_argument("--critical-area-fraction", type=float, default=0.35,
                         help="доля площади кадра для CRITICAL_PROXIMITY")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Не удалось открыть камеру с индексом {args.cam_index}. "
              "Проверьте Device Manager -> Cameras и попробуйте другой индекс.")
        return 1

    detector = YoloDetector(args.model)

    controller = None
    have_history = False
    prev_area = 0.0
    prev_time = 0.0

    print(f"Камера {args.cam_index} открыта. Окно предпросмотра: 'q' — выход.")

    try:
        while True:
            loop_start = time.perf_counter()

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("Пустой кадр — камера отключена или занята другим приложением?")
                break

            if controller is None:
                h, w = frame_bgr.shape[:2]
                controller = PAK_OUK_Controller(w, h, args.critical_area_fraction)

            # ---- 1. Детекция (YOLO) -----------------------------------------
            bbox = detector.detect(frame_bgr)
            detected = bbox is not None

            # ---- 2. Servo-ошибка --------------------------------------------
            servo_err = controller.compute_servo_error(bbox) if detected else \
                controller.compute_servo_error(None)

            # Скорость роста площади bbox — как в webcam_test.py (classic).
            now = time.perf_counter()
            distance_rate = 0.0
            if detected:
                area = float(bbox[2] * bbox[3])
                if have_history:
                    dt = now - prev_time
                    if dt > 0.0:
                        distance_rate = (area - prev_area) / dt
                prev_area = area
                prev_time = now
                have_history = True
            else:
                have_history = False

            # ---- 4. Курс и коррекция ------------------------------------------
            cmd = controller.compute_course_correction(servo_err, bbox, distance_rate)

            # ---- 5. Диагностика --------------------------------------------------
            frame_dt = now - loop_start
            telemetry = FrameTelemetry(
                fps=(1.0 / frame_dt) if frame_dt > 0.0 else 0.0,
                buffer_queue_depth=0,
                buffer_queue_capacity=1,
            )
            raw_centroid = (bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5) if detected else None
            healthy, filtered_centroid = controller.run_diagnostics(telemetry, raw_centroid)
            if not healthy:
                print(f"[diag] pipeline unhealthy: fps={telemetry.fps:.1f}")

            # ---- Предпросмотр --------------------------------------------------
            vis = frame_bgr
            if detected:
                x, y, w, h = bbox
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(vis, (int(filtered_centroid[0]), int(filtered_centroid[1])),
                           4, (0, 0, 255), -1)
            overlay = (f"det={int(detected)} V={cmd.linear_velocity:.2f} "
                       f"w={cmd.angular_velocity:.2f} stop={int(cmd.critical_stop)} "
                       f"fps={telemetry.fps:.1f} {'OK' if healthy else 'UNHEALTHY'}")
            cv2.putText(vis, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

            cv2.imshow("PAK_OUK YOLO webcam test - q to quit", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
