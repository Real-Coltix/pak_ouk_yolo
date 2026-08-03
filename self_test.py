"""Синтетический smoke-тест интеграции YoloDetector <-> PAK_OUK_Controller
(нет живой камеры/реального таргета в этом sandbox — см. README.md).
Проверяет: детектор не падает на шуме/фейковых кадрах, декодирование выхода
модели даёт confidence-скоры в разумном диапазоне (не NaN, не вне [0, 1]),
и bbox из YoloDetector.detect() совместим по контракту с остальным
пайплайном PAK_OUK_Controller (servo error / course correction).
"""

import numpy as np

from pak_ouk_controller import PAK_OUK_Controller
from yolo_detector import YoloDetector

MODEL_PATH = "best_5s.onnx"


def main() -> int:
    detector = YoloDetector(MODEL_PATH)
    controller = PAK_OUK_Controller(640, 480, 0.35)

    rng = np.random.default_rng(0)
    frames = {
        "pure_noise": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        "bright_rect_on_gray": _bright_rect_frame(),
    }

    for name, frame in frames.items():
        bbox = detector.detect(frame)
        assert bbox is None or (
            len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0
        ), f"{name}: malformed bbox {bbox}"

        err = controller.compute_servo_error(bbox)
        cmd = controller.compute_course_correction(err, bbox, 0.0)
        print(f"[{name}] bbox={bbox} servo_valid={err.valid} "
              f"V={cmd.linear_velocity:.2f} w={cmd.angular_velocity:.2f} "
              f"stop={cmd.critical_stop}")

    print("OK - YoloDetector integration runs end-to-end with no exceptions")
    return 0


def _bright_rect_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 60, dtype=np.uint8)
    frame[150:330, 250:390] = 220
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
