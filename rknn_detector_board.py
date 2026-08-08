"""YOLO-детектор на NPU RK3588/RK3588S (RKNNLite), прямая замена YoloDetector.

Возвращает тот же контракт `(x, y, w, h) | None`, что и `YoloDetector.detect()`
и `PAK_OUK_Controller.detect_target()`, поэтому подключается в webcam_test.py
без изменений servo/proximity/course-логики.

ЗАПУСКАЕТСЯ ТОЛЬКО НА ПЛАТЕ (aarch64 + rknpu-драйвер + librknnrt.so).
На x86-хосте импорт rknnlite упадёт — это ожидаемо, для хоста есть
yolo_detector.py (onnxruntime) и eval_rknn_accuracy.py (симулятор).

Граф — RKNN-экспорт (см. export_rknn_onnx.py): 9 выходов
(3 уровня x box_raw / cls_sigmoid / cls_sum), DFL и декод НЕ в графе.
Декод ниже — та же математика, что в eval_rknn_accuracy.py; держите их
синхронными, иначе симулятор на хосте и плата разойдутся в результатах.

Препроцессинг: mean/std зашиты в модель при конвертации
(mean=0, std=255), поэтому на вход подаётся СЫРОЙ uint8 RGB NHWC —
никакого /255 и никакого float на CPU.

NB: не проверено на реальном железе (в песочнице разработки платы нет).
Проверялась только совместимость декода с eval_rknn_accuracy.py.
"""

import cv2
import numpy as np

from rknnlite.api import RKNNLite


class RknnYoloDetector:
    INPUT_SIZE = 640
    CONF_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.45
    PAD = 114

    #: RK3588/RK3588S: 3 ядра NPU. AUTO отдаёт модель на любое свободное —
    #: это то, что нужно при нескольких инстансах в пуле потоков.
    #: 0_1_2 просит рантайм разложить слои одной модели на 3 ядра: latency
    #: одного кадра ниже, суммарный throughput пула — тоже (ядра заняты
    #: одним запросом). Для «один поток, минимальная задержка» — 0_1_2,
    #: для «максимум FPS» — пул из 3 инстансов с AUTO/по ядру на каждый.
    CORE_MASK = RKNNLite.NPU_CORE_0_1_2

    def __init__(self, rknn_path: str, core_mask: int = None, imgsz: int = None,
                 conf: float = None, nms: float = None, classes=None):
        # imgsz обязан совпадать с --imgsz, которым модель экспортировали;
        # класс-атрибуты выше — только значения по умолчанию.
        self.INPUT_SIZE = imgsz or self.INPUT_SIZE
        self.CONF_THRESHOLD = self.CONF_THRESHOLD if conf is None else conf
        self.NMS_THRESHOLD = self.NMS_THRESHOLD if nms is None else nms
        # None = любой класс считается целью (верно для одноклассовой модели).
        # Для многоклассовой передайте список нужных id, иначе детектор будет
        # наводиться на первый попавшийся класс.
        self.classes = None if classes is None else np.asarray(classes)

        self._rknn = RKNNLite()
        if self._rknn.load_rknn(rknn_path) != 0:
            raise RuntimeError(f"load_rknn failed: {rknn_path}")
        if self._rknn.init_runtime(core_mask=core_mask or self.CORE_MASK) != 0:
            raise RuntimeError("init_runtime failed (rknpu driver / librknnrt.so?)")

    def close(self):
        self._rknn.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- inference -------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray):
        """Вход: BGR-кадр. Возврат: (x, y, w, h) в координатах кадра, либо None."""
        canvas, scale, pad_x, pad_y = self._letterbox(frame_bgr)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # uint8 HWC: нормализацию делает сам NPU (mean/std из rknn.config()),
        # поэтому НИКАКОГО /255 и float-приведения тут быть не должно.
        # Форма вызова — как в официальном примере rknn-toolkit-lite2/examples/
        # resnet18 (inputs=[img], без data_format): в lite2 наличие data_format
        # не подтверждено, а хостовый eval_rknn_accuracy.py передаёт его явно
        # уже полному тулкиту, где он точно есть.
        outs = self._rknn.inference(inputs=[rgb])
        outs = [np.asarray(o, dtype=np.float32) for o in outs]

        boxes_xyxy, scores, cls_ids = self._decode(outs, self.INPUT_SIZE)
        keep = scores >= self.CONF_THRESHOLD
        if self.classes is not None:              # многоклассовая модель:
            keep &= np.isin(cls_ids, self.classes)  # берём только нужные классы
        if not keep.any():
            return None
        boxes_xyxy, scores = boxes_xyxy[keep], scores[keep]

        # xyxy -> xywh для cv2.dnn.NMSBoxes (он ждёт именно xywh)
        wh = boxes_xyxy[:, 2:4] - boxes_xyxy[:, 0:2]
        boxes_xywh = np.concatenate([boxes_xyxy[:, 0:2], wh], axis=1)

        idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(),
                               self.CONF_THRESHOLD, self.NMS_THRESHOLD)
        if len(idx) == 0:
            return None
        idx = np.array(idx).flatten()
        best = idx[np.argmax(scores[idx])]

        bx, by, bw, bh = boxes_xywh[best]
        bx = (bx - pad_x) / scale
        by = (by - pad_y) / scale
        bw, bh = bw / scale, bh / scale

        fh, fw = frame_bgr.shape[:2]
        x0, y0 = max(0, int(round(bx))), max(0, int(round(by)))
        x1, y1 = min(fw, int(round(bx + bw))), min(fh, int(round(by + bh)))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    # ---- decode (зеркало eval_rknn_accuracy.py) --------------------------

    @classmethod
    def _decode(cls, outs, size):
        """outs: 3*nl тензоров -> (xyxy [N,4], лучший скор [N], класс [N]).

        Обобщённо по nl (число уровней), nc (число классов) и reg_max —
        все три выводятся из форм тензоров, ничего не захардкожено.
        """
        nl = len(outs) // 3
        boxes, confs = [], []
        for i in range(nl):
            boxes.append(cls._flatten(cls._box_process(outs[3 * i], size)))
            confs.append(cls._flatten(outs[3 * i + 1]))
        c = np.concatenate(confs)
        return np.concatenate(boxes), c.max(axis=-1), c.argmax(axis=-1)

    @staticmethod
    def _flatten(a):
        return a.transpose(0, 2, 3, 1).reshape(-1, a.shape[1])

    @classmethod
    def _box_process(cls, position, size):
        gh, gw = position.shape[2:4]
        col, row = np.meshgrid(np.arange(gw), np.arange(gh))
        grid = np.concatenate((col.reshape(1, 1, gh, gw),
                               row.reshape(1, 1, gh, gw)), 1).astype(np.float32)
        stride = np.array([size // gh, size // gw]).reshape(1, 2, 1, 1)
        d = cls._dfl(position)
        xy1 = (grid + 0.5 - d[:, 0:2]) * stride
        xy2 = (grid + 0.5 + d[:, 2:4]) * stride
        return np.concatenate((xy1, xy2), 1)

    @staticmethod
    def _dfl(position):
        n, c, h, w = position.shape
        mc = c // 4
        y = position.reshape(n, 4, mc, h, w)
        y = y - y.max(axis=2, keepdims=True)   # softmax, численно устойчиво
        y = np.exp(y)
        y /= y.sum(axis=2, keepdims=True)
        acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
        return (y * acc).sum(axis=2)

    # ---- preprocessing (идентичен build_calib.py и yolo_detector.py) -----

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.INPUT_SIZE / w, self.INPUT_SIZE / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), self.PAD, np.uint8)
        px, py = (self.INPUT_SIZE - nw) // 2, (self.INPUT_SIZE - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        return canvas, scale, px, py


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="smoke-test на плате")
    ap.add_argument("--model", default="best_5s_rknn_rk3588_i8.rknn")
    ap.add_argument("--cam-index", type=int, default=0)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="должен совпадать с --imgsz при экспорте (по умолч. 640)")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                    help="id классов-целей для многоклассовой модели")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.cam_index)
    with RknnYoloDetector(a.model, imgsz=a.imgsz, conf=a.conf, classes=a.classes) as det:
        t0, n, hits = time.time(), 0, 0
        while n < a.frames:
            ok, frame = cap.read()
            if not ok:
                break
            box = det.detect(frame)
            n += 1
            hits += box is not None
        dt = time.time() - t0
    cap.release()
    print(f"{n} кадров за {dt:.1f}s -> {n / dt:.1f} FPS (сквозной), детекций {hits}")
