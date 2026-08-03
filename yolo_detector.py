"""YOLO-детектор цели поверх best_5s.onnx (single-class YOLOv8-style export,
output0: [1, 5, 8400] = 4 box coords + 1 class score, no separate objectness).
Возвращает тот же (x, y, w, h)-контракт, что и
PAK_OUK_Controller.detect_target(), чтобы служить ему прямой заменой в
webcam_test.py без изменений остального пайплайна (servo/proximity/course).
"""

import cv2
import numpy as np
import onnxruntime as ort


class YoloDetector:
    INPUT_SIZE = 640
    CONF_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.45

    def __init__(self, onnx_path: str):
        self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, frame_bgr: np.ndarray):
        """Вход: цветной BGR-кадр (cv2.VideoCapture.read()). Возврат: (x, y, w, h) или None."""
        canvas, scale, pad_x, pad_y = self._letterbox(frame_bgr)

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        output = self._session.run(None, {self._input_name: blob})[0]
        preds = output[0].T  # [8400, 5] -> (cx, cy, w, h, conf)

        scores = preds[:, 4]
        keep = scores >= self.CONF_THRESHOLD
        preds = preds[keep]
        scores = scores[keep]
        if preds.shape[0] == 0:
            return None

        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        boxes_xywh = np.stack([cx - w * 0.5, cy - h * 0.5, w, h], axis=1)

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores.tolist(), self.CONF_THRESHOLD, self.NMS_THRESHOLD)
        if len(indices) == 0:
            return None
        indices = np.array(indices).flatten()
        best = indices[np.argmax(scores[indices])]

        bx, by, bw, bh = boxes_xywh[best]
        # letterbox canvas coords -> original frame coords
        bx = (bx - pad_x) / scale
        by = (by - pad_y) / scale
        bw = bw / scale
        bh = bh / scale

        frame_h, frame_w = frame_bgr.shape[:2]
        x0 = max(0, int(round(bx)))
        y0 = max(0, int(round(by)))
        x1 = min(frame_w, int(round(bx + bw)))
        y1 = min(frame_h, int(round(by + bh)))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def _letterbox(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        scale = min(self.INPUT_SIZE / w, self.INPUT_SIZE / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, dtype=np.uint8)
        pad_x = (self.INPUT_SIZE - nw) // 2
        pad_y = (self.INPUT_SIZE - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y
