# PAK_OUK + YOLO детекция (веб-камера, Windows/host)

Тот же `PAK_OUK_Controller` (servo-ошибка / proximity / курс / диагностика,
1:1 порт `../ssc338q-opencv-project/src/pak_ouk_controller.hpp`), но модуль
детекции (`detect_target()`, adaptive-threshold + contours) заменён на
YOLO-инференс (`best_5s.onnx`, single-class экспорт) через `yolo_detector.py`.
Остальной пайплайн не менялся — `YoloDetector.detect()` отдаёт тот же
контракт `(x, y, w, h) | None`, что и `detect_target()`, и подключается к
`compute_servo_error`/`check_proximity`/`compute_course_correction` без
изменений.

`best_5s.onnx`: вход `images` `[1, 3, 640, 640]` (letterbox до 640×640, BGR->RGB,
`/255`), выход `output0` `[1, 5, 8400]` = 4 координаты bbox (cx, cy, w, h) + 1
class score (объектность отдельно не идёт — экспорт с одним классом).
Постобработка: порог по confidence (`YoloDetector.CONF_THRESHOLD = 0.25`),
`cv2.dnn.NMSBoxes`, затем лучший бокс размаппливается из letterbox-координат
обратно в координаты исходного кадра.

## Запуск

```bash
pip install -r requirements.txt
python webcam_test.py --cam-index 0 --model best_5s.onnx
```

Опции те же, что у классического `webcam_test.py` в `windows_webcam_test/`:
`--cam-index` (по умолч. `0`), `--critical-area-fraction` (по умолч. `0.35`).
В окне предпросмотра — bbox цели, отфильтрованный центроид, строка телеметрии
(`det`, `V`, `w` (omega), `stop`, `fps`, статус диагностики). `q` — выход.

## Проверено (без реальной камеры/GPU — только этот sandbox)

`self_test.py` прогоняет `YoloDetector` на синтетических кадрах (чистый шум,
серый фон с ярким прямоугольником) и через полный `PAK_OUK_Controller`-пайплайн
— проверяет, что интеграция не падает и что декодирование выхода модели дает
разумные (не NaN/не вне диапазона) confidence-скоры. Это **не** проверка
качества детекции — модель обучена на реальном таргете, синтетический
прямоугольник её не активирует (max score ~0.0007 на тестовом кадре), это
ожидаемо и не баг. Реальная проверка качества детекции возможна только на
живых кадрах с камеры "ELE01" на Windows-машине пользователя (как и для
классического `detect_target()` — см. `windows_webcam_test/README.md`).

```bash
python self_test.py
```
