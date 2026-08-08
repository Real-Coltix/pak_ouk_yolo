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

## RK3588 / RKNN (INT8 на NPU)

Отдельная host-side ветка пайплайна: конвертация `best_5s.pt` в `.rknn` для
NPU Rockchip (RK3588/RK3588S, 3 ядра, 6 TOPS). Ставится в **отдельное**
окружение — см. шапку `requirements-rknn.txt` (там же причины всех пинов:
Python ≤3.12, `onnx==1.16.1`, `numpy<=1.26.4`, torch с CPU-индекса).

```bash
python export_rknn_onnx.py --weights /root/best_5s.pt   # .pt  -> best_5s_rknn.onnx
python verify_rknn_onnx.py                              # сверка с best_5s.onnx
python build_calib.py --n 300 --seed 0                  # calib/*.npy + dataset.txt
python convert_rknn.py --platform rk3588 --dtype i8     # -> best_5s_rknn_rk3588_i8.rknn
```

Экспорт даёт **не** тот же граф, что `best_5s.onnx`: RKNN-вариант отдаёт 9
выходов (3 уровня × `box_raw` / `cls_sigmoid` / `cls_sum`) без DFL и декода
внутри графа — декод делается на CPU (`decode()` в `eval_rknn_accuracy.py`).
Так требует RKNN-ветка экспорта airockchip; в обмен весь backbone уходит на
NPU целиком. `mean/std = 0/255` заданы в `rknn.config()`, поэтому на плату
подаются сырые uint8-пиксели, а нормализацию делает сам NPU.

Калибровка (`build_calib.py`) идёт **тем же letterbox'ом**, что и инференс
(aspect-preserved + серые поля 114, RGB), а не resize'ом, который сделал бы
загрузчик RKNN по умолчанию — иначе диапазоны квантования меряются на другом
распределении, чем видит продакшн. Подробности и грабли — в докстринге файла.

### Что стоило INT8-квантование

`eval_rknn_accuracy.py` гоняет `.rknn` в симуляторе toolkit'а (бит-точный по
арифметике, но **не** измеряет FPS — для этого нужна плата) и сравнивает с
FP32 ONNX на тех же кадрах. 300 калибровочных изображений (seed 0),
`quantized_algorithm=normal`, 60 оценочных кадров, порог 0.25:

| метрика | значение |
|---|---|
| кадров с детекцией | fp32 53 / int8 54 (оба 52) |
| int8 потерял детекций | **1 / 53** |
| int8 добавил детекций | 2 (fp32 чуть ниже порога) |
| сдвиг топ-скора | mean 0.0425, worst 0.2563 |
| cosine score-карт (на кадрах, где сработал fp32) | mean 0.98844, worst 0.81687 |
| IoU топ-бокса | mean 0.9602, worst 0.7752 |
| кадров с IoU < 0.5 | **0 / 52** |

То есть локализация не деградировала вообще, а единственная потеря — кадр,
где fp32 и сам был на грани порога (0.330 → 0.196). Для approach-and-stop
задачи (см. `../ssc338q-opencv-project`) это приемлемо: цель ведётся много
кадров подряд, одиночный промах гасится фильтром.

Замерен только `--algorithm normal`. `mmse` (обычно точнее, но кратно дороже
по памяти при сборке) **не проверялся**: сборка с ним не влезает в 3 ГБ RAM
этого хоста и роняет машину по OOM. Если будет хост пожирнее — сравнивать
через `--algorithm mmse` и тот же `eval_rknn_accuracy.py`.

`--json` сохраняет пофреймовые записи, `--from-json` переотчитывает их без
повторной (многоминутной) сборки квантованного графа.
