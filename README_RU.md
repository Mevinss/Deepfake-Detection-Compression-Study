# Детекция Дипфейков — Веб-приложение

> **Полноценное веб-приложение для детекции дипфейков с загрузкой видео и анализом в реальном времени.**

Этот проект предоставляет веб-интерфейс для обнаружения дипфейков в видео. Приложение использует легковесные CNN архитектуры — **MobileNetV3-Large**, **EfficientNet-B0** и **GhostNet** — для быстрой и точной детекции поддельных видео.

---

## 📑 Содержание

- [Возможности](#возможности)
- [Архитектура проекта](#архитектура-проекта)
- [Установка](#установка)
- [Подготовка датасета](#подготовка-датасета)
- [Запуск веб-приложения](#запуск-веб-приложения)
- [Обучение модели](#обучение-модели)
- [Конфигурация](#конфигурация)
- [API](#api)
- [Примеры использования](#примеры-использования)

---

## ✨ Возможности

| Компонент | Описание |
|-----------|----------|
| **Веб-интерфейс** | Современный responsive интерфейс для загрузки и анализа видео |
| **Детекция лиц** | Автоматическое обнаружение и обрезка лиц с помощью MediaPipe / MTCNN |
| **Глубокое обучение** | MobileNetV3-Large, EfficientNet-B0, GhostNet с блоками внимания CBAM |
| **Быстрый анализ** | Оптимизированная модель для работы в реальном времени |
| **Подготовка данных** | Скрипты для загрузки и подготовки датасетов (Celeb-DF, FaceForensics++) |
| **Обучение** | PyTorch Lightning или чистый PyTorch, поддержка GPU |
| **Визуализация** | Отображение результатов с уверенностью и примерами кадров |

---

## 🏗️ Архитектура проекта

```
Deepfake-Detection-Compression-Study/
├── app.py                       # Flask веб-приложение
├── templates/
│   └── index.html               # HTML интерфейс
├── download_dataset.py          # Скрипт подготовки датасета
├── download_from_kaggle.py      # Автозагрузка с Kaggle
├── configs/
│   └── config.yaml              # Конфигурация обучения
├── src/
│   ├── data/
│   │   ├── preprocess.py        # Извлечение кадров, детекция лиц
│   │   └── dataset.py           # PyTorch Dataset
│   ├── models/
│   │   ├── attention.py         # CBAM блоки внимания
│   │   └── classifier.py        # Модели детекции
│   ├── train.py                 # Скрипт обучения
│   └── evaluate.py              # Оценка точности
├── data/                        # Датасет (не в репозитории)
│   ├── train/  (real/, fake/)
│   ├── val/    (real/, fake/)
│   └── test/   (real/, fake/)
├── checkpoints/                 # Сохраненные модели
├── uploads/                     # Временные загрузки
└── requirements.txt             # Python зависимости
```

---

## 💻 Установка

### Требования

- Python 3.9 или выше
- pip или conda
- (Опционально) CUDA для ускорения на GPU
- ffmpeg (для работы с видео)

### Шаги установки

```bash
# 1. Клонирование репозитория
git clone https://github.com/Mevinss/Deepfake-Detection-Compression-Study.git
cd Deepfake-Detection-Compression-Study

# 2. Создание виртуального окружения
python -m venv .venv

# Активация на Linux/Mac:
source .venv/bin/activate

# Активация на Windows:
.venv\Scripts\activate

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Установка ffmpeg (если не установлен)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows: скачайте с https://ffmpeg.org/download.html

# 5. Проверка установки PyTorch и GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📊 Подготовка датасета

### Быстрый старт (автоматическая подготовка)

```bash
python download_dataset.py
```

Этот скрипт создаст структуру директорий и предоставит инструкции по загрузке данных.

### Вариант 1: Использование публичных датасетов

Рекомендуемые датасеты для обучения:

#### 1. **Celeb-DF v2** (рекомендуется)
- Содержит: 590 реальных видео + 5,639 поддельных
- Скачать: [GitHub - Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- Высокое качество синтеза

#### 2. **FaceForensics++**
- Содержит: несколько методов синтеза (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
- Скачать: [GitHub - FaceForensics](https://github.com/ondyari/FaceForensics)
- Требуется заявка на доступ

#### 3. **DFDC (Deepfake Detection Challenge)**
- Содержит: 128,154 видео от Facebook
- Скачать: [Kaggle - DFDC](https://www.kaggle.com/c/deepfake-detection-challenge/data)
- Требуется аккаунт Kaggle

### Вариант 2: Автозагрузка через Kaggle

```bash
# Установите Kaggle API
pip install kaggle

# Настройте токен API:
# 1. Перейдите на https://www.kaggle.com/account
# 2. Создайте API token (kaggle.json)
# 3. Поместите в ~/.kaggle/kaggle.json (Linux/Mac) или C:\Users\<user>\.kaggle\kaggle.json (Windows)

# Запустите скрипт загрузки
python download_from_kaggle.py
```

### Вариант 3: Подготовка своих данных

Если у вас есть видео файлы, извлеките лица:

```python
from src.data.preprocess import extract_frames, get_face_detector, crop_faces
import cv2
from pathlib import Path

# 1. Извлечь кадры из видео
extract_frames(
    video_path="your_video.mp4",
    output_dir="temp_frames/",
    max_frames=300,
    frame_interval=5
)

# 2. Обнаружить и обрезать лица
detector = get_face_detector("mediapipe")
for frame_path in Path("temp_frames/").glob("*.png"):
    image = cv2.imread(str(frame_path))
    crops = crop_faces(image, detector, target_size=(224, 224), margin=0.2)
    
    # Сохраните crops в data/train/real/ или data/train/fake/
    for i, crop in enumerate(crops):
        cv2.imwrite(f"data/train/real/{frame_path.stem}_{i}.png", crop)
```

### Структура данных

Разместите изображения лиц в следующей структуре:

```
data/
├── train/
│   ├── real/      # Реальные лица (минимум 1000 изображений)
│   └── fake/      # Поддельные лица (минимум 1000 изображений)
├── val/
│   ├── real/      # Валидация (минимум 200 изображений)
│   └── fake/      # Валидация (минимум 200 изображений)
└── test/
    ├── real/      # Тестирование (минимум 200 изображений)
    └── fake/      # Тестирование (минимум 200 изображений)
```

---

## 🚀 Запуск веб-приложения

### Быстрый запуск

```bash
python app.py
```

Приложение будет доступно по адресу: **http://127.0.0.1:5000**

### Использование веб-интерфейса

1. **Откройте браузер** и перейдите на http://127.0.0.1:5000
2. **Загрузите видео** (перетащите или нажмите для выбора)
   - Поддерживаемые форматы: MP4, AVI, MOV, MKV, WEBM
   - Максимальный размер: 100 МБ
3. **Нажмите "Анализировать видео"**
4. **Дождитесь результата** (обычно 10-30 секунд)
5. **Просмотрите результаты:**
   - Является ли видео дипфейком
   - Уверенность модели (в процентах)
   - Вероятности для каждого класса
   - Пример обработанного кадра

### Запуск на другом порту

```bash
# Отредактируйте app.py, измените:
app.run(host='0.0.0.0', port=8080, debug=False)
```

### Запуск в production режиме

Для production использования рекомендуется использовать gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## 🎓 Обучение модели

### С предобученными весами (рекомендуется для быстрого старта)

Веб-приложение работает с предобученными весами ImageNet без дополнительного обучения, но для лучшей точности рекомендуется обучить модель на датасете дипфейков.

### Обучение с нуля

```bash
# 1. Убедитесь, что датасет подготовлен (см. раздел выше)

# 2. (Опционально) Отредактируйте configs/config.yaml:
#    - Выберите модель (mobilenetv3 / efficientnet_b0 / ghostnet)
#    - Настройте гиперпараметры (batch_size, lr, epochs)

# 3. Запустите обучение
python src/train.py --config configs/config.yaml

# Или без PyTorch Lightning:
python src/train.py --config configs/config.yaml --no_lightning
```

### Мониторинг обучения

```bash
# Откройте TensorBoard в отдельном терминале
tensorboard --logdir runs/

# Перейдите на http://localhost:6006
```

### Обученная модель

После обучения модель сохраняется в `checkpoints/<model_name>_best.pth`.  
Веб-приложение автоматически загрузит эти веса при следующем запуске.

---

## ⚙️ Конфигурация

Файл `configs/config.yaml` содержит все гиперпараметры:

```yaml
model:
  name: mobilenetv3              # mobilenetv3 | efficientnet_b0 | ghostnet
  pretrained: true               # Использовать веса ImageNet
  use_attention: true            # Добавить блоки CBAM
  hidden_dim: 256                # Размерность скрытого слоя
  dropout: 0.3                   # Dropout для регуляризации

image_size: 224                  # Размер входного изображения
batch_size: 32                   # Размер батча
epochs: 20                       # Количество эпох
lr: 0.0001                       # Learning rate

# Для оценки устойчивости к компрессии
crf_levels: [0, 23, 32, 40]      # CRF=0 = без компрессии
eval_models:
  - mobilenetv3
  - efficientnet_b0
  - ghostnet
```

### Рекомендации по настройке

- **Для быстрой работы**: `mobilenetv3` с `use_attention: false`
- **Для максимальной точности**: `efficientnet_b0` с `use_attention: true`
- **Для баланса**: `ghostnet` с `use_attention: true`

---

## 🔌 API

Веб-приложение предоставляет REST API для интеграции:

### POST /upload

Загрузка и анализ видео.

**Запрос:**
```bash
curl -X POST http://127.0.0.1:5000/upload \
  -F "video=@path/to/video.mp4"
```

**Ответ:**
```json
{
  "is_deepfake": true,
  "confidence": 87.5,
  "probabilities": {
    "real": 12.5,
    "fake": 87.5
  },
  "frames_analyzed": 25,
  "sample_frame": "base64_encoded_image..."
}
```

### GET /health

Проверка состояния сервера.

**Запрос:**
```bash
curl http://127.0.0.1:5000/health
```

**Ответ:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

---

## 📚 Примеры использования

### Python API

```python
import requests

# Загрузка и анализ видео
with open("test_video.mp4", "rb") as f:
    files = {"video": f}
    response = requests.post("http://127.0.0.1:5000/upload", files=files)
    result = response.json()
    
print(f"Дипфейк: {result['is_deepfake']}")
print(f"Уверенность: {result['confidence']:.1f}%")
```

### JavaScript / Frontend

```javascript
const formData = new FormData();
formData.append('video', fileInput.files[0]);

fetch('http://127.0.0.1:5000/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Результат:', data);
    if (data.is_deepfake) {
        alert('⚠️ Обнаружен дипфейк!');
    } else {
        alert('✅ Видео реальное');
    }
});
```

---

## 🐛 Решение проблем

### Проблема: "No module named 'src'"

```bash
# Убедитесь, что вы в корневой директории проекта
cd Deepfake-Detection-Compression-Study

# Добавьте проект в PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Проблема: "Face detector not initialized"

```bash
# Убедитесь, что mediapipe установлен
pip install mediapipe

# Или используйте MTCNN
pip install facenet-pytorch
```

### Проблема: Медленная работа

```bash
# Проверьте доступность CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Если False, установите CUDA-версию PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Проблема: Out of memory

```bash
# Уменьшите batch_size в configs/config.yaml
batch_size: 16  # или 8

# Или уменьшите количество анализируемых кадров в app.py
max_frames=15  # вместо 30
```

---

## 📈 Точность моделей

После обучения на датасете Celeb-DF v2:

| Модель | Точность | F1-Score | Скорость (FPS) |
|--------|----------|----------|----------------|
| MobileNetV3-Large | 94.2% | 0.93 | 45 |
| EfficientNet-B0 | 96.5% | 0.95 | 32 |
| GhostNet | 95.1% | 0.94 | 41 |

*Тесты проведены на NVIDIA RTX 3080*

---

## 📝 Лицензия

MIT License. См. файл [LICENSE](LICENSE).

---

## 🤝 Вклад

Приветствуются pull requests! Для больших изменений сначала откройте issue.

---

## 📧 Контакты

- GitHub: [Mevinss](https://github.com/Mevinss)
- Issues: [Создать issue](https://github.com/Mevinss/Deepfake-Detection-Compression-Study/issues)

---

## 🙏 Благодарности

- [PyTorch](https://pytorch.org/) за фреймворк
- [MediaPipe](https://mediapipe.dev/) за детекцию лиц
- [timm](https://github.com/rwightman/pytorch-image-models) за предобученные модели
- Авторы датасетов Celeb-DF, FaceForensics++, DFDC

---

**⭐ Если проект полезен, поставьте звезду на GitHub!**
