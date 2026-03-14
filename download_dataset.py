"""
Скрипт для загрузки и подготовки датасета для обучения детекции дипфейков.

Использует Celeb-DF v2 датасет через Kaggle API.
Альтернативно, можно использовать небольшой тестовый датасет из открытых источников.
"""

import os
import zipfile
from pathlib import Path
import urllib.request
import shutil
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Progress bar для загрузки файлов"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Загрузка файла с progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def prepare_sample_dataset():
    """
    Создает минимальный демо-датасет для быстрого тестирования.
    В реальном проекте следует использовать полноценные датасеты:
    - FaceForensics++
    - Celeb-DF v2
    - DFDC (Deepfake Detection Challenge)
    """
    
    print("=" * 70)
    print("Подготовка демо-датасета для обучения модели детекции дипфейков")
    print("=" * 70)
    
    # Создаем структуру директорий
    data_dir = Path("data")
    for split in ["train", "val", "test"]:
        for class_name in ["real", "fake"]:
            (data_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    print("\n✓ Структура директорий создана:")
    print("  data/")
    print("    ├── train/ (real/, fake/)")
    print("    ├── val/   (real/, fake/)")
    print("    └── test/  (real/, fake/)")
    
    # Создаем README с инструкциями по добавлению данных
    readme_content = """# Датасет для детекции дипфейков

## Структура

```
data/
├── train/
│   ├── real/  # Реальные изображения лиц
│   └── fake/  # Поддельные (deepfake) изображения
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

## Как добавить данные

### Вариант 1: Использовать публичные датасеты

Рекомендуемые датасеты:

1. **Celeb-DF v2** - содержит 590 настоящих видео и 5639 поддельных
   - Скачать: https://github.com/yuezunli/celeb-deepfakeforensics

2. **FaceForensics++** - большой датасет с различными методами генерации
   - Скачать: https://github.com/ondyari/FaceForensics

3. **DFDC (Deepfake Detection Challenge)** - датасет от Facebook
   - Скачать: https://www.kaggle.com/c/deepfake-detection-challenge/data

### Вариант 2: Использовать скрипт для извлечения кадров

После загрузки видео, используйте наши утилиты для извлечения лиц:

```python
from src.data.preprocess import extract_frames, get_face_detector, crop_faces
import cv2
from pathlib import Path

# 1. Извлечь кадры из видео
extract_frames("video.mp4", "frames/", max_frames=300)

# 2. Детектировать и обрезать лица
detector = get_face_detector("mediapipe")
for frame_path in Path("frames/").glob("*.png"):
    image = cv2.imread(str(frame_path))
    crops = crop_faces(image, detector, target_size=(224, 224))
    # Сохранить crops в соответствующую директорию (real/ или fake/)
```

### Вариант 3: Быстрый старт с готовыми изображениями

Если у вас есть готовые изображения лиц:
1. Поместите реальные изображения в `train/real/`, `val/real/`, `test/real/`
2. Поместите фейковые изображения в `train/fake/`, `val/fake/`, `test/fake/`

Рекомендуемое распределение:
- train: 70% данных
- val: 15% данных  
- test: 15% данных

## Минимальные требования

Для базового обучения рекомендуется:
- Минимум 1000 изображений для каждого класса в train
- Минимум 200 изображений для каждого класса в val
- Минимум 200 изображений для каждого класса в test

## Автоматическая загрузка через Kaggle

Если у вас есть аккаунт Kaggle:

```bash
# Установите Kaggle API
pip install kaggle

# Настройте API токен
# https://www.kaggle.com/docs/api

# Загрузите датасет (например, небольшой тестовый датасет)
kaggle datasets download -d dagnelies/deepfake-faces
unzip deepfake-faces.zip -d data/
```

## Демо данные

Для быстрого тестирования можно использовать несколько изображений.
Веб-приложение будет работать с предобученной моделью даже без 
полноценного обучения на большом датасете.
"""
    
    with open(data_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("\n✓ Создан data/README.md с инструкциями по добавлению данных")
    
    # Создаем скрипт для загрузки примера датасета с Kaggle
    download_script = """#!/usr/bin/env python3
\"\"\"
Скрипт для загрузки тестового датасета через Kaggle API
Требует установленного kaggle и настроенного API токена
\"\"\"

import os
import subprocess
import sys

def download_from_kaggle():
    try:
        import kaggle
    except ImportError:
        print("Kaggle API не установлен. Установите: pip install kaggle")
        return False
    
    # Проверка конфигурации
    kaggle_config = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_config):
        print("Kaggle API токен не настроен!")
        print("1. Перейдите на https://www.kaggle.com/account")
        print("2. Создайте новый API токен")
        print(f"3. Сохраните kaggle.json в {kaggle_config}")
        return False
    
    print("Загрузка датасета с Kaggle...")
    
    # Загружаем небольшой тестовый датасет
    datasets = [
        "dagnelies/deepfake-faces",
        # Можно добавить другие датасеты
    ]
    
    for dataset in datasets:
        try:
            print(f"\\nЗагрузка {dataset}...")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset, "-p", "data/temp"],
                check=True
            )
            print(f"✓ {dataset} загружен")
        except subprocess.CalledProcessError as e:
            print(f"✗ Ошибка при загрузке {dataset}: {e}")
    
    return True

if __name__ == "__main__":
    success = download_from_kaggle()
    if not success:
        print("\\nНе удалось загрузить данные через Kaggle.")
        print("Пожалуйста, загрузите датасет вручную согласно data/README.md")
        sys.exit(1)
"""
    
    with open("download_from_kaggle.py", "w", encoding="utf-8") as f:
        f.write(download_script)
    
    os.chmod("download_from_kaggle.py", 0o755)
    
    print("\n✓ Создан download_from_kaggle.py для автоматической загрузки")
    
    print("\n" + "=" * 70)
    print("СЛЕДУЮЩИЕ ШАГИ:")
    print("=" * 70)
    print("\n1. Загрузите датасет одним из способов:")
    print("   - Автоматически через Kaggle: python download_from_kaggle.py")
    print("   - Вручную из публичных источников (см. data/README.md)")
    print("   - Используйте свои видео и утилиты извлечения кадров")
    print("\n2. После добавления данных, обучите модель:")
    print("   python src/train.py --config configs/config.yaml")
    print("\n3. Или запустите веб-приложение с предобученной моделью:")
    print("   python app.py")
    print("\n" + "=" * 70)


def main():
    """Главная функция"""
    prepare_sample_dataset()
    
    print("\n💡 СОВЕТ: Для быстрого старта можно использовать веб-приложение")
    print("   без полного обучения. Оно будет использовать предобученные веса.")


if __name__ == "__main__":
    main()
