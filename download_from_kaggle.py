#!/usr/bin/env python3
"""
Скрипт для загрузки тестового датасета через Kaggle API
Требует установленного kaggle и настроенного API токена
"""

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
            print(f"\nЗагрузка {dataset}...")
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
        print("\nНе удалось загрузить данные через Kaggle.")
        print("Пожалуйста, загрузите датасет вручную согласно data/README.md")
        sys.exit(1)
