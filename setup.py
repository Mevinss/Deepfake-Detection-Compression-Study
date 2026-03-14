#!/usr/bin/env python3
"""
Скрипт для настройки и проверки окружения
"""

import sys
import subprocess
import os

def check_python_version():
    """Проверка версии Python"""
    print("Проверка версии Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python {version.major}.{version.minor} слишком старый")
        print(f"   Требуется Python 3.9 или выше")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_ffmpeg():
    """Проверка наличия ffmpeg"""
    print("\nПроверка ffmpeg...")
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ {version_line}")
            return True
        else:
            print("❌ ffmpeg не найден")
            return False
    except FileNotFoundError:
        print("❌ ffmpeg не установлен")
        print("\nУстановите ffmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: Скачайте с https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"⚠ Ошибка при проверке ffmpeg: {e}")
        return False


def install_dependencies():
    """Установка Python зависимостей"""
    print("\n" + "=" * 70)
    print("УСТАНОВКА ЗАВИСИМОСТЕЙ")
    print("=" * 70 + "\n")
    
    try:
        print("Установка зависимостей из requirements.txt...")
        print("Это может занять несколько минут...\n")
        
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            check=True
        )
        
        print("\n✓ Все зависимости установлены успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка при установке зависимостей: {e}")
        return False


def check_cuda():
    """Проверка доступности CUDA"""
    print("\n" + "=" * 70)
    print("ПРОВЕРКА GPU")
    print("=" * 70 + "\n")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA доступна")
            print(f"  Устройство: {torch.cuda.get_device_name(0)}")
            print(f"  Версия CUDA: {torch.version.cuda}")
            print(f"  Количество GPU: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠ CUDA недоступна")
            print("  Модель будет работать на CPU (медленнее)")
            return False
    except ImportError:
        print("⚠ PyTorch не установлен, невозможно проверить CUDA")
        return False


def create_directories():
    """Создание необходимых директорий"""
    print("\n" + "=" * 70)
    print("СОЗДАНИЕ ДИРЕКТОРИЙ")
    print("=" * 70 + "\n")
    
    dirs = ['data', 'checkpoints', 'uploads', 'runs', 'results']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ {dir_name}/")
    
    return True


def test_imports():
    """Тестирование импортов"""
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ИМПОРТОВ")
    print("=" * 70 + "\n")
    
    modules = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('flask', 'Flask'),
        ('mediapipe', 'MediaPipe'),
        ('albumentations', 'Albumentations'),
        ('timm', 'timm'),
        ('yaml', 'PyYAML'),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"❌ {display_name}")
            all_ok = False
    
    return all_ok


def main():
    """Главная функция"""
    print("\n" + "=" * 70)
    print("НАСТРОЙКА ОКРУЖЕНИЯ ДЛЯ ДЕТЕКЦИИ ДИПФЕЙКОВ")
    print("=" * 70 + "\n")
    
    # Проверка Python
    if not check_python_version():
        return False
    
    # Проверка ffmpeg
    ffmpeg_ok = check_ffmpeg()
    
    # Установка зависимостей
    print("\nУстановить зависимости Python? (y/n): ", end='')
    response = input().lower().strip()
    
    if response == 'y':
        if not install_dependencies():
            return False
    else:
        print("Пропуск установки зависимостей")
    
    # Создание директорий
    create_directories()
    
    # Проверка импортов
    imports_ok = test_imports()
    
    # Проверка CUDA
    check_cuda()
    
    # Итоговый результат
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТ")
    print("=" * 70 + "\n")
    
    if imports_ok and ffmpeg_ok:
        print("✅ Окружение настроено успешно!")
        print("\nТеперь вы можете:")
        print("  1. Подготовить датасет: python download_dataset.py")
        print("  2. Запустить веб-приложение: python app.py")
        print("  3. Обучить модель: python src/train.py --config configs/config.yaml")
    elif imports_ok:
        print("⚠ Окружение настроено, но ffmpeg не установлен")
        print("  Установите ffmpeg для работы с видео")
    else:
        print("❌ Не все компоненты установлены")
        print("  Проверьте ошибки выше и установите недостающие компоненты")
    
    print("\n" + "=" * 70)
    return imports_ok and ffmpeg_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
