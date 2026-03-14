"""
Демо-скрипт для проверки структуры приложения без полных зависимостей
"""

import os
from pathlib import Path

def check_structure():
    """Проверка структуры проекта"""
    
    print("=" * 70)
    print("ПРОВЕРКА СТРУКТУРЫ ПРОЕКТА")
    print("=" * 70 + "\n")
    
    # Проверяем ключевые файлы
    files = {
        'app.py': 'Веб-приложение Flask',
        'download_dataset.py': 'Скрипт подготовки датасета',
        'download_from_kaggle.py': 'Загрузка с Kaggle',
        'setup.py': 'Настройка окружения',
        'requirements.txt': 'Python зависимости',
        'README_RU.md': 'Русская документация',
        'QUICKSTART_RU.md': 'Быстрый старт',
        'templates/index.html': 'HTML интерфейс',
        'configs/config.yaml': 'Конфигурация',
    }
    
    all_ok = True
    for file_path, description in files.items():
        if Path(file_path).exists():
            print(f"✓ {file_path:30s} - {description}")
        else:
            print(f"❌ {file_path:30s} - {description} (НЕ НАЙДЕН)")
            all_ok = False
    
    # Проверяем директории
    print("\nДиректории:")
    dirs = ['src', 'src/models', 'src/data', 'configs', 'templates']
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ (НЕ НАЙДЕНА)")
            all_ok = False
    
    # Проверяем Python модули
    print("\nPython модули:")
    modules = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/models/classifier.py',
        'src/models/attention.py',
        'src/data/__init__.py',
        'src/data/dataset.py',
        'src/data/preprocess.py',
        'src/train.py',
        'src/evaluate.py',
    ]
    
    for module_path in modules:
        if Path(module_path).exists():
            print(f"✓ {module_path}")
        else:
            print(f"❌ {module_path} (НЕ НАЙДЕН)")
            all_ok = False
    
    print("\n" + "=" * 70)
    if all_ok:
        print("✅ Структура проекта корректна!")
    else:
        print("⚠ Некоторые файлы отсутствуют")
    print("=" * 70)
    
    return all_ok


def check_gitignore():
    """Проверка .gitignore"""
    print("\n" + "=" * 70)
    print("ПРОВЕРКА .GITIGNORE")
    print("=" * 70 + "\n")
    
    if not Path('.gitignore').exists():
        print("❌ .gitignore не найден")
        return False
    
    with open('.gitignore', 'r') as f:
        content = f.read()
    
    patterns = ['data/', 'checkpoints/', 'uploads/', '*.mp4', '__pycache__/']
    all_ok = True
    
    for pattern in patterns:
        if pattern in content:
            print(f"✓ {pattern}")
        else:
            print(f"⚠ {pattern} отсутствует в .gitignore")
            all_ok = False
    
    return all_ok


def show_next_steps():
    """Показать следующие шаги"""
    print("\n" + "=" * 70)
    print("СЛЕДУЮЩИЕ ШАГИ")
    print("=" * 70 + "\n")
    
    print("1. Настройка окружения:")
    print("   python setup.py")
    print()
    print("2. Подготовка датасета:")
    print("   python download_dataset.py")
    print()
    print("3. Запуск веб-приложения:")
    print("   python app.py")
    print()
    print("4. Или обучение модели:")
    print("   python src/train.py --config configs/config.yaml")
    print()
    print("Подробнее см. README_RU.md и QUICKSTART_RU.md")
    print()


def main():
    """Главная функция"""
    
    # Проверяем, что мы в правильной директории
    if not Path('app.py').exists():
        print("❌ Запустите скрипт из корневой директории проекта")
        return False
    
    structure_ok = check_structure()
    gitignore_ok = check_gitignore()
    
    if structure_ok:
        show_next_steps()
    
    return structure_ok and gitignore_ok


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
