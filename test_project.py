#!/usr/bin/env python3
"""
Тестовый скрипт для проверки основных функций без запуска полного сервера
"""

import sys
from pathlib import Path

def test_file_structure():
    """Тест структуры файлов"""
    print("Тест 1: Структура файлов")
    required_files = [
        'app.py',
        'templates/index.html',
        'requirements.txt',
        'README_RU.md',
        'QUICKSTART_RU.md'
    ]
    
    all_ok = True
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ❌ {file} не найден")
            all_ok = False
    
    return all_ok


def test_syntax():
    """Тест синтаксиса Python файлов"""
    print("\nТест 2: Синтаксис Python")
    
    python_files = [
        'app.py',
        'download_dataset.py',
        'setup.py',
        'check_project.py'
    ]
    
    all_ok = True
    for file in python_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, file, 'exec')
            print(f"  ✓ {file}")
        except SyntaxError as e:
            print(f"  ❌ {file}: {e}")
            all_ok = False
        except Exception as e:
            print(f"  ⚠ {file}: {e}")
    
    return all_ok


def test_html():
    """Тест HTML файлов"""
    print("\nТест 3: HTML файлы")
    
    html_file = 'templates/index.html'
    if not Path(html_file).exists():
        print(f"  ❌ {html_file} не найден")
        return False
    
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ('<!DOCTYPE html>', 'DOCTYPE объявлен'),
        ('<html', 'HTML тег присутствует'),
        ('upload', 'Функция upload присутствует'),
        ('fetch', 'API fetch используется'),
        ('.result', 'CSS класс результата'),
    ]
    
    all_ok = True
    for check, description in checks:
        if check in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ⚠ {description} не найден")
            all_ok = False
    
    return all_ok


def test_documentation():
    """Тест документации"""
    print("\nТест 4: Документация")
    
    docs = [
        ('README_RU.md', ['Установка', 'Запуск', 'веб-приложение']),
        ('QUICKSTART_RU.md', ['Быстрый старт', 'pip install', 'python app.py']),
    ]
    
    all_ok = True
    for doc_file, keywords in docs:
        if not Path(doc_file).exists():
            print(f"  ❌ {doc_file} не найден")
            all_ok = False
            continue
        
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read().lower()
        
        doc_ok = all(keyword.lower() in content for keyword in keywords)
        if doc_ok:
            print(f"  ✓ {doc_file} содержит необходимые разделы")
        else:
            print(f"  ⚠ {doc_file} может быть неполным")
            all_ok = False
    
    return all_ok


def test_gitignore():
    """Тест .gitignore"""
    print("\nТест 5: .gitignore")
    
    if not Path('.gitignore').exists():
        print("  ❌ .gitignore не найден")
        return False
    
    with open('.gitignore', 'r') as f:
        content = f.read()
    
    patterns = ['data/', 'checkpoints/', 'uploads/', '*.mp4', '*.pyc']
    all_ok = True
    
    for pattern in patterns:
        if pattern in content:
            print(f"  ✓ {pattern}")
        else:
            print(f"  ⚠ {pattern} не в .gitignore")
            all_ok = False
    
    return all_ok


def test_requirements():
    """Тест requirements.txt"""
    print("\nТест 6: requirements.txt")
    
    if not Path('requirements.txt').exists():
        print("  ❌ requirements.txt не найден")
        return False
    
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    packages = ['torch', 'flask', 'opencv-python', 'mediapipe', 'timm']
    all_ok = True
    
    for package in packages:
        if package in content.lower():
            print(f"  ✓ {package}")
        else:
            print(f"  ❌ {package} не в requirements.txt")
            all_ok = False
    
    return all_ok


def main():
    """Запуск всех тестов"""
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ПРОЕКТА DEEPFAKE DETECTION")
    print("=" * 70 + "\n")
    
    tests = [
        ("Структура файлов", test_file_structure),
        ("Синтаксис Python", test_syntax),
        ("HTML файлы", test_html),
        ("Документация", test_documentation),
        (".gitignore", test_gitignore),
        ("requirements.txt", test_requirements),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  ❌ Ошибка в тесте: {e}")
            results.append((test_name, False))
    
    # Итоговые результаты
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 70 + "\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ НЕ ПРОЙДЕН"
        print(f"{test_name:30s} {status}")
    
    print(f"\nИтого: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("\n✅ Все тесты пройдены успешно!")
        print("\nПроект готов к использованию.")
        print("Запустите: python app.py")
    else:
        print("\n⚠ Некоторые тесты не прошли.")
        print("Проверьте ошибки выше.")
    
    print("\n" + "=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
