"""
Веб-приложение для детекции дипфейков
Flask приложение с загрузкой видео и детекцией
"""

import os
import io
import sys
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tempfile

# Try to import dependencies
try:
    import cv2
    import torch
    import numpy as np
    import yaml
    from PIL import Image
    import base64
    from src.models.classifier import build_model
    from src.data.preprocess import extract_frames, get_face_detector, crop_faces
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"⚠ Не все зависимости установлены: {e}")
    print("Запустите: pip install -r requirements.txt")
    DEPENDENCIES_OK = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Глобальные переменные для модели
model = None
face_detector = None
device = None
config = None


def allowed_file(filename):
    """Проверка разрешенных расширений файлов"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model():
    """Загрузка модели детекции дипфейков"""
    global model, face_detector, device, config
    
    if not DEPENDENCIES_OK:
        print("⚠ Зависимости не установлены. Установите их с помощью:")
        print("  pip install -r requirements.txt")
        return False
    
    try:
        # Определяем устройство (GPU или CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {device}")
        
        # Загружаем конфигурацию
        config_path = Path("configs/config.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            # Конфигурация по умолчанию
            config = {
                'model': {
                    'name': 'mobilenetv3',
                    'pretrained': True,
                    'use_attention': True,
                    'hidden_dim': 256,
                    'dropout': 0.3
                },
                'image_size': 224
            }
        
        # Создаем модель
        print("Загрузка модели...")
        model = build_model(
            model_name=config['model']['name'],
            pretrained=config['model']['pretrained'],
            use_attention=config['model']['use_attention'],
            hidden_dim=config['model']['hidden_dim'],
            dropout=config['model']['dropout']
        )
        model = model.to(device)
        
        # Пытаемся загрузить обученные веса, если они есть
        checkpoint_path = Path(f"checkpoints/{config['model']['name']}_best.pth")
        if checkpoint_path.exists():
            print(f"Загрузка весов из {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✓ Веса успешно загружены")
        else:
            print("⚠ Обученные веса не найдены. Используются предобученные веса ImageNet.")
            print("  Для лучшей точности обучите модель: python src/train.py --config configs/config.yaml")
        
        model.eval()
        
        # Загружаем детектор лиц
        print("Загрузка детектора лиц...")
        face_detector = get_face_detector("mediapipe")
        print("✓ Детектор лиц загружен")
        
        print("=" * 70)
        print("Модель готова к работе!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        traceback.print_exc()
        return False


def process_video(video_path):
    """
    Обработка видео и определение, является ли оно дипфейком
    
    Returns:
        dict: {
            'is_deepfake': bool,
            'confidence': float,
            'probabilities': {'real': float, 'fake': float},
            'frames_analyzed': int,
            'sample_frame': base64 encoded image
        }
    """
    try:
        # 1. Извлечение кадров
        print(f"Обработка видео: {video_path}")
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_paths = extract_frames(
                video_path=video_path,
                output_dir=temp_dir,
                max_frames=30,  # Анализируем до 30 кадров для скорости
                frame_interval=10  # Берем каждый 10-й кадр
            )
            
            if not frame_paths:
                return {
                    'error': 'Не удалось извлечь кадры из видео',
                    'is_deepfake': False,
                    'confidence': 0.0
                }
            
            print(f"Извлечено {len(frame_paths)} кадров")
            
            # 2. Обработка каждого кадра
            predictions = []
            sample_frame_encoded = None
            
            for i, frame_path in enumerate(frame_paths):
                # Читаем кадр
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                
                # Детектируем и обрезаем лица
                face_crops = crop_faces(
                    frame, 
                    face_detector, 
                    target_size=(config['image_size'], config['image_size']),
                    margin=0.2
                )
                
                if not face_crops:
                    continue
                
                # Берем первое лицо (можно обработать все лица)
                face = face_crops[0]
                
                # Сохраняем первый кадр для отображения
                if sample_frame_encoded is None and i < 3:
                    # Конвертируем BGR в RGB для PIL
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format='JPEG')
                    sample_frame_encoded = base64.b64encode(buffer.getvalue()).decode()
                
                # Нормализация и подготовка для модели
                # Конвертируем BGR в RGB
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_normalized = face_rgb.astype(np.float32) / 255.0
                
                # Нормализация ImageNet
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                face_normalized = (face_normalized - mean) / std
                
                # Преобразуем в тензор (C, H, W)
                face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).float()
                face_tensor = face_tensor.unsqueeze(0).to(device)
                
                # Предсказание
                with torch.no_grad():
                    output = model(face_tensor)
                    prob = torch.sigmoid(output).item()
                    predictions.append(prob)
            
            if not predictions:
                return {
                    'error': 'В видео не обнаружены лица',
                    'is_deepfake': False,
                    'confidence': 0.0
                }
            
            # 3. Агрегация результатов
            avg_prob = np.mean(predictions)
            is_deepfake = avg_prob > 0.5
            confidence = avg_prob if is_deepfake else (1 - avg_prob)
            
            result = {
                'is_deepfake': bool(is_deepfake),
                'confidence': float(confidence * 100),  # В процентах
                'probabilities': {
                    'real': float((1 - avg_prob) * 100),
                    'fake': float(avg_prob * 100)
                },
                'frames_analyzed': len(predictions),
                'sample_frame': sample_frame_encoded
            }
            
            print(f"Результат: {'ДИПФЕЙК' if is_deepfake else 'РЕАЛЬНОЕ'} (уверенность: {confidence*100:.1f}%)")
            return result
            
    except Exception as e:
        print(f"Ошибка при обработке видео: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': f'Ошибка обработки: {str(e)}',
            'is_deepfake': False,
            'confidence': 0.0
        }


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """Обработка загруженного видео"""
    
    # Check if model is loaded
    if not DEPENDENCIES_OK or model is None:
        return jsonify({
            'error': 'Модель не загружена. Убедитесь, что все зависимости установлены.'
        }), 503
    
    if 'video' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Недопустимый формат файла. Разрешены: mp4, avi, mov, mkv, webm'}), 400
    
    try:
        # Сохраняем файл
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Обрабатываем видео
        result = process_video(filepath)
        
        # Удаляем временный файл
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Ошибка сервера: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not set'
    })


if __name__ == '__main__':
    # Создаем необходимые директории
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Загружаем модель
    print("\n" + "=" * 70)
    print("ИНИЦИАЛИЗАЦИЯ ВЕБ-ПРИЛОЖЕНИЯ ДЛЯ ДЕТЕКЦИИ ДИПФЕЙКОВ")
    print("=" * 70 + "\n")
    
    if not DEPENDENCIES_OK:
        print("\n" + "!" * 70)
        print("ОШИБКА: Не установлены необходимые зависимости!")
        print("!" * 70)
        print("\nУстановите зависимости командой:")
        print("  pip install -r requirements.txt")
        print("\n" + "!" * 70)
        sys.exit(1)
    
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n" + "!" * 70)
        print("ПРЕДУПРЕЖДЕНИЕ: Модель не загружена!")
        print("!" * 70)
        print("\nВеб-сервер запустится, но обработка видео может не работать.")
        print("Проверьте установку зависимостей и конфигурацию.")
        print("\n" + "!" * 70 + "\n")
    
    print("\n" + "=" * 70)
    print("ЗАПУСК СЕРВЕРА")
    print("=" * 70)
    print("\nОткройте браузер и перейдите по адресу: http://127.0.0.1:5000")
    print("Для остановки сервера нажмите Ctrl+C\n")
    
    # Запускаем приложение
    app.run(host='0.0.0.0', port=5000, debug=False)
