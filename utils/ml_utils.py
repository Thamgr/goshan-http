"""
Утилиты для v3 модели - обработка изображений
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from PIL import Image
from utils.features import ExtractGoshan, PipesExtractor2


def rgb_to_hsv(rgb):
    """
    Конвертирует RGB изображение в HSV.
    
    Args:
        rgb: numpy array в формате RGB (значения 0-255)
        
    Returns:
        numpy array в формате HSV (H: 0-179, S: 0-255, V: 0-255)
    """
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    
    deltac = maxc - minc
    s = deltac / np.maximum(maxc, 1e-10)
    s[maxc == 0] = 0
    
    # Hue
    h = np.zeros_like(v)
    
    mask_r = (maxc == r) & (deltac != 0)
    mask_g = (maxc == g) & (deltac != 0)
    mask_b = (maxc == b) & (deltac != 0)
    
    h[mask_r] = ((g[mask_r] - b[mask_r]) / deltac[mask_r]) % 6
    h[mask_g] = ((b[mask_g] - r[mask_g]) / deltac[mask_g]) + 2
    h[mask_b] = ((r[mask_b] - g[mask_b]) / deltac[mask_b]) + 4
    
    h = h * 30  # Convert to degrees (0-180 for OpenCV compatibility)
    h = h.astype(np.uint8)
    s = (s * 255).astype(np.uint8)
    v = (v * 255).astype(np.uint8)
    
    return np.stack([h, s, v], axis=2)


def crop_white_borders(image, threshold=240):
    """
    Обрезает белые пиксели по краям изображения.
    
    Args:
        image: numpy array изображения (RGB или grayscale)
        threshold: порог яркости для определения "белого" (0-255)
        
    Returns:
        numpy array: обрезанное изображение
    """
    # Конвертируем в grayscale если нужно
    if len(image.shape) == 3:
        # Используем стандартные веса для RGB -> Grayscale (ITU-R BT.601)
        gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    else:
        gray = image.copy()
    
    # Находим все пиксели темнее порога
    mask = gray < threshold
    
    # Находим координаты не-белых пикселей
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        # Если все пиксели белые, возвращаем оригинальное изображение
        return image
    
    # Получаем границы
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Обрезаем изображение
    cropped = image[y_min:y_max+1, x_min:x_max+1]
    
    return cropped


def recolor_image(image):
    hsv = rgb_to_hsv(image)
    
    # Маска для оранжевых пикселей (Hue в диапазоне оранжевого, достаточная насыщенность и яркость)
    # Диапазон обновлен на основе анализа реальных труб из with_pipes.png:
    # Реальные HSV значения оранжевых труб: H=4-11, S=113-206, V=100-210
    # Используем более широкий диапазон для надежного обнаружения
    orange_mask = (
        (hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 20) &    # Hue для оранжевого (расширенный)
        (hsv[:, :, 1] >= 100) & (hsv[:, :, 1] <= 255) & # Достаточная насыщенность (смягчено)
        (hsv[:, :, 2] >= 90) & (hsv[:, :, 2] <= 255)   # Достаточная яркость (смягчено)
    )
    
    # Маска: оранжевые
    keep_mask = orange_mask
    
    # Все остальные пиксели делаем белыми
    image[keep_mask] = [0, 0, 0]
    image[~keep_mask] = [255, 255, 255]

    return image


def crop_target_field_part(image):

    # down_part_height = height - (height // 5)  # Убираем последнюю 1/5 часть
    # image = image[:down_part_height, :]


    return image


def resize_image(image):
    # Используем фиксированный размер для всех изображений
    # чтобы все имели одинаковую форму после flatten()
    TARGET_WIDTH = 64
    # Конвертируем numpy array в PIL Image
    pil_image = Image.fromarray(image.astype(np.uint8))
    # Изменяем размер
    pil_image = pil_image.resize((TARGET_WIDTH, TARGET_WIDTH), Image.LANCZOS)
    # Конвертируем обратно в numpy array
    image = np.array(pil_image)

    return image


def _preprocess_image_2d(image):
    """
    Предобрабатывает изображение без flatten (внутренняя функция).
    
    Returns:
        numpy array: 2D изображение (grayscale, нормализованное)
    """
    img_processed = image.copy()
    
    img_processed = crop_target_field_part(img_processed)
    # Конвертируем в grayscale
    img_processed = resize_image(img_processed)
    img_processed = recolor_image(img_processed)
    img_processed = np.dot(img_processed[...,:3], [0.299, 0.587, 0.114])
    # ImageSaver.save_image_universal(img_processed)
    img_processed = img_processed / 255.0
    
    return img_processed


def prepare_features(image):
    """
    Извлекает все признаки из изображения игры в формате датасета.
    
    Обработка:
    1. Находит Гошана с помощью ExtractGoshan
    2. Делает preprocess_image для получения обработанного изображения
    3. Извлекает координаты труб с помощью PipesExtractor
    4. Добавляет фиктивные трубы в конец массива для гарантии минимум 2 элементов
    5. Формирует словарь с готовыми признаками для модели
    
    Args:
        image: numpy array изображения (RGB)
        
    Returns:
        dict: словарь с признаками в формате датасета:
            - goshan_x: float - координата x Гошана (относительная)
            - goshan_y: float - координата y Гошана (относительная)
            - goshan_confidence: float - уверенность детекции Гошана
            - pipe_0_x: float - координата x первой трубы
            - pipe_0_y_up: float - координата верхней части первой трубы
            - pipe_0_y_down: float - координата нижней части первой трубы
            - pipe_1_x: float - координата x второй трубы
            - pipe_1_y_up: float - координата верхней части второй трубы
            - pipe_1_y_down: float - координата нижней части второй трубы
            - image: numpy array обработанного изображения (2D, grayscale, нормализованный)
    """
    # 1. Находим Гошана
    goshan_extractor = ExtractGoshan()
    goshan = goshan_extractor.process(image)
    
    # 2. Предобрабатываем изображение (переиспользуем preprocess_image)
    img_processed_2d = _preprocess_image_2d(image)
    
    # 3. Извлекаем трубы (используем ненормализованное изображение для PipesExtractor)
    pipes_extractor = PipesExtractor2()
    pipes = pipes_extractor.process(img_processed_2d)  # Денормализуем для PipesExtractor
    
    # 4. Добавляем фиктивные трубы, чтобы гарантировать минимум 2 элемента в массиве
    # Фиктивные трубы находятся "за экраном" (x=1.0 - относительная координата)
    dummy_pipe = {'x': 1.0, 'y_up': 0.0, 'y_down': 1.0}
    while len(pipes) == 0:
        pipes.append(dummy_pipe.copy())
    
    # 5. Извлекаем данные Гошана с дефолтными значениями
    if goshan is not None:
        goshan_x = goshan.get('x', 0)
        goshan_y = goshan.get('y', 0)
        goshan_confidence = goshan.get('confidence', 0)
    else:
        goshan_x = 0
        goshan_y = -0.5
        goshan_confidence = 0
    
    # 6. Извлекаем данные первых двух труб
    pipe_0 = pipes[-1]
    pipe_0_x = pipe_0['x']
    pipe_0_y_up = pipe_0['y_up']
    pipe_0_y_down = pipe_0['y_down']
    
    features = {
        'goshan_x': goshan_x,
        'goshan_y': goshan_y,
        'pipe_0_x': pipe_0_x,
        'pipe_0_y_up': pipe_0_y_up,
        'pipe_0_y_down': pipe_0_y_down,
    }

    return features, img_processed_2d


def prepare_input(image, ms_since_last_keypress=None):
    """
    Подготавливает входные данные для модели из изображения.
    
    Возвращает массив признаков в том же формате, что используется
    в датасете v2 (см. create_dataset_v2.py).
    
    Args:
        image: numpy array изображения (RGB) или dict с признаками из prepare_features
        ms_since_last_keypress: время с последнего нажатия в миллисекундах (опционально)
        
    Returns:
        numpy array: массив признаков для модели [
            ms_since_last_keypress,
            goshan_x, goshan_y, goshan_confidence,
            pipe_0_x, pipe_0_y_up, pipe_0_y_down,
            pipe_1_x, pipe_1_y_up, pipe_1_y_down
        ]
    """
    features, img_processed_2d = prepare_features(image)
    
    # Формируем массив из словаря признаков
    input_array = np.array([
        ms_since_last_keypress if ms_since_last_keypress is not None else 0.5,
        features['goshan_x'],
        features['goshan_y'],
        features['goshan_confidence'],
        features['pipe_0_x'],
        features['pipe_0_y_up'],
        features['pipe_0_y_down'],
        features['pipe_1_x'],
        features['pipe_1_y_up'],
        features['pipe_1_y_down']
    ])
    
    return input_array, img_processed_2d
