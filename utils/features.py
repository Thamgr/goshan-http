import numpy as np
from typing import Optional, Dict, List


class ExtractGoshan:
    """Находит позицию Гошана по скоплению черных пикселей (очки)."""
    
    def process(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        h, w = image.shape[:2]
        search_w = w  # Левая 1/6 изображения
        
        # Преобразуем в grayscale если нужно
        if len(image.shape) == 3:
            gray = np.mean(image[:, :search_w], axis=2)
        else:
            gray = image[:, :search_w]
        
        # Ищем все квадраты 3x3 черных пикселей (яркость < 50)
        black_squares_x = []
        black_squares_y = []
        
        for y in range(h - 3):
            for x in range(search_w - 3):
                if len(black_squares_x) > 5:
                    break
                if np.all(gray[y:y+3, x:x+3] < 50):
                    black_squares_x.append(x + 1.5)
                    black_squares_y.append(y + 1.5)
        
        # Если нашли хотя бы один квадрат, возвращаем медианные координаты
        if black_squares_x:
            return {
                'y': np.median(black_squares_y) / h,
                'x': np.median(black_squares_x) / w,
                'confidence': 1.0
            }
        
        return None


class PipesExtractor2:
    """Экстрактор труб через анализ паттернов в столбцах."""
    
    BLACK_COLOR = 0
    WHITE_COLOR = 1
    MIN_BLACK_STREAK = 4
    MIN_WHITE_STREAK = 4
    MERGE_DISTANCE = 5
    
    def process(self, image: np.ndarray) -> List[Dict[str, float]]:
        """
        Определяет координаты труб на изображении.
        
        Args:
            image: numpy array (grayscale)
        
        Returns:
            Список словарей {'x': ..., 'y_up': ..., 'y_down': ...}
        """
        height, width = image.shape
        pipe_candidates = []
        
        for x in range(width):
            if pipe_data := self._check_column(image[:, x], x):
                pipe_candidates.append(pipe_data)
        
        merged_pipes = self._merge_pipes(pipe_candidates)
        return self._normalize_pipes(merged_pipes, width, height)
    
    def _check_column(self, column: np.ndarray, x: int) -> Optional[Dict[str, int]]:
        """Проверяет столбец на наличие паттерна труб."""
        if not self._has_pipe_marker(column):
            return None
        
        boundaries = self._find_pipe_boundaries(column)
        if not boundaries:
            return None
        
        return {'x': x, 'y_up': boundaries['y_up'], 'y_down': boundaries['y_down']}
    
    def _has_pipe_marker(self, column: np.ndarray) -> bool:
        """Проверяет наличие 4+ черных пикселей подряд."""
        streak = 0
        for pixel in column:
            if pixel == self.BLACK_COLOR:
                streak += 1
                if streak >= self.MIN_BLACK_STREAK:
                    return True
            else:
                streak = 0
        return False
    
    def _find_pipe_boundaries(self, column: np.ndarray) -> Optional[Dict[str, int]]:
        """Находит границы труб - границы белой серии длиной >=4."""
        streaks = self._extract_streaks(column)
        filtered = [s for s in streaks if s['length'] >= self.MIN_BLACK_STREAK]
        
        for streak in filtered:
            if streak['color'] == self.WHITE_COLOR:
                return {
                    'y_up': streak['start'],
                    'y_down': streak['end']
                }
        
        return None
    
    def _extract_streaks(self, column: np.ndarray) -> List[Dict]:
        """Извлекает последовательности черных и белых пикселей."""
        streaks = []
        current_color = column[0]
        start = 0
        
        for i in range(1, len(column)):
            if column[i] != current_color:
                streaks.append({
                    'color': current_color,
                    'start': start,
                    'end': i,
                    'length': i - start
                })
                current_color = column[i]
                start = i
        
        streaks.append({
            'color': current_color,
            'start': start,
            'end': len(column),
            'length': len(column) - start
        })
        
        return streaks
    
    def _merge_pipes(self, candidates: List[Dict[str, int]]) -> List[Dict[str, int]]:
        """Объединяет соседние столбцы в одну трубу."""
        if not candidates:
            return []
        
        merged = []
        current = candidates[0].copy()
        
        for next_pipe in candidates[1:]:
            if next_pipe['x'] - current['x'] < self.MERGE_DISTANCE:
                current['x'] = next_pipe['x']
                current['y_up'] = (current['y_up'] + next_pipe['y_up']) // 2
                current['y_down'] = (current['y_down'] + next_pipe['y_down']) // 2
            else:
                merged.append(current)
                current = next_pipe.copy()
        
        merged.append(current)
        return merged
    
    def _normalize_pipes(self, pipes: List[Dict[str, int]], width: int, height: int) -> List[Dict[str, float]]:
        """Нормализует координаты труб."""
        return [
            {
                'x': pipe['x'] / width,
                'y_up': pipe['y_up'] / height,
                'y_down': pipe['y_down'] / height
            }
            for pipe in pipes
        ]
