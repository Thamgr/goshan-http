import math
from utils.ml_utils import prepare_features



class GoshanBrain:
    """
    Класс для предсказания вероятности клика по скриншоту.
    Реализован как синглтон - существует только один экземпляр класса.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """
        Создает или возвращает существующий экземпляр класса (синглтон).
        """
        if cls._instance is None:
            cls._instance = super(GoshanBrain, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Инициализация модели. 
        """
        pass


    def prepare_features(self, screenshot, ms_since_last):
        features, img_processed_2d = prepare_features(screenshot)
        features['ms_since_last'] = ms_since_last

        return features, img_processed_2d


    def accelerate_dist(self, ms):
        v, a = 0.2, 1.05
        return v * ms + a * ms ** 2


    def predict_brute_force(self, screenshot, ms_since_last):
        
        features, img_processed_2d = self.prepare_features(screenshot, ms_since_last)

        FLOOR_Y = 0.9

        delay = 80
        ms_since_last += delay

        PADDING_TOP = 0.25
        PADDING_BOTTOM_BASE = 0.15
        PADDING_BOTTOM = PADDING_BOTTOM_BASE
        GOSHAN_Y = features['goshan_y'] + self.accelerate_dist(1 * ms_since_last / 1000 - 0.1)
        GOSHAN_X = features['goshan_x'] + 0.001 * delay
        if GOSHAN_X > features['pipe_0_x']:
            pipe_up = 0
            pipe_down = FLOOR_Y
            pipe_x = 0
        else:
            pipe_up = features['pipe_0_y_up']
            pipe_down = min(FLOOR_Y, features['pipe_0_y_down'])
            pipe_x = features['pipe_0_x']


        if GOSHAN_Y < pipe_up + PADDING_TOP:
            decision = 0
        else:
            decision = math.ceil(max((GOSHAN_Y - pipe_down + PADDING_BOTTOM) / 0.15, 0))
    
        features['decision'] = decision
        features['goshan_delta'] = GOSHAN_Y - features['goshan_y']
        features['padding_bottom'] = PADDING_BOTTOM

        return decision


    def predict(self, screenshot, ms_since_last):
        return self.predict_brute_force(screenshot, ms_since_last)



