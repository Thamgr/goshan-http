import random


class GoshanBrain:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def process(self):
        return random.randint(0, 1)

