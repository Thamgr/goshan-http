import random


class GoshanBrain:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def predict(self, image, ms_since_click):
        # TODO: Implement the prediction logic
        # The logic should be based on the image and the time since the last click
        # The image is a numpy array of shape (height, width, 3)
        # The ms_since_click is an integer representing the time since the last click in milliseconds
        # The timestamp is a datetime object representing the timestamp of the click
        # The return value should be a integer value: clicks amount
        return random.randint(0, 5)

