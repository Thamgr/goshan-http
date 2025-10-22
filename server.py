from flask import Flask, jsonify, request
from goshan_brain import GoshanBrain
import base64
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)
goshan = GoshanBrain()


@app.route('/should-click', methods=['POST'])
def should_click():
    data = request.get_json()
    
    image_base64 = data.get('image')
    ms_since_click = data.get('ms_since_click')
    timestamp = data.get('timestamp')
    
    # Декодируем base64 в numpy array
    image_bytes = base64.b64decode(image_base64)
    image_buffer = BytesIO(image_bytes)
    image_pil = Image.open(image_buffer)
    image_array = np.array(image_pil)
    
    status = goshan.predict(image_array, ms_since_click)
    return jsonify({'status': status})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5300)

