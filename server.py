from flask import Flask, jsonify
from goshan_brain import GoshanBrain

app = Flask(__name__)
goshan = GoshanBrain()


@app.route('/should-click', methods=['POST'])
def should_click():
    status = goshan.process()
    return jsonify({'status': status})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5300)

