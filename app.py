from flask import Flask, request, jsonify, send_file
import ChatTTS
import torch
import torchaudio
import logging
from flask_cors import CORS
import os

app = Flask(__name__)
app.logger.setLevel(logging.INFO)  # Set the logging level to INFO
CORS(app)  # Enable CORS for all routes
# CORS(app, resources={r"/generate_audio": {"origins": "ws://127.0.0.1:54589/HeLzIXKCYCo=/ws"}})

@app.route('/test')
def test():
    return 'test'

@app.route('/generate_audio', methods=['GET']) 
def generate_audio_route():
    text = request.args.get('text')
    app.logger.info(f"Receive text: {text}")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    generate_audio(text)
    output_path = "./output.wav"

    if not os.path.exists(output_path):
        app.logger.error(f"Audio file not found at {output_path}")
        return jsonify({"error": "Audio file not found"}), 500

    try:
        return send_file(output_path, as_attachment=True), 200
    except Exception as e:
        app.logger.error(f"Error sending file: {e}")
        return jsonify({"error": "Failed to send audio file"}), 500

def generate_audio(text):
    chat = ChatTTS.Chat()
    chat.load(compile=False)

    wavs = chat.infer(text)

    torchaudio.save("output.wav", torch.from_numpy(wavs[0]), 24000)
