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
    temperature = request.args.get('temperature')
    top_p = request.args.get('top_p')
    top_k = request.args.get('top_k')
    app.logger.info(f"Receive text: {text}, temperature: {temperature}, top_p: {top_p}, top_k: {top_k}")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    generate_audio(text, temperature, top_p, top_k)
    output_path = "./output.wav"

    if not os.path.exists(output_path):
        app.logger.error(f"Audio file not found at {output_path}")
        return jsonify({"error": "Audio file not found"}), 500

    try:
        return send_file(output_path, as_attachment=True), 200
    except Exception as e:
        app.logger.error(f"Error sending file: {e}")
        return jsonify({"error": "Failed to send audio file"}), 500

def generate_audio(text, temp, p, k):
    chat = ChatTTS.Chat()
    chat.load(compile=False)

    rand_spk = chat.sample_random_speaker()
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb = rand_spk, # add sampled speaker 
        temperature = temp,   # using custom temperature
        top_P = p,        # top P decode
        top_K = k,         # top K decode
    )

    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_2][laugh_0][break_4]',
    )

    wavs = chat.infer(text)

    torchaudio.save("output.wav", torch.from_numpy(wavs[0]), 24000)
