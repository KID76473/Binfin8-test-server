import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

texts = ["I plan to go home now.", "PUT YOUR 2nd TEXT HERE"]

# Sample a speaker from Gaussian.

rand_spk = chat.sample_random_speaker()
print(rand_spk) # save it for later timbre recovery

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .1,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)

wavs = chat.infer(texts)

torchaudio.save("test.wav", torch.from_numpy(wavs[0]), 24000)
