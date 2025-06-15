from fastapi import FastAPI, UploadFile, File

from inference.audio_model import predict_audio_emotion
from inference.text_model import predict_text_emotion
from inference.fusion_model import predict_fused_emotion
from inference.whisper_utils import transcribe_audio

import tempfile
import torch

from transformers import XLMRobertaTokenizerFast

from download_models import download_models
download_models()

app=FastAPI()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion2id = {
    "Angry": 0,
    "Sad": 1,
    "Neutral": 2,
    "Surprised": 3,
    "Happy": 4
}

id2emotion = {v: k for k, v in emotion2id.items()}

@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path=tmp.name

    transcript=transcribe_audio(tmp_path)
    audio_emotion=predict_audio_emotion(tmp_path)
    text_emotion=predict_text_emotion(texts=transcript)
    final_emotion=predict_fused_emotion(audio_probs=audio_emotion, text_probs=text_emotion)

    audio_pred=id2emotion[torch.argmax(torch.tensor(audio_emotion)).item()]
    text_pred=id2emotion[torch.argmax(torch.tensor(text_emotion)).item()]
    final_pred=id2emotion[final_emotion]

    return {
        "audio_emotion": audio_emotion,
        "text_emotion": text_emotion,
        "final_emotion": final_emotion,
        "transcription": transcript,
        "audio predicted emotion":audio_pred,
        "text predicted emotion": text_pred,
        "final prediction": final_pred
    }