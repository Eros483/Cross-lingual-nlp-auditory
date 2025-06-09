import torch
import torchaudio
import numpy as np
import torch.nn as nn
from .models.audio_cnn import AudioCNN
import os

num_classes = 5

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model=None
_mfcc_shape=(200,40)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pytorch_states", "audio_cnn.pt")
torch.load(MODEL_PATH)

def extract_mfcc(audio_path, n_mfcc=40, max_len=200):
    waveform, sample_rate=torchaudio.load(audio_path)

    mfcc_transform=torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)
    mfcc=mfcc_transform(waveform)
    mfcc=mfcc.squeeze(0).transpose(0,1)

    if mfcc.shape[0]<max_len:
        pad_len=max_len-mfcc.shape[0]
        mfcc=torch.nn.functional.pad(mfcc, (0,0, 0, pad_len))
    else:
        mfcc=mfcc[:max_len, :]

    mfcc=mfcc.unsqueeze(0).unsqueeze(0)
    return mfcc

def load_model(mfcc_shape=(200,80)):
    global _model
    if _model is None:
        _model=AudioCNN(n_classes=num_classes, mfcc_shape=_mfcc_shape).to(device)
        _model.load_state_dict(torch.load(MODEL_PATH))
        _model.eval()


def predict_audio_emotion(audio_path):
    mfcc=extract_mfcc(audio_path).to(device)
    mfcc_shape = mfcc.shape[-2:] 
    mfcc=mfcc.to(device)

    load_model(mfcc_shape)

    with torch.no_grad():
        output=_model(mfcc)
        probs=torch.softmax(output, dim=1)

    return probs.squeeze().cpu().tolist()