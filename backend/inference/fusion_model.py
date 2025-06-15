import torch
import pickle
from .models.fusion_mlp_model import FusionMLP
import torch.nn.functional as F
import os

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pytorch_states", "fusion_mlp_final.pt")

_model=None

def load_model():
    global _model
    if _model is None:
        _model=FusionMLP().to(device)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        _model.load_state_dict(state_dict)
        _model.eval()
    return _model


def predict_fused_emotion(audio_probs, text_probs):

    model=load_model()

    if isinstance(audio_probs, list):
        audio_probs=torch.tensor(audio_probs)
    if isinstance(text_probs, list):
        text_probs=torch.tensor(text_probs)

    if audio_probs.dim() == 1:
        audio_probs = audio_probs.unsqueeze(0) 
    if text_probs.dim() == 1:
        text_probs = text_probs.unsqueeze(0)  

    fusion_input=torch.cat([audio_probs, text_probs], dim=1).to(device)
    with torch.no_grad():
        outputs=model(fusion_input)
        probs=F.softmax(outputs, dim=1)
        pred=torch.argmax(probs, dim=1).item()
    return pred