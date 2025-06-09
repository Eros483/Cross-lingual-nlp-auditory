import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
import torch.nn.functional as F
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pytorch_states", "xlm_roberta_emotion_augmented.pt")
torch.load(MODEL_PATH)


class EmotionClassifier(nn.Module):
    def __init__(self, dropout=0.3, num_classes=5):
        super(EmotionClassifier, self).__init__()
        self.base_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.base_model.config.hidden_size)
        self.out = nn.Linear(self.base_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        x = self.dropout(self.norm(pooled_output))
        return self.out(x)

def load_model(model_path, device):
    model = EmotionClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def prepare_inputs(texts, tokenizer, device, max_length=512):
    encodings=tokenizer(
        texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    )
    input_ids=encodings['input_ids'].to(device)
    attention_mask=encodings['attention_mask'].to(device)
    return input_ids, attention_mask

def predict_text_emotion(texts, tokenizer, device):
    model=load_model(model_path=MODEL_PATH, device=device)
    model.eval()
    input_ids, attention_mask = prepare_inputs(texts, tokenizer, device)
    with torch.no_grad():
        outputs=model(input_ids, attention_mask)
        probs=F.softmax(outputs, dim=1)
    return probs.cpu().tolist()