# 🎧 Multi-Lingual Audio-Based Sentiment Detection

**Languages Supported:** English, French, German, Italian, Español  
**Modalities:** Audio (MFCC via CNN) + Text (XLM-RoBERTa) + Fusion (MLP on logits)

---

## 🗂️ Directory Structure

```
AUDIO-NLP/
├── backend/
│   ├── inference/
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── audio_cnn.py
│   │       ├── fusion_mlp_model.py
│   ├── pytorch_states/
│   ├── __init__.py
│   ├── audio_model.py
│   ├── fusion_model.py
│   ├── text_model.py
│   ├── whisper_utils.py
│   ├── wheels/
│   ├── app.py
│   ├── download_models.py
│   └── requirements.txt
├── datasets/
│   ├── fusion/
│   ├── processed_audio/
│   ├── processed_text/
│   ├── ravdess/
│   ├── tess_audio/
│   ├── ravdess_dataset_download.ipynb
│   └── textualData.csv
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── training/
│   ├── models/
│   ├── data_preparation.ipynb
│   ├── train_audio_cnn.ipynb
│   ├── train_fusion.ipynb
│   └── train_text_emotion.ipynb
├── utils/
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## ✅ Stepwise Plan

### 1. Dataset Preparation
- **TESS**: Preprocess audio clips to MFCC → `processed_audio/`
- **Kaggle Multilingual Dataset**: Clean multilingual text → `processed_text/`
- Align emotion labels for audio/text fusion  
- Create validation splits

### 2. Audio Emotion Model
- Train CNN on MFCC features  
- Notebook: `train_audio_cnn.ipynb`  
- Save model as `audio_cnn.pt`

### 3. Text Emotion Model
- Fine-tune XLM-RoBERTa  
- Notebook: `train_text_emotion.ipynb`  
- Save model as `text_emotion_model.pt`

### 4. Fusion Model
- Use predictions from audio and text models  
- Train MLP on combined outputs  
- Notebook: `train_fusion.ipynb`  
- Save as `fusion_model.pkl`

### 5. Backend Setup
- Whisper → transcript
- Run XLM-R for text emotion
- Run CNN for audio emotion
- Combine using fusion model
- FastAPI endpoint: `/analyze/`

### 6. Deployment
- Dockerized backend ready (see `Dockerfile`)
- GPU optimized
- Frontend optional (basic Streamlit prototype exists)

---

## ⚠️ Notes
- Emotion classes across datasets are aligned (manually).
- No joint audio-text data → fusion is trained on separate dataset predictions.
- Consider collecting paired audio+text later for supervised multimodal learning.
