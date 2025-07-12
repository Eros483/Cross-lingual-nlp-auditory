# Multi-Lingual Multi-Modal Sentiment Analysis
Pipeline to determine sentiment via audio input.

Supports English, French, German, Italian and Español.

## Usage Instructions
Pipeline was encapsulated with a FastAPI backend, and a streamlit frontend, and converted to a Docker Image for ease of use.
Follow the below instructions to use the app, via Docker:
```
docker pull eros483/audio_nlp:latest
docker run -it --rm eros483/audio_nlp:latest
``` 
## Personal Usage Setup Instructions
```
git clone https://github.com/Eros483/Cross-lingual-nlp-auditory.git
cd Cross-lingual-nlp-auditory
conda env create -f environment.yml
conda activate emotion-detection
```

## Working Explaination
### Overview
1. Receives audio input via streamlit interface (user can upload an audio file, or speak into provided microphone).
2. The audio is converted into text via `openai-whisper`.
3. Both Audio and Text are analyzed for sentiment detection.
4. Both predictions are then combined.

### Data Processing (`training/data_preparation.ipynb`)
1. Audio Dataset was pulled from Kaggle's [Ravdess Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).
2. Text Dataset was pulled from [Kaggle](https://www.kaggle.com/datasets/suraj520/multi-task-learning/data) as well.
3. Classified the 5 primary sentiments, Angry, Sad, Neutral, Suprised and Happy into a unified format in both datasets.
### Text Dataset
1. Observed imbalance in dataset, particularly with respect to `sad` values, thus augmented the data with `nlpaug`.
2. Used `xlm-roberta-base` to tokenize textual data, and saved them as pytorch states.
### Audio Dataset
1. Loaded files using `librosa` and extracted `mfccs`.
    - mfccs capture audio data in a manner that mimics human perception of audio frequencies.
2. As variable audio files were present, padded, or truncated the mfccs to a standard shape of (_, 200, 40) and saved as numpy states.
 - Note: Should have standardised usage of torch or numpy states. Causes future conflict.

### Audio Model Training (`training/train_audio_cnn.ipynb`)
1. Applied Train, test split.
2. Wrapped in torch's Data Loader class.
3. Created a basic 2-layer convolutional neural network.
4. Trained over 20 epochs.
5. Obtained a high 93% accuracy.
6. Saved model as pytorch state.
7. Saved ground truth labels, predicted labels, and audio prediction probabilities in preparation for fusion.

### Text Model Training (`training/train_text_emotion.ipynb`)
1. Loaded tokenized data.
2. Mapped labels, and wrapped data in Data Loader class.
3. Trained `XLM-Roberta-Base`, adding a custom head, which applied additional Drop out and layer normalization to improve model performance.
4. Considered `label smoothening loss` and `cross entropy loss` in attempts to improve accuracy, ultimately went with cross entropy loss.
5. Received overall low accuracy of 83%.
    - Key problem was poor performance in `Sad` prediction.
    - Futute note: Handle class imbalances in a improved manner/use different dataset.
6. Saved softmax probabilities and predicted emotion class indices as pytorch dataset.

### Fusion Model Training (`training/train_fusion.ipynb`)
1. Loaded all previous saved probabilities.
    - Forced to slice text probabilities to a shape of [173,5] to match audio dataset.
    - Likely impacted fusion prediction accuracy.
2. Concatenated audio probabilities and text probabilties with 5+5 distribution.
3. Defined a small fully connected MLP with:
    - input_dim = 10
    - hidden layer = 32
    - output = 5 classes
    - Applied dropout after ReLU to regularize.
4. Trained over 20 epochs.
    - Low accuracy of 63%.
    - Reasons likely due to factors considered above while training text dataset, and creating the fusion dataset.
    - Additionally, very small dataset created by fusion, likely further impacted prediction capabilities.

### FastAPI backend and Streamlit frontend
Created API layer for request calls in `backend/app.py`, and connected to streamlit frontend in `frontend/app.py`.




