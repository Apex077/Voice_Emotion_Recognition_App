# Speech_Emotion_Recognition

This Python script is designed to perform speech emotion recognition using a deep learning model. It uses the TESS Toronto emotional speech set data for training and validation. The script is divided into several sections:

1. **Data Loading and Visualization**: The script starts by loading the dataset, which consists of audio files and their corresponding emotion labels. It then visualizes the distribution of the labels and the waveforms and spectrograms of sample audio files for each emotion.

**Link to the Dataset**: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

2. **Feature Extraction**: The script extracts Mel-frequency cepstral coefficients (MFCCs) from the audio files. MFCCs are a type of spectral feature that are widely used in speech and audio processing.

3. **Model Building and Training**: The script defines a deep learning model using Keras. The model is a recurrent neural network (RNN) with LSTM layers, followed by dense layers. The model is trained using the extracted MFCCs and the emotion labels.

4. **Model Evaluation**: The script plots the training and validation accuracy and loss over the epochs to evaluate the performance of the model.

5. **Emotion Prediction**: The script records a 2-second audio clip, reduces noise from the clip, extracts its MFCCs, and uses the trained model to predict the emotion in the clip. The predicted emotion is then printed out.

This script requires several Python libraries, including pandas, numpy, seaborn, matplotlib, librosa, keras, and sklearn.

**Steps to Run the code:**
1) Download the TESS Dataset from the link provided above and extract it, and place it in the same file directory as the python script.
2) Run the python script in the terminal using the python <scriptname>.py command

NOTE: This project is just a proof of concept. It still needs a lot of refinement and fine-tuning and generally only works on female voices because of the dataset used.
