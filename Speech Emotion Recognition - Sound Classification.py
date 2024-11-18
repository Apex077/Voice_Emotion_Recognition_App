import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')
import sounddevice as sd
from scipy.io.wavfile import write
import noisereduce as nr
import matplotlib.pylab as plt
import time
import random

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

paths = []
labels = []
for dirname, _, filenames in os.walk('./TESS Toronto emotional speech set data'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break
print('Dataset is Loaded')

## Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()

df['label'].value_counts()

sns.countplot(data=df, x='label')

def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()
    
def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

emotion = 'fear'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'angry'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'disgust'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'neutral'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'sad'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'ps'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'happy'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

extract_mfcc(df['speech'][0])

X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))

X = [x for x in X_mfcc]
X = np.array(X)
X.shape

## input split
X = np.expand_dims(X, -1)
X.shape

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])
y = y.toarray()
y.shape

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)

epochs = list(range(50))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

duration = 2
fs = 44100
print("Welcome to the Emotional Response Refinement and Optimization through Recurrent Speech Analysis(ERRORS):\n")
for i in range(3, 0, -1):
    print(f"{i}! ")
    time.sleep(1)
print("Recording has begun!")
recording = sd.rec(int(duration * fs), samplerate = fs, channels = 2)
sd.wait()
print("Recording has completed!")
write('output.wav', fs, recording)

data, sampling_rate = librosa.load('output.wav')
reduced_noise = nr.reduce_noise(y = data, sr = fs, prop_decrease= 0.2)

data = librosa.util.normalize(reduced_noise)
mfccs = librosa.feature.mfcc(y = data, sr = sampling_rate, n_mfcc = 40)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

Audio('output.wav')

mfccs_processed = np.mean(mfccs.T, axis = 0)
mfccs_reshaped = mfccs_processed.reshape(1, -1)

probabilities = model.predict(mfccs_reshaped)
predicted_index = np.argmax(probabilities)

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
predicted_emotion = emotions[predicted_index]

print(f"Predicted emotion: {predicted_emotion}")

random_file_path = random.choice(paths)
print(f"Random File Name: {random_file_path}")
# Load the audio file
audio, sample_rate = librosa.load(random_file_path) 

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_processed = np.mean(mfccs.T, axis=0)

# Reshape it for model input
mfccs_reshaped = mfccs_processed.reshape(1, -1)

# Predict the emotion
probabilities = model.predict(mfccs_reshaped)
predicted_index = np.argmax(probabilities)

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
predicted_emotion = emotions[predicted_index]

print(f"Predicted emotion: {predicted_emotion}")