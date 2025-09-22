import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# 1. SET YOUR DATA PATH
DATA_PATH = r'F:\Codealpha\Task 2\dataset_task_2'
SAMPLE_RATE = 22050   # Common sample rate for librosa.load
N_MFCC = 40           # Number of MFCCs to extract
MAX_PAD_LEN = 174     # Fixed length for padding/truncating MFCC frames

def extract_features(file_path, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    """
    Extract MFCC features from an audio file, and pad (or truncate) 
    to a fixed length to ensure consistent input dimensions.
    """
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=sample_rate)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # Pad or truncate MFCC
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        
        return mfcc
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None

def load_data(data_path=DATA_PATH):
    """
    Traverse the directory structure at data_path, load each .wav file, 
    extract features, and label them according to the directory name.
    """
    features = []
    labels = []
    
    for emotion_label in os.listdir(data_path):
        emotion_folder = os.path.join(data_path, emotion_label)
        
        # Check if it's a directory (skip if it's not)
        if not os.path.isdir(emotion_folder):
            continue
        
        # Process each .wav file in the emotion folder
        for file_name in os.listdir(emotion_folder):
            if file_name.endswith('.wav'):
                file_path = os.path.join(emotion_folder, file_name)
                
                mfcc_features = extract_features(file_path)
                if mfcc_features is not None:
                    features.append(mfcc_features)
                    labels.append(emotion_label)

    return np.array(features), np.array(labels)

def build_model(input_shape, num_classes):
    """
    Build a simple CNN for emotion classification from MFCC inputs.
    """
    model = Sequential()
    
    # Convolutional block 1
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # Convolutional block 2
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # 2. LOAD YOUR DATA
    print("Loading data from:", DATA_PATH)
    X, y = load_data(DATA_PATH)

    # 3. PREPARE DATA FOR CNN
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    # Reshape MFCC features to (samples, N_MFCC, MAX_PAD_LEN, 1) for CNN
    X = X[..., np.newaxis]

    # 4. SPLIT INTO TRAIN/TEST
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    # 5. BUILD AND TRAIN THE MODEL
    print("Building model...")
    model = build_model(input_shape=(N_MFCC, MAX_PAD_LEN, 1), num_classes=y_onehot.shape[1])
    model.summary()

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2
    )

    # 6. EVALUATE THE MODEL
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
