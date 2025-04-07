from fastapi import FastAPI, File, UploadFile
import io
import numpy as np
import tensorflow as tf
import librosa
from skimage.transform import resize

app = FastAPI()


model = tf.keras.models.load_model("train_model.keras")


classes = ["Classical", "Jazz", "Rock", "Pop", "Hip-Hop", "Metal", "Blues", "Reggae", "Country", "Electronic"]

def preprocess_file(file_path, target_shape=(128, 128)):
    """Convert an MP3 file into spectrograms for model input."""
    data = []
    audio_data, sr = librosa.load(file_path, sr=None)
    
    duration = 4  
    overlap_duration = 2  
    samples = duration * sr
    overlap_sample = overlap_duration * sr
    total_samples = max(1, int(np.ceil((len(audio_data) - samples) / (samples - overlap_sample))) + 1)

    for i in range(total_samples):
        start = i * (samples - overlap_sample)
        end = start + samples
        chunk = audio_data[start:end]

        if len(chunk) < 512:  
            continue

        n_fft = min(2048, len(chunk) // 2)
        spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr, n_fft=n_fft)
        spectrogram = resize(np.expand_dims(spectrogram, axis=-1), target_shape)  

        data.append(spectrogram)

    return np.array(data) 
def model_pred(x_test):
    """Make predictions using the loaded model."""
    if model is None:
        return "Error: Model not loaded"

    y_pred = model.predict(x_test)  
    pred_cat = np.argmax(y_pred, axis=1)  

    
    unique_el, cnt = np.unique(pred_cat, return_counts=True)
    max_el = unique_el[np.argmax(cnt)]  # Most common prediction
    return classes[max_el]  # Convert index to label

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """API endpoint to receive an MP3 file and predict its genre."""
    contents = await file.read()
    audio_file = io.BytesIO(contents)

    # ✅ Preprocess the file
    spectrograms = preprocess_file(audio_file)

    if spectrograms.size == 0:
        return {"error": "No valid audio segments found"}

    # ✅ Predict the genre
    genre = model_pred(spectrograms)

    return {
        "filename": file.filename,
        "num_segments": len(spectrograms),
        "predicted_genre": genre
    }
