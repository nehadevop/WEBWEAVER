from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import io
import soundfile as sf
import librosa
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
from pydub import AudioSegment  # pydub is used for MP3 to WAV conversion

# Initialize FastAPI app and allow all CORS origins.
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model from your saved directory.
model_path = "best_model_1"
model = AutoModelForAudioClassification.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

# Instead of using AutoFeatureExtractor with your model dir (which lacks preprocessor_config.json),
# load the feature extractor from the original pretrained model.
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# Send model to the appropriate device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your label mapping.
id2label = {0: "spam", 1: "legitimate"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file contents.
    contents = await file.read()
    audio_bytes = io.BytesIO(contents)
    
    # If the file is an MP3, convert it to WAV using pydub.
    if file.filename.lower().endswith(".mp3"):
        audio = AudioSegment.from_file(audio_bytes, format="mp3")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        audio_bytes = wav_io
    
    # Load the audio file using soundfile.
    audio_array, sr = sf.read(audio_bytes)
    
    # Resample the audio if its sampling rate does not match the feature extractor's expected rate.
    if sr != feature_extractor.sampling_rate:
        audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=sr, target_sr=feature_extractor.sampling_rate)
    
    # Ensure the audio is in float32 format.
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    
    # Process the audio using the feature extractor.
    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Move inputs to the same device as the model.
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Perform inference.
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Apply softmax to get probabilities.
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
    predicted_label = int(np.argmax(probabilities))
    score = float(np.max(probabilities))
    
    return {"label": id2label[predicted_label], "score": score}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
