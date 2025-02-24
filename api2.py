# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline
import speech_recognition as sr
import soundfile as sf
import os
from groq import Groq

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
tts_pipeline = KPipeline(lang_code='a')
groq_client = Groq(api_key="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")
recognizer = sr.Recognizer()

SYSTEM_PROMPT = """You are a scam call detection assistant. Analyze the user's description of a phone call and:
1. Determine if it's likely spam/scam
2. Explain the red flags
3. Provide protection advice
4. Keep responses under 100 words
5. Use simple, conversational language"""

@app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    # Save audio
    with open("input.wav", "wb") as f:
        f.write(await file.read())
    
    # Transcribe audio using SpeechRecognition
    try:
        with sr.AudioFile("input.wav") as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return {"error": "Could not understand audio"}
    except sr.RequestError as e:
        return {"error": f"Recognition error: {e}"}
    
    # Get LLM analysis
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        model="llama3-70b-8192"
    )
    analysis = response.choices[0].message.content
    
    # Generate response audio
    output_file = "response.wav"
    generator = tts_pipeline(analysis, voice='af_heart', speed=1)
    for i, (_, _, audio) in enumerate(generator):
        sf.write(output_file, audio, 24000)
        break
    
    return {"text": analysis, "audio": output_file}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)