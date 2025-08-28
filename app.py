import gradio as gr
import whisper
from gtts import gTTS
import tempfile
from groq import Groq
import os

# 1. Load Whisper model once (STT)
stt_model = whisper.load_model("small")

# 2. Initialize Groq client (LLM) - use env variable
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def voice_agent(audio_file):
    # --- Step 1: Speech-to-Text ---
    user_text = stt_model.transcribe(audio_file)["text"]

    # --- Step 2: LLM Response ---
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": user_text}]
    )
    ai_text = completion.choices[0].message["content"]

    # --- Step 3: Text-to-Speech ---
    tts = gTTS(ai_text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        audio_output = f.name

    return user_text, ai_text, audio_output

# Gradio UI
demo = gr.Interface(
    fn=voice_agent,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs=[
        gr.Textbox(label="You said"),
        gr.Textbox(label="AI Response"),
        gr.Audio(label="AI Voice")
    ],
    title="🎙️ AI Voice Agent",
    description="Talk to an AI powered by Whisper (STT) + Groq LLaMA3 (LLM) + gTTS (TTS)"
)

if __name__ == "__main__":
    # Render assigns a PORT env var, must use 0.0.0.0
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
