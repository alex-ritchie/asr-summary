import gradio as gr
import numpy as np
from transformers import pipeline

# Initialize ASR pipeline
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def process_audio(audio, state=""):
    # Convert audio to numpy array if it isn't already
    if isinstance(audio, dict):
        audio = audio["audio"]
    
    # Process the audio chunk
    try:
        result = asr(audio)
        transcription = result["text"]
        
        # Append new transcription to existing state
        new_state = state + " " + transcription if state else transcription
        return new_state, new_state
    except Exception as e:
        return state, f"Error processing audio: {str(e)}"

def clear_output():
    return "", ""

with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Streaming Speech Recognition")
    
    # State for maintaining continuous transcription
    state = gr.State("")
    
    with gr.Row():
        # Audio input component configured for streaming
        audio = gr.Audio(
            sources=["microphone"],
            type="numpy",
            streaming=True,
            label="Speak into your microphone"
        )
    
    with gr.Row():
        # Output text area for transcription
        output = gr.Textbox(
            label="Transcription",
            placeholder="Your speech will appear here...",
            lines=5
        )
    
    with gr.Row():
        # Clear button
        clear_btn = gr.Button("Clear Transcription")
    
    # Handle streaming audio input
    audio.stream(
        fn=process_audio, 
        inputs=[audio, state],
        outputs=[state, output],
        show_progress=False
    )
    
    # Clear button functionality
    clear_btn.click(
        fn=clear_output,
        inputs=[],
        outputs=[state, output]
    )

if __name__ == "__main__":
    demo.queue().launch()