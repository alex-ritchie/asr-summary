import gradio as gr
from transformers import pipeline
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", 
                       device=device, chunk_length_s=30, stride_length_s=(5,5), 
                       return_timestamps=True
)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

def transcribe(stream, new_chunk):
    sr, y = new_chunk
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
        
    # sliding window for audio input
    

    transcription = transcriber({"sampling_rate": sr, "raw": stream})["text"]
    
    # Save transcription to file
    with open("transcription.txt", "w") as f:
        f.write(transcription + "\n")
    
    return stream, transcription, "transcription.txt"

def summarize_transcription():
    with open("transcription.txt", "r") as f:
        transcription = f.read()
    
    summary = summarizer(transcription, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary

with gr.Blocks() as app:
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], streaming=True)
        state = gr.State()
        transcription_output = gr.Textbox(label="Transcription")
        file_output = gr.File(label="Transcription File")
        
    audio_input.stream(transcribe, inputs=[state, audio_input], 
                       outputs=[state, transcription_output, file_output], 
                       stream_every=0.5)
    transcribe_button = gr.Button("Transcribe")
    transcribe_button.click(transcribe, inputs=[state, audio_input], outputs=[state, transcription_output, file_output])
    
    summary_button = gr.Button("Summarize Transcription")
    summary_output = gr.Textbox(label="Summary")
    summary_button.click(summarize_transcription, inputs=[], outputs=summary_output)

app.launch()
