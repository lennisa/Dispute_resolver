import uvicorn
from fastapi import FastAPI
import gradio as gr
import threading

# Import your FastAPI app from main.py
from main import app  # Assumes your FastAPI instance in main.py is named `app`

# Function to run Uvicorn in the background
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Start FastAPI in a background thread
    server_thread = threading.Thread(target=run_fastapi, daemon=True)
    server_thread.start()

    # Launch a minimal Gradio interface to satisfy the Hugging Face Space requirement
    with gr.Blocks() as demo:
        gr.Markdown("#  Dispute Resolver ML Backend is Live!")
        gr.Markdown("Your FastAPI backend is running. Access your API docs interactively below:")
        gr.HTML('<p>Go to <a href="/docs" target="_blank"><b>/docs</b></a> to view the Swagger UI.</p>')

    demo.launch(server_name="0.0.0.0", server_port=7860)