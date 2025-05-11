"""main module for launching the Gradio demo."""
import gradio as gr
from loguru import logger

from voice_chat import create_voice_chat_interface

# Static demo definition for Gradio CLI hot-reload detection (RHS must call gradio API)
demo = gr.Blocks()
# Override with actual interface
demo = create_voice_chat_interface()

def main() -> None:
    """Run the main application."""
    logger.info("Starting voice chat application")
    demo.launch()

if __name__ == "__main__":
    main()
