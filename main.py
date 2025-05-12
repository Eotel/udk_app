"""main module for launching the Gradio demo."""

import gradio as gr
from loguru import logger

from settings import settings
from voice_chat import create_voice_chat_interface

demo = create_voice_chat_interface()


def main() -> None:
    """Run the main application."""
    logger.info("Starting voice chat application")
    if settings.debug:
        logger.info("Running in debug mode")
    demo.launch(
        server_name=settings.host,
        server_port=7861,  # Use a different port
        debug=settings.debug,
        share=False,  # Set to True to create a public URL
        max_threads=40,  # More threads for better streaming
    )


if __name__ == "__main__":
    main()
