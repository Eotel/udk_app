"""main module for launching the Gradio demo."""
from loguru import logger

from voice_chat import create_voice_chat_interface

# Expose Gradio interface for CLI hot-reload detection
demo = create_voice_chat_interface()

def main() -> None:
    """Run the main application."""
    logger.info("Starting voice chat application")
    demo.launch()

if __name__ == "__main__":
    main()
