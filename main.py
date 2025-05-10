"""main."""

from loguru import logger

from voice_chat import create_voice_chat_interface


def main() -> None:
    """Run the main application."""
    logger.info("Starting voice chat application")
    interface = create_voice_chat_interface()
    interface.launch()


if __name__ == "__main__":
    main()
