"""Sound effects module for the UDK app."""

from pathlib import Path

import gradio as gr
from loguru import logger

from settings import settings


class SoundEffects:
    """Sound effects manager for the UDK app."""

    def __init__(self, sounds_dir: str | Path | None = None) -> None:
        """Initialize the sound effects manager.

        Args:
            sounds_dir: Directory containing sound effect files, defaults to settings.sounds_dir

        """
        # Accept either a Path or a string for sounds_dir
        if sounds_dir is not None:
            self.sounds_dir = Path(sounds_dir)
        else:
            self.sounds_dir = settings.sounds_dir
        self.sound_effects: dict[str, str] = self._load_sound_effects()
        self.current_bgm: str | None = None

    def _load_sound_effects(self) -> dict[str, str]:
        """Load sound effects from the sounds directory.

        Returns:
            Dictionary mapping sound effect names to file paths

        """
        sound_effects = {}

        if not self.sounds_dir.exists():
            logger.warning(f"Sounds directory {self.sounds_dir} does not exist")
            return sound_effects

        for sound_file in self.sounds_dir.glob("*.mp3"):
            sound_name = sound_file.stem
            sound_effects[sound_name] = str(sound_file)
            logger.info(f"Loaded sound effect: {sound_name} from {sound_file}")

        return sound_effects

    def get_sound_effect(self, name: str) -> str:
        """Get the path to a sound effect by name.

        Args:
            name: Name of the sound effect

        Returns:
            Path to the sound effect file

        """
        if name not in self.sound_effects:
            logger.warning(f"Sound effect {name} not found")
            return ""

        return self.sound_effects[name]

    def get_available_sound_effects(self) -> list[str]:
        """Get a list of available sound effect names.

        Returns:
            List of sound effect names

        """
        return list(self.sound_effects.keys())

    def play_sound_effect(self, name: str) -> str:
        """Play a sound effect by name.

        Args:
            name: Name of the sound effect

        Returns:
            Path to the sound effect file for Gradio to play

        """
        sound_path = self.get_sound_effect(name)
        if not sound_path:
            return ""

        logger.info(f"Playing sound effect: {name}")
        return sound_path

    def play_bgm(self, name: str) -> str:
        """Play background music by name.

        Args:
            name: Name of the BGM sound file

        Returns:
            Path to the BGM sound file for Gradio to play

        """
        sound_path = self.get_sound_effect(name)
        if not sound_path:
            return ""

        self.current_bgm = name
        logger.info(f"Starting BGM: {name}")
        return sound_path

    def stop_bgm(self) -> str:
        """Stop currently playing background music.

        Returns:
            Empty string to clear the audio element

        """
        if self.current_bgm:
            logger.info(f"Stopping BGM: {self.current_bgm}")
            self.current_bgm = None
        return ""
