"""Unit tests for the sound_effects module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sound_effects import SoundEffects


@pytest.fixture
def mock_sound_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with mock sound files."""
    sound_dir = tmp_path / "sounds"
    sound_dir.mkdir()

    (sound_dir / "click.mp3").touch()
    (sound_dir / "notification.mp3").touch()
    (sound_dir / "success.mp3").touch()
    (sound_dir / "waiting.mp3").touch()
    (sound_dir / "bgm.mp3").touch()

    return sound_dir


class TestSoundEffects:
    """Test cases for the SoundEffects class."""

    def test_init_with_existing_dir(self, mock_sound_dir: Path) -> None:
        """Test initialization with an existing sounds directory."""
        sound_effects = SoundEffects(sounds_dir=str(mock_sound_dir))

        assert len(sound_effects.sound_effects) == 5
        assert "click" in sound_effects.sound_effects
        assert "notification" in sound_effects.sound_effects
        assert "success" in sound_effects.sound_effects
        assert "waiting" in sound_effects.sound_effects
        assert "bgm" in sound_effects.sound_effects

    def test_init_with_nonexistent_dir(self) -> None:
        """Test initialization with a non-existent sounds directory."""
        with patch("pathlib.Path.exists", return_value=False):
            sound_effects = SoundEffects(sounds_dir="nonexistent")

            assert len(sound_effects.sound_effects) == 0

    def test_get_sound_effect_existing(self, mock_sound_dir: Path) -> None:
        """Test getting an existing sound effect."""
        sound_effects = SoundEffects(sounds_dir=str(mock_sound_dir))

        result = sound_effects.get_sound_effect("click")

        assert result == str(mock_sound_dir / "click.mp3")

    def test_get_sound_effect_nonexistent(self, mock_sound_dir: Path) -> None:
        """Test getting a non-existent sound effect."""
        sound_effects = SoundEffects(sounds_dir=str(mock_sound_dir))

        result = sound_effects.get_sound_effect("nonexistent")

        assert result == ""

    def test_get_available_sound_effects(self, mock_sound_dir: Path) -> None:
        """Test getting available sound effects."""
        sound_effects = SoundEffects(sounds_dir=str(mock_sound_dir))

        result = sound_effects.get_available_sound_effects()

        assert sorted(result) == ["bgm", "click", "notification", "success", "waiting"]

    def test_play_sound_effect(self, mock_sound_dir: Path) -> None:
        """Test playing a sound effect."""
        sound_effects = SoundEffects(sounds_dir=str(mock_sound_dir))

        with patch.object(sound_effects, "get_sound_effect") as mock_get:
            mock_get.return_value = str(mock_sound_dir / "click.mp3")

            result = sound_effects.play_sound_effect("click")

            mock_get.assert_called_once_with("click")
            assert result == str(mock_sound_dir / "click.mp3")

    def test_play_bgm(self, mock_sound_dir: Path) -> None:
        """Test playing background music."""
        sound_effects = SoundEffects(sounds_dir=str(mock_sound_dir))

        with patch.object(sound_effects, "get_sound_effect") as mock_get:
            mock_get.return_value = str(mock_sound_dir / "bgm.mp3")

            result = sound_effects.play_bgm("bgm")

            mock_get.assert_called_once_with("bgm")
            assert result == str(mock_sound_dir / "bgm.mp3")
            assert sound_effects.current_bgm == "bgm"

    def test_stop_bgm_with_active_bgm(self, mock_sound_dir: Path) -> None:
        """Test stopping background music when it is playing."""
        sound_effects = SoundEffects(sounds_dir=str(mock_sound_dir))
        sound_effects.current_bgm = "bgm"

        result = sound_effects.stop_bgm()

        assert result == ""
        assert sound_effects.current_bgm is None

    def test_stop_bgm_with_no_bgm(self, mock_sound_dir: Path) -> None:
        """Test stopping background music when none is playing."""
        sound_effects = SoundEffects(sounds_dir=str(mock_sound_dir))
        sound_effects.current_bgm = None

        result = sound_effects.stop_bgm()

        assert result == ""
        assert sound_effects.current_bgm is None
