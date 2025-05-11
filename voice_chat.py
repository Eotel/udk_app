"""Voice chat implementation using OpenAI API and Gradio."""

import os
import tempfile
import base64
from pathlib import Path

import gradio as gr
import openai
from loguru import logger

from models import (
    ChatMessage,
    ChatRequest,
    SpeechRequest,
    TranscriptionRequest,
    VoiceChatState,
)
from sound_effects import SoundEffects


class VoiceChat:
    """Voice chat implementation using OpenAI API."""

    def __init__(self) -> None:
        """Initialize the voice chat with OpenAI API key."""
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.state = VoiceChatState()
        self.sound_effects = SoundEffects()

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using OpenAI API."""
        logger.info(f"Transcribing audio from {audio_path}")
        request = TranscriptionRequest(file_path=audio_path)

        with Path(request.file_path).open("rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=request.model,
                file=audio_file,
            )

        transcribed_text = response.text
        logger.info(f"Transcribed text: {transcribed_text}")
        return transcribed_text

    def generate_response(self, user_message: str) -> str:
        """Generate a response to the user message using OpenAI API."""
        logger.info(f"Generating response to: {user_message}")

        self.state.conversation_history.append(
            ChatMessage(role="user", content=user_message),
        )

        request = ChatRequest(messages=self.state.conversation_history)

        response = self.client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        response_text = response.choices[0].message.content

        self.state.conversation_history.append(
            ChatMessage(role="assistant", content=response_text),
        )

        logger.info(f"Generated response: {response_text}")
        return response_text

    def synthesize_speech(self, text: str) -> str:
        """Synthesize speech from text using OpenAI API."""
        logger.info(f"Synthesizing speech for: {text}")
        request = SpeechRequest(text=text)

        response = self.client.audio.speech.create(
            model=request.model,
            voice=request.voice,
            input=request.text,
        )

        temp_dir = Path(tempfile.gettempdir())
        audio_path = temp_dir / "response.mp3"
        response.stream_to_file(str(audio_path))

        logger.info(f"Speech synthesized to {audio_path}")
        return str(audio_path)

    def process_voice_input(self, audio_input: str | tuple) -> tuple[str, str, list]:
        """Process voice input and return the response."""
        audio_path = audio_input[0] if isinstance(audio_input, tuple) else audio_input

        user_text = self.transcribe_audio(audio_path)

        return self.process_input(user_text)

    def process_text_input(self, text_input: str) -> tuple[str, str, list]:
        """Process text input and return the response."""
        logger.info(f"Processing text input: {text_input}")

        return self.process_input(text_input)

    def process_input(self, user_text: str) -> tuple[str, str, list]:
        """Process user input (text or transcribed voice) and return the response."""
        response_text = self.generate_response(user_text)

        audio_output = self.synthesize_speech(response_text)

        chat_history = [
            (
                self.state.conversation_history[i].content,
                self.state.conversation_history[i + 1].content,
            )
            for i in range(0, len(self.state.conversation_history), 2)
            if i + 1 < len(self.state.conversation_history)
        ]

        return user_text, audio_output, chat_history


def create_voice_chat_interface() -> gr.Blocks:
    """Create a Gradio interface for voice chat."""
    voice_chat = VoiceChat()
    sound_effects = voice_chat.sound_effects
    available_effects = sound_effects.get_available_sound_effects()

    def play_sound_effect(effect_name: str) -> str:
        """Play a sound effect and return HTML with audio element.

        This function allows simultaneous playback with voice narration
        by returning HTML that auto-plays the sound effect.

        Args:
            effect_name: Name of the sound effect to play

        Returns:
            HTML string with audio element set to autoplay

        """
        sound_path = sound_effects.play_sound_effect(effect_name)
        if not sound_path:
            return ""
        # Embed sound effect as base64 to avoid missing static file routes
        try:
            with open(sound_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode('utf-8')
            # Use data URI for audio playback
            return f'<audio src="data:audio/mpeg;base64,{b64}" autoplay style="display:none"></audio>'
        except Exception as e:
            logger.warning(f"Failed to embed sound effect '{effect_name}': {e}")
            return ""

    with gr.Blocks(title="Voice Chat with OpenAI") as interface:
        gr.Markdown("# Voice Chat with OpenAI")
        gr.Markdown(
            "Speak into the microphone or type your message to get a voice response."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Your Voice Input",
                )
                text_output = gr.Textbox(label="Transcribed Text", interactive=False)

                text_input = gr.Textbox(
                    label="Or Type Your Message Here",
                    placeholder="Type your message and press Enter...",
                )

                submit_button = gr.Button("Submit")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="AI Voice Response")
                sound_effect_output = gr.HTML(label="Sound Effect")

                gr.Markdown("## Sound Effects")
                sound_buttons = []

                with gr.Row():
                    for effect_name in available_effects:
                        button = gr.Button(f"{effect_name.capitalize()}")
                        sound_buttons.append((button, effect_name))

        chat_history = gr.Chatbot(label="Chat History")

        audio_input.change(
            fn=voice_chat.process_voice_input,
            inputs=audio_input,
            outputs=[text_output, audio_output, chat_history],
        )

        text_input.submit(
            fn=voice_chat.process_text_input,
            inputs=text_input,
            outputs=[text_output, audio_output, chat_history],
        )

        submit_button.click(
            fn=voice_chat.process_text_input,
            inputs=text_input,
            outputs=[text_output, audio_output, chat_history],
        )

        for button, effect_name in sound_buttons:
            button.click(
                fn=lambda name=effect_name: play_sound_effect(name),
                inputs=None,
                outputs=sound_effect_output,
            )

    return interface
