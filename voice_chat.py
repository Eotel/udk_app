"""Voice chat implementation using OpenAI API and Gradio."""

import os
import tempfile
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


class VoiceChat:
    """Voice chat implementation using OpenAI API."""

    def __init__(self) -> None:
        """Initialize the voice chat with OpenAI API key."""
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.state = VoiceChatState()

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

    with gr.Blocks(title="Voice Chat with OpenAI") as interface:
        gr.Markdown("# Voice Chat with OpenAI")
        gr.Markdown("Speak into the microphone and get a voice response.")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Your Voice Input",
                )
                text_output = gr.Textbox(label="Transcribed Text")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="AI Voice Response")

        chat_history = gr.Chatbot(label="Chat History")

        audio_input.change(
            fn=voice_chat.process_voice_input,
            inputs=audio_input,
            outputs=[text_output, audio_output, chat_history],
        )

    return interface
