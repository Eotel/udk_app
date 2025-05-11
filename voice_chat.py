"""Voice chat implementation using OpenAI API and Gradio."""

import base64
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import openai
from jinja2 import Environment, FileSystemLoader, TemplateError, select_autoescape
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
        # Setup Jinja2 environment for prompt templates
        # Setup Jinja2 environment for prompt templates (relative to this file)
        base_dir = Path(__file__).parent
        prompts_dir = base_dir / "prompts"
        if not prompts_dir.exists():
            logger.warning(f"Prompts directory not found at {prompts_dir}")
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=select_autoescape()
        )
        # Discover available prompt templates
        self.prompt_templates = []
        if prompts_dir.exists() and prompts_dir.is_dir():
            self.prompt_templates = [p.stem for p in prompts_dir.glob("*.j2") if p.is_file()]

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

    def process_input(self, user_text: str) -> tuple[str, str, list[dict]]:
        """Process user input (voice or text) and return the response and chat history."""
        # Generate assistant response and add to conversation state
        response_text = self.generate_response(user_text)
        # Synthesize speech for the response
        audio_output = self.synthesize_speech(response_text)

        # Build chat history entries for UI, excluding system prompts
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in self.state.conversation_history
            if msg.role in ("user", "assistant")
        ]
        return user_text, audio_output, chat_history

    def load_prompt(self, template_name: str, context: dict | None = None) -> None:
        """Switch prompt template and reset conversation state."""
        template_file = f"{template_name}.j2"
        try:
            template = self.jinja_env.get_template(template_file)
            rendered = template.render(**(context or {}))
        except TemplateError as e:
            logger.error(f"Failed to load prompt template '{template_name}': {e}")
            return
        # Reset conversation history with a system message containing the prompt
        self.state.conversation_history = [
            ChatMessage(role="system", content=rendered)
        ]
        logger.info(f"Prompt template '{template_name}' loaded")


def create_voice_chat_interface() -> gr.Blocks:
    """Create a Gradio interface for voice chat."""
    voice_chat = VoiceChat()
    sound_effects = voice_chat.sound_effects
    available_effects = sound_effects.get_available_sound_effects()
    # Prompt template selection setup
    available_prompts = voice_chat.prompt_templates
    default_prompt = available_prompts[0] if available_prompts else None
    # Load default prompt if available
    if default_prompt:
        voice_chat.load_prompt(default_prompt)

    def select_prompt(template_name: str) -> list:
        """Switch prompt template and reset chat history."""
        voice_chat.load_prompt(template_name)
        # Return empty chat history to clear UI
        return []

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
            # Read binary data from file
            data = Path(sound_path).read_bytes()
        except OSError as e:
            logger.warning(f"Failed to read sound effect '{effect_name}': {e}")
            return ""
        # Encode as base64 and use data URI for audio playback
        b64 = base64.b64encode(data).decode("utf-8")
        # Include a timestamp attribute to force re-render on repeated clicks
        ts = int(time.time() * 1000)
        return (
            f'<audio src="data:audio/mpeg;base64,{b64}" '
            f'data-ts="{ts}" autoplay style="display:none"></audio>'
        )

    with gr.Blocks(title="Voice Chat with OpenAI") as interface:
        gr.Markdown("# Voice Chat with OpenAI")
        gr.Markdown(
            "Speak into the microphone or type your message to get a voice response."
        )
        # Prompt template dropdown
        if available_prompts:
            prompt_selector = gr.Dropdown(
                choices=available_prompts,
                value=default_prompt,
                label="Prompt Template",
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

        chat_history = gr.Chatbot(label="Chat History", type="messages")
        # Connect prompt selector to clear and load new prompt
        if available_prompts:
            prompt_selector.change(
                fn=select_prompt,
                inputs=[prompt_selector],
                outputs=[chat_history],
            )

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
