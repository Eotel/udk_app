"""Voice chat implementation using OpenAI API and Gradio."""
from __future__ import annotations

import base64
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
from settings import settings
from sound_effects import SoundEffects


class VoiceChat:
    """Voice chat implementation using OpenAI API."""

    def __init__(self) -> None:
        """Initialize the voice chat with OpenAI API key."""
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.state = VoiceChatState()
        self.sound_effects = SoundEffects()
        if not settings.prompts_dir.exists():
            logger.warning(f"Prompts directory not found at {settings.prompts_dir}")
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(settings.prompts_dir)),
            autoescape=select_autoescape(),
        )
        # Discover available prompt templates
        self.prompt_templates = []
        if settings.prompts_dir.exists() and settings.prompts_dir.is_dir():
            self.prompt_templates = [
                p.stem for p in settings.prompts_dir.glob("*.j2") if p.is_file()
            ]

        # Load TTS instructions from template if available
        self.tts_instructions = ""
        if "instructions" in self.prompt_templates:
            try:
                template = self.jinja_env.get_template("instructions.j2")
                self.tts_instructions = template.render()
                logger.info("Loaded TTS instructions template")
            except TemplateError as e:
                logger.error(f"Failed to load TTS instructions template: {e}")

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using OpenAI API."""
        logger.info(f"Transcribing audio from {audio_path}")
        request = TranscriptionRequest(
            file_path=audio_path, model=settings.transcription_model
        )

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

        # Prepare chat request parameters
        request = ChatRequest(
            messages=self.state.conversation_history,
            model=settings.chat_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        # Convert conversation history into Responses API input items
        input_items = [
            {
                "role": msg.role,
                "content": [{"text": msg.content, "type": "input_text"}],
                "type": "message",
            }
            for msg in request.messages
        ]
        # Call the unified Responses API for chat completion
        resp = self.client.responses.create(
            model=request.model,
            input=input_items,
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        )
        # Extract the assistant's generated text from the response
        response_text = ""
        for item in getattr(resp, "output", []):
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) == "output_text":
                        response_text = getattr(content, "text", "")
                        break
            if response_text:
                break

        self.state.conversation_history.append(
            ChatMessage(role="assistant", content=response_text),
        )

        logger.info(f"Generated response: {response_text}")
        return response_text

    def synthesize_speech(self, text: str) -> str:
        """Synthesize speech from text using OpenAI API."""
        # Skip synthesis if no text provided
        if not text or not text.strip():
            logger.warning("Empty response text; skipping speech synthesis")
            return None

        logger.info(f"Synthesizing speech for: {text}")
        # Use the loaded instructions template if available, otherwise use settings value
        tts_instructions = self.tts_instructions or settings.tts_instructions

        request = SpeechRequest(
            text=text,
            model=settings.tts_model,
            voice=settings.tts_voice,
            instructions=tts_instructions,
            streaming=settings.tts_streaming,
        )

        temp_dir = Path(tempfile.gettempdir())
        audio_path = temp_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.wav"

        logger.info("Using streaming TTS mode")
        with self.client.audio.speech.with_streaming_response.create(
            model=request.model,
            voice=request.voice,
            input=request.text,
            instructions=request.instructions,
            response_format="wav",
        ) as streaming_response:
            streaming_response.stream_to_file(str(audio_path))

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
        # Synthesize speech for the response (skip if empty)
        audio_output = self.synthesize_speech(response_text) if response_text else None

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
        self.state.conversation_history = [ChatMessage(role="system", content=rendered)]
        logger.info(f"Prompt template '{template_name}' loaded")


def create_voice_chat_interface() -> gr.Blocks:
    """Create a Gradio interface for voice chat."""
    # Initialize per-room VoiceChat instances in a closure
    default_room = "default"
    voice_chats: dict[str, VoiceChat] = {}
    vc_default = VoiceChat()
    available_prompts = vc_default.prompt_templates
    default_prompt = available_prompts[0] if available_prompts else None
    if default_prompt:
        vc_default.load_prompt(default_prompt)
    voice_chats[default_room] = vc_default

    # Get sound effects and available effects from the default instance
    sound_effects = vc_default.sound_effects
    available_effects = sound_effects.get_available_sound_effects()

    def select_prompt(template_name: str, room_name: str) -> list[dict]:
        """Switch prompt template for the current room and clear UI history."""
        # Load the selected prompt into the room's VoiceChat
        vc = voice_chats.get(room_name)
        if vc:
            vc.load_prompt(template_name)
        # Return empty chat history to reset UI
        return []

    # Callback to create a new chat room
    def create_room_fn(new_room: str, cur_room: str) -> tuple[dict, str]:
        if new_room and new_room not in voice_chats:
            vc = VoiceChat()
            if default_prompt:
                vc.load_prompt(default_prompt)
            voice_chats[new_room] = vc
        room = new_room or cur_room
        # Return updated dropdown and new room
        return (
            gr.update(choices=list(voice_chats.keys()), value=room),
            room,
        )

    # Callback to play sound effect for the current room
    def play_sound_effect_room(effect_name: str, room_name: str) -> str:
        vc = voice_chats.get(room_name)
        if not vc:
            return ""
        sound_path = vc.sound_effects.play_sound_effect(effect_name)
        if not sound_path:
            return ""
        data = Path(sound_path).read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        ts = int(time.time() * 1000)
        return (
            f'<audio src="data:audio/mpeg;base64,{b64}" '
            f'data-ts="{ts}" autoplay style="display:none"></audio>'
        )

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
        # State for current room name
        state_room = gr.State(default_room)
        # Chat room selection and creation
        with gr.Row():
            room_selector = gr.Dropdown(
                choices=[default_room],
                value=default_room,
                label="Chat Room",
            )
            new_room_name = gr.Textbox(label="New Room Name")
            new_room_button = gr.Button("Create Room")
            # Wire up room creation
        new_room_button.click(
            fn=create_room_fn,
            inputs=[new_room_name, state_room],
            outputs=[room_selector, state_room],
        )
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
                inputs=[prompt_selector, state_room],
                outputs=[chat_history],
            )

        # Process voice input per room
        audio_input.change(
            fn=lambda audio, room: voice_chats[room].process_voice_input(audio),
            inputs=[audio_input, state_room],
            outputs=[text_output, audio_output, chat_history],
        )

        # Process text input per room
        text_input.submit(
            fn=lambda text, room: voice_chats[room].process_text_input(text),
            inputs=[text_input, state_room],
            outputs=[text_output, audio_output, chat_history],
        )

        submit_button.click(
            fn=lambda text, room: voice_chats[room].process_text_input(text),
            inputs=[text_input, state_room],
            outputs=[text_output, audio_output, chat_history],
        )

        # Sound effect buttons per room
        for button, effect_name in sound_buttons:
            # Each click plays the named effect in the current room
            button.click(
                fn=lambda room, name=effect_name: play_sound_effect_room(name, room),
                inputs=[state_room],
                outputs=sound_effect_output,
            )

    return interface
