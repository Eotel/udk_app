"""Pydantic models for the voice chat application."""

from pydantic import BaseModel, Field


class TranscriptionRequest(BaseModel):
    """Request model for speech-to-text transcription."""

    file_path: str = Field(
        description="Path to the audio file to transcribe",
    )
    model: str = Field(
        default="whisper-1",
        description="The model to use for transcription",
    )


class TranscriptionResponse(BaseModel):
    """Response model for speech-to-text transcription."""

    text: str = Field(
        description="The transcribed text",
    )


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str = Field(
        description="The role of the message sender (system, user, assistant)",
    )
    content: str = Field(
        description="The content of the message",
    )


class ChatRequest(BaseModel):
    """Request model for chat completion."""

    messages: list[ChatMessage] = Field(
        description="The list of messages in the conversation",
    )
    model: str = Field(
        default="gpt-4o",
        description="The model to use for chat completion",
    )
    temperature: float = Field(
        default=0.7,
        description="The sampling temperature for response generation",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int | None = Field(
        default=None,
        description="The maximum number of tokens to generate",
    )


class ChatResponse(BaseModel):
    """Response model for chat completion."""

    message: ChatMessage = Field(
        description="The generated response message",
    )


class SpeechRequest(BaseModel):
    """Request model for text-to-speech synthesis."""

    text: str = Field(
        description="The text to convert to speech",
    )
    model: str = Field(
        default="tts-1",
        description="The model to use for speech synthesis",
    )
    voice: str = Field(
        default="alloy",
        description="The voice to use for speech synthesis",
    )


class SpeechResponse(BaseModel):
    """Response model for text-to-speech synthesis."""

    audio_path: str = Field(
        description="Path to the generated audio file",
    )


class VoiceChatState(BaseModel):
    """State model for the voice chat application."""

    conversation_history: list[ChatMessage] = Field(
        default_factory=list,
        description="The history of the conversation",
    )
