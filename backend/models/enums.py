"""
Layer 0 — core enums.

Channel = WHERE the input came from.
Modality = WHAT evidence is available.

These are intentionally kept separate (see design principle in the
Layer 0 spec) so downstream layers reason about evidence, not source.
"""

from enum import Enum


class Channel(str, Enum):
    VOICE_CALL = "voice_call"
    CHATBOT = "chatbot"
    PORTAL = "portal"


class Modality(str, Enum):
    AUDIO = "audio"
    TEXT = "text"
    STRUCTURED_DATA = "structured_data"


# Deterministic, allow-listed channel -> permitted modality supersets.
# A request may use any non-empty subset of its channel's allowed set,
# but may not introduce a modality outside of it (see validator.py).
CHANNEL_ALLOWED_MODALITIES: dict[Channel, set[Modality]] = {
    Channel.VOICE_CALL: {Modality.AUDIO, Modality.TEXT},
    Channel.CHATBOT: {Modality.TEXT},
    Channel.PORTAL: {Modality.TEXT, Modality.STRUCTURED_DATA, Modality.AUDIO},
}

# The modality set a channel produces by default when the caller does not
# (or cannot yet) specify one explicitly — used only as a fallback default,
# never as a business/risk decision.
CHANNEL_DEFAULT_MODALITIES: dict[Channel, list[Modality]] = {
    Channel.VOICE_CALL: [Modality.AUDIO, Modality.TEXT],
    Channel.CHATBOT: [Modality.TEXT],
    Channel.PORTAL: [Modality.TEXT, Modality.STRUCTURED_DATA],
}
