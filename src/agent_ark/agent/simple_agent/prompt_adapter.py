"""Select native prompts for exact, known environment wrapper instructions.

Only the parsed bodies of the shipped wrapper templates are recognized. The
frozen fingerprints deliberately fail closed when a template is customized or
changes upstream. Task text and arbitrary messages are never filtered by tags.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from importlib.resources import files

from .types import Message, TextPart


# SHA-256 of parse_system_task_prompt(template)[0], with LF newlines. Keep
# these snapshots explicit: reading current shared templates as the allowlist
# would silently treat future/custom instructions as safe to replace.
_KNOWN_WRAPPER_PROMPTS = {
    "8dfb42953800b03c380dc1c66b9aa78f222c186cd7784714ed986751e2770c9d": "native_system_func_legacy_think.txt",
    "99e47d2df7199b5c0ac28170d96a5de538b8397bc51b24ab40e28f7ea0d7a50f": "native_system_code_legacy_think.txt",
    "2d02cda2e5768e9543139ee5b35a03bfe7902b8e0ccb9d80d30f5e36f4b8ff3f": "native_system_legacy_legacy_think.txt",
    "fa1ba3f05659c76c998c00639255df58701dbd9cfc708f6bb1903ff0ff8491fe": "native_system_func_brief_rationale.txt",
    "54b7108943ab691eb528ab87f076f5ac1a00489a1b23f0d4738c8925e9bb0225": "native_system_code_brief_rationale.txt",
    "4ee04ad559763dc3f6debf847b1a6a15cb1b8dc5dd64edb67449cb8abc25a5c9": "native_system_legacy_brief_rationale.txt",
    "f3ba12b65eae2cc31131e4865039da4ed2ed2b98b83219ae6f327f54ab689159": "native_system_func_zh.txt",
}


@lru_cache(maxsize=7)
def _native_prompt(name: str) -> str:
    return files(__package__).joinpath("info", name).read_text(encoding="utf-8")


def adapt_native_environment_pin(message: Message) -> Message:
    """Replace a known standalone wrapper pin, preserving every other message.

    The caller chooses the native-tools path. Role alone never authorizes a
    rewrite: the message must have one text part whose entire body matches a
    frozen template. CRLF/LF differences are the only normalization permitted.
    """
    if (message.role not in ("system", "developer") or len(message.parts) != 1
        or not isinstance(message.parts[0], TextPart)):
        return message
    text = message.parts[0].text.replace("\r\n", "\n")
    name = _KNOWN_WRAPPER_PROMPTS.get(sha256(text.encode("utf-8")).hexdigest())
    return Message.text(message.role, _native_prompt(name)) if name is not None else message
