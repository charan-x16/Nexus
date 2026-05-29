import re
import hashlib
import os
import tempfile
from typing import Any

import tiktoken

_ENCODING: Any | None = None
_ENCODING_LOAD_FAILED = False
_CL100K_BPE_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"


def estimate_tokens(text: str) -> int:
    return len(_encode(text or ""))


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    normalized = (text or "").strip()
    if not normalized:
        return []

    pieces = _recursive_split(normalized, ["\n\n", "\n", ". ", " "], chunk_size)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        piece_tokens = estimate_tokens(piece)
        if piece_tokens > chunk_size:
            for token_chunk in _split_by_tokens(piece, chunk_size):
                chunks.append(token_chunk)
            current_parts = []
            current_tokens = 0
            continue

        if current_parts and current_tokens + piece_tokens > chunk_size:
            chunks.append(" ".join(current_parts).strip())
            current_parts, current_tokens = _overlap_seed(chunks[-1], overlap)

        current_parts.append(piece)
        current_tokens += piece_tokens

    if current_parts:
        chunks.append(" ".join(current_parts).strip())

    return [_trim_to_token_limit(chunk, chunk_size) for chunk in chunks if chunk.strip()]


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if estimate_tokens(text) <= chunk_size:
        return [text]
    if not separators:
        return _split_by_tokens(text, chunk_size)

    separator = separators[0]
    parts = text.split(separator)
    if len(parts) == 1:
        return _recursive_split(text, separators[1:], chunk_size)

    split_parts: list[str] = []
    for index, part in enumerate(parts):
        candidate = part.strip()
        if not candidate:
            continue
        if separator == ". " and index < len(parts) - 1:
            candidate = f"{candidate}."
        if estimate_tokens(candidate) <= chunk_size:
            split_parts.append(candidate)
        else:
            split_parts.extend(_recursive_split(candidate, separators[1:], chunk_size))
    return split_parts


def _split_by_tokens(text: str, chunk_size: int) -> list[str]:
    tokens = _encode(text)
    return [_decode(tokens[index : index + chunk_size]).strip() for index in range(0, len(tokens), chunk_size)]


def _overlap_seed(previous_chunk: str, overlap: int) -> tuple[list[str], int]:
    if overlap == 0:
        return [], 0
    tokens = _encode(previous_chunk)
    seed = _decode(tokens[-overlap:]).strip()
    if not seed:
        return [], 0
    return [seed], estimate_tokens(seed)


def _trim_to_token_limit(text: str, chunk_size: int) -> str:
    tokens = _encode(text)
    if len(tokens) <= chunk_size:
        return text
    return _decode(tokens[:chunk_size]).strip()


def _get_encoding() -> Any | None:
    global _ENCODING, _ENCODING_LOAD_FAILED
    if _ENCODING is not None:
        return _ENCODING
    if _ENCODING_LOAD_FAILED:
        return None
    if not _cl100k_cache_exists() and os.getenv("NEXUS_ALLOW_TIKTOKEN_DOWNLOAD") != "1":
        _ENCODING_LOAD_FAILED = True
        return None
    try:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
        return _ENCODING
    except Exception:
        _ENCODING_LOAD_FAILED = True
        return None


def _cl100k_cache_exists() -> bool:
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        cache_dir = os.environ["DATA_GYM_CACHE_DIR"]
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
    if cache_dir == "":
        return False
    cache_key = hashlib.sha1(_CL100K_BPE_URL.encode()).hexdigest()
    return os.path.exists(os.path.join(cache_dir, cache_key))


def _encode(text: str) -> list[Any]:
    encoding = _get_encoding()
    if encoding is not None:
        return encoding.encode(text)
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _decode(tokens: list[Any]) -> str:
    encoding = _get_encoding()
    if encoding is not None and all(isinstance(token, int) for token in tokens):
        return encoding.decode(tokens)
    text = " ".join(str(token) for token in tokens)
    return re.sub(r"\s+([.,!?;:])", r"\1", text).strip()
