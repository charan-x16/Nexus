from typing import Any

from backend.config import settings

_checkpointer: Any | None = None
_checkpointer_context: Any | None = None


def setup_checkpointer(connection_string: str) -> Any:
    global _checkpointer, _checkpointer_context
    if _checkpointer is not None:
        return _checkpointer

    from langgraph.checkpoint.postgres import PostgresSaver

    saver_or_context = PostgresSaver.from_conn_string(connection_string)
    if hasattr(saver_or_context, "__enter__"):
        _checkpointer_context = saver_or_context
        _checkpointer = _checkpointer_context.__enter__()
    else:
        _checkpointer = saver_or_context

    if hasattr(_checkpointer, "create_tables"):
        _checkpointer.create_tables()
    elif hasattr(_checkpointer, "setup"):
        _checkpointer.setup()
    return _checkpointer


def get_checkpointer() -> Any:
    return setup_checkpointer(settings.DATABASE_URL)


def close_checkpointer() -> None:
    global _checkpointer, _checkpointer_context
    if _checkpointer_context is not None and hasattr(_checkpointer_context, "__exit__"):
        _checkpointer_context.__exit__(None, None, None)
    _checkpointer = None
    _checkpointer_context = None
