"""Encoder registry for stage 1 models."""

from typing import Callable, Dict, Optional, Type, Union
from pathlib import Path

ARCHS: Dict[str, Type] = {}
__all__ = ["ARCHS", "register_encoder", "get_hf_cache_dir"]


def get_hf_cache_dir() -> Path:
    """Get the project-local HuggingFace cache directory."""
    project_root = Path(__file__).parent.parent.parent.parent
    cache_dir = project_root / "tmp" / "cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _add_to_registry(name: str, cls: Type) -> Type:
    if name in ARCHS and ARCHS[name] is not cls:
        raise ValueError(f"Encoder '{name}' is already registered.")
    ARCHS[name] = cls
    return cls


def register_encoder(cls: Optional[Type] = None, *, name: Optional[str] = None) -> Union[Callable[[Type], Type], Type]:
    """Register an encoder class in ``ARCHS``.

    Can be used either as ``@register_encoder()`` (optionally passing ``name``) or
    via ``register_encoder(MyClass)`` after the class definition.
    """

    def decorator(inner_cls: Type) -> Type:
        encoder_name = name or inner_cls.__name__
        return _add_to_registry(encoder_name, inner_cls)

    if cls is None:
        return decorator

    return decorator(cls)


# Import modules that perform registration on import.
from . import dinov2  
from . import siglip2
from . import mae