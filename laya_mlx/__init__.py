"""Laya typed decisions on Apple silicon with MLX."""

try:
    import mlx.core  # noqa: F401

    _has_mlx = True
except ImportError:
    _has_mlx = False

if _has_mlx:
    from .agent import Agent, RLAgent, load
    from .quantize import DEFAULT_GROUP_SIZE, SUPPORTED_BITS, export_quantized, quantize_model
    from .router import DEFAULT_MODELS, RouteDecision, Router
    from .server import DecisionHandler, serve
else:
    Agent = RLAgent = load = None  # type: ignore[assignment, misc]
    DEFAULT_MODELS = RouteDecision = Router = None  # type: ignore[assignment, misc]
    DecisionHandler = serve = None  # type: ignore[assignment, misc]
    DEFAULT_GROUP_SIZE = 64
    SUPPORTED_BITS = (4, 8)
    export_quantized = quantize_model = None  # type: ignore[assignment]

from .device import (
    HardwareProfile,
    get_chip_name,
    get_total_memory_gb,
    is_apple_silicon,
    profile_hardware,
    suggest_device_config,
)
from .email import clean_email_body, email_state
from .lang import analyse as detect_language
from .lang import detect_script, is_english
from .presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "RLAgent",
    "load",
    "Router",
    "RouteDecision",
    "DEFAULT_MODELS",
    "detect_language",
    "detect_script",
    "is_english",
    "clean_email_body",
    "email_state",
    "email_questions",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
    "quantize_model",
    "export_quantized",
    "SUPPORTED_BITS",
    "DEFAULT_GROUP_SIZE",
    "HardwareProfile",
    "profile_hardware",
    "suggest_device_config",
    "get_total_memory_gb",
    "get_chip_name",
    "is_apple_silicon",
    "serve",
    "DecisionHandler",
]
