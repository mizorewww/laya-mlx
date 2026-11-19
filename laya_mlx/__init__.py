"""Laya: Fast, non-autoregressive System 1 decision engine with calibrated probabilities."""

import os as _os

# Tell transformers not to import TensorFlow, before anything can import transformers.
#
# Laya's model path is torch-only, but on transformers 4.x `build_model` reaches
# `transformers.modeling_utils` for `no_init_weights`, and importing that module pulls a chain
# (`loss_utils` -> `loss_d_fine` -> `loss_for_object_detection` -> `image_transforms`) that ends in
# `import tensorflow`. In an environment where TensorFlow is installed but cannot load, that is a
# native crash rather than an exception -- `Fatal Python error: Bus error` -- from a library Laya
# never uses (#915).
#
# `setdefault`, not assignment: a caller who set `USE_TF` to something deliberate keeps it, and one
# who wants TensorFlow in the same process can still have it by setting the variable themselves. CI,
# the Dockerfile and the test suite already set `USE_TF=0` for exactly this reason; this makes the
# package behave the same way without the caller having to know.
#
# It has to be here, above the imports below, because transformers decides TensorFlow's
# availability when it is first imported -- and `import laya` deliberately imports no transformers
# (the torch-backed names are lazy, see `_LAZY_ATTRS`), so this runs first in the normal case.
_os.environ.setdefault("USE_TF", "0")

from .email import clean_email_body, email_state
from .hooks import AsyncHook, BaseHook, Hook, OptionStabilityHook, PredictContext, PredictHook
from .lang import analyse as detect_language
from .lang import detect_script, is_english
from .presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .router import DEFAULT_MODELS, RouteDecision, Router
from .structured import DecisionResult, decide, decide_batch

__version__ = "0.7.2"

# Routing, language detection and email cleaning are pure Python. The torch-backed names are
# resolved lazily so that `import laya` -- and therefore `from laya_mlx import Router` or
# `from laya_mlx.lang import detect_script` -- does not pay torch's import time and memory.
_LAZY_ATTRS = {
    # stdlib-only module, but keep it lazy so `import laya` does not pull in hashlib
    "PINNED_REVISIONS": (".revisions", "PINNED_REVISIONS"),
    "Agent": (".agent", "Agent"),
    "RLAgent": (".agent", "RLAgent"),
    "load": (".agent", "load"),
    "fit_temperatures": (".calibrate", "fit_temperatures"),
    "fit_one_temperature": (".calibrate", "fit_one_temperature"),
    "fit_temperature_map": (".calibrate", "fit_temperature_map"),
    "fit_abstention_thresholds": (".calibrate", "fit_abstention_thresholds"),
    "fit_binning_map": (".calibrate", "fit_binning_map"),
    "apply_binning_map": (".calibrate", "apply_binning_map"),
    "jev_confidence": (".confidence", "jev_confidence"),
    "proper_reward": (".common", "proper_reward"),
    "td_lambda_targets": (".common", "td_lambda_targets"),
    "ece_score": (".common", "ece_score"),
    "answer_confidence": (".common", "answer_confidence"),
    "confidence_from_probs": (".common", "confidence_from_probs"),
    "check_min_confidence": (".confidence", "check_min_confidence"),
    "check_min_confidence_map": (".confidence", "check_min_confidence_map"),
    "resolve_min_confidence": (".confidence", "resolve_min_confidence"),
    "flag_low_confidence": (".confidence", "flag_low_confidence"),
    "apply_confidence_gate": (".confidence", "apply_confidence_gate"),
    "GATE_STATES": (".confidence", "GATE_STATES"),
    "render_options": (".common", "render_options"),
    "QTYPES": (".common", "QTYPES"),
    "QTYPE_NAMES": (".common", "QTYPE_NAMES"),
    "shortlist_choice": (".shortlist", "shortlist_choice"),
    "predict_shortlist": (".shortlist", "predict_shortlist"),
    "predict_tournament": (".shortlist", "predict_tournament"),
    "embed_fn_from_agent": (".shortlist", "embed_fn_from_agent"),
    "cached_embed_fn": (".shortlist", "cached_embed_fn"),
    "LayaRouter": (".integrations", "LayaRouter"),
    "LayaGuardrail": (".integrations", "LayaGuardrail"),
    "LayaGuardrailError": (".integrations", "LayaGuardrailError"),
    "LayaTriage": (".integrations", "LayaTriage"),
    "LayaEvaluator": (".integrations", "LayaEvaluator"),
    "LayaDecision": (".integrations", "LayaDecision"),
}


def __getattr__(name):
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError("module %r has no attribute %r" % (__name__, name)) from None
    import importlib

    value = getattr(importlib.import_module(module_name, __name__), attr)
    globals()[name] = value  # cache: __getattr__ runs at most once per name
    return value


def __dir__():
    return sorted(list(globals()) + list(_LAZY_ATTRS))


__all__ = [
    "Agent",
    "RLAgent",
    "load",
    "fit_temperatures",
    "fit_one_temperature",
    "fit_temperature_map",
    "fit_abstention_thresholds",
    "fit_binning_map",
    "apply_binning_map",
    "Router",
    "RouteDecision",
    "DEFAULT_MODELS",
    "shortlist_choice",
    "predict_shortlist",
    "predict_tournament",
    "embed_fn_from_agent",
    "cached_embed_fn",
    "detect_language",
    "detect_script",
    "is_english",
    "clean_email_body",
    "email_questions",
    "email_state",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
    "proper_reward",
    "td_lambda_targets",
    "ece_score",
    "answer_confidence",
    "confidence_from_probs",
    "check_min_confidence",
    "check_min_confidence_map",
    "resolve_min_confidence",
    "flag_low_confidence",
    "apply_confidence_gate",
    "GATE_STATES",
    "render_options",
    "QTYPES",
    "QTYPE_NAMES",
    "LayaRouter",
    "LayaGuardrail",
    "LayaGuardrailError",
    "LayaTriage",
    "LayaEvaluator",
    "LayaDecision",
    "PredictContext",
    "PredictHook",
    "Hook",
    "BaseHook",
    "OptionStabilityHook",
    "jev_confidence",
    "AsyncHook",
    "decide",
    "decide_batch",
    "DecisionResult",
    "PINNED_REVISIONS",
    "__version__",
    # Submodules for direct import paths
    "confidence",
    "email",
    "hooks",
    "lang",
    "presets",
    "router",
    "structured",
]
