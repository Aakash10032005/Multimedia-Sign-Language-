"""Model exports, loaded on demand so the basic demo needs no Torch."""

from importlib import import_module

_MODULES = {
    "SwinStream": ".swin_stream",
    "ViTStream": ".vit_stream",
    "CrossAttentionBridge": ".cross_attention",
    "TemporalLSTM": ".temporal_lstm",
    "HSTFe": ".hstfe",
    "ClassifierHead": ".classifier_head",
    "FallbackRecognizer": ".fallback_recognizer",
}

__all__ = list(_MODULES)


def __getattr__(name):
    """Import a model only when a caller requests that model's class."""
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
