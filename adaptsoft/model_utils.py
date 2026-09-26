"""Small helpers shared by the forward and the verl patch."""
from __future__ import annotations


def unwrap_model(model):
    """Peel PEFT / DDP / FSDP wrappers to reach the underlying HF model."""
    m = model
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()
        except Exception:
            pass
    return getattr(m, "module", m)
