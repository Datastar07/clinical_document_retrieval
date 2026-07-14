from __future__ import annotations

"""Shared TF/transformers guards used before heavy model imports."""

import os


def apply_tf_guards() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import transformers.utils.import_utils as _iu

        _iu.is_tf_available = lambda: False  # type: ignore[method-assign]
    except Exception:
        pass


apply_tf_guards()
