# SPDX-License-Identifier: Apache-2.0
"""Model loading helpers with pluggable custom quantization loading."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CustomQuantizationLoader = Callable[[str, bool], tuple[Any, Any, bool]]


def _read_config(model_name: str) -> dict[str, Any] | None:
    """Read local config.json for a model path, if available."""
    config_path = Path(model_name) / "config.json"
    if not config_path.is_file():
        return None

    try:
        with open(config_path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def _detect_custom_quantization_format(model_name: str) -> str | None:
    """Return lowercased custom quantization format from local config.json."""
    config = _read_config(model_name)
    if config is None:
        return None

    qconfig = config.get("quantization_config")
    if not isinstance(qconfig, dict):
        return None

    quant_method = qconfig.get("quant_method")
    if not isinstance(quant_method, str):
        return None

    normalized = quant_method.strip().lower()
    return normalized or None


def _load_mlx_lm_text_model(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
):
    from mlx_lm import load

    return load(model_name, tokenizer_config=tokenizer_config)


def _load_mlx_vlm_model(model_name: str):
    from mlx_vlm.utils import load as vlm_load

    return vlm_load(model_name)


def _load_paroquant_mlx_model(
    model_name: str,
    force_text: bool,
):
    try:
        from paroquant.inference.backends.mlx.load import load as paroquant_load
    except ImportError as e:
        raise ImportError(
            "ParoQuant model detected, but paroquant is not installed. "
            "Install with: pip install 'paroquant[mlx]'"
        ) from e

    return paroquant_load(model_name, force_text=force_text)


_CUSTOM_QUANTIZATION_LOADERS: dict[str, _CustomQuantizationLoader] = {
    "paroquant": _load_paroquant_mlx_model,
}


def _load_via_custom_quantization_loader(
    model_name: str,
    quantization_format: str,
    force_text: bool,
) -> tuple[Any, Any, bool] | None:
    loader = _CUSTOM_QUANTIZATION_LOADERS.get(quantization_format)
    if loader is None:
        return None

    logger.info(
        f"Detected custom quantization format "
        f"'{quantization_format}': {model_name}"
    )
    return loader(model_name, force_text)


def load_text_model(
    model_name: str,
    tokenizer_config: dict[str, Any] | None = None,
):
    """Load an LLM model/tokenizer pair, with custom quantization loading."""
    quantization_format = _detect_custom_quantization_format(model_name)
    if quantization_format is None:
        return _load_mlx_lm_text_model(
            model_name=model_name,
            tokenizer_config=tokenizer_config,
        )

    loaded = _load_via_custom_quantization_loader(
        model_name=model_name,
        quantization_format=quantization_format,
        force_text=True,
    )
    if loaded is None:
        logger.warning(
            f"Custom quantization format '{quantization_format}' "
            "is not registered; "
            "falling back to mlx-lm loader."
        )
        return _load_mlx_lm_text_model(
            model_name=model_name,
            tokenizer_config=tokenizer_config,
        )

    model, processor, _ = loaded
    return model, getattr(processor, "tokenizer", processor)


def load_vlm_model(model_name: str):
    """Load a VLM model/processor pair, with custom quantization loading."""
    quantization_format = _detect_custom_quantization_format(model_name)
    if quantization_format is None:
        return _load_mlx_vlm_model(model_name)

    loaded = _load_via_custom_quantization_loader(
        model_name=model_name,
        quantization_format=quantization_format,
        force_text=False,
    )
    if loaded is None:
        logger.warning(
            f"Custom quantization format '{quantization_format}' "
            "is not registered; "
            "falling back to mlx-vlm loader."
        )
        return _load_mlx_vlm_model(model_name)

    model, processor, is_vlm = loaded
    if not is_vlm:
        raise RuntimeError(
            f"Model '{model_name}' is marked as custom quantization format "
            f"'{quantization_format}' but is not a VLM model."
        )
    return model, processor
