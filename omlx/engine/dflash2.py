# SPDX-License-Identifier: Apache-2.0
"""Small adapter from the DFlash 2 MLX runtime to oMLX DFlash events."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..patches.dflash2.vendor import model_mlx as dflash2_mlx

logger = logging.getLogger(__name__)


def read_model_config(model_ref: str | Path) -> dict[str, Any]:
    """Read a local Hugging Face config without resolving remote model IDs."""
    config_path = Path(model_ref) / "config.json"
    if not config_path.is_file():
        return {}
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def is_dflash2_draft(model_ref: str | Path) -> bool:
    """Detect DFlash 2 by its checkpoint architecture, then by Hub name."""
    config = read_model_config(model_ref)
    if config:
        architectures = config.get("architectures") or []
        return "DFlash2DraftModel" in architectures
    name = str(model_ref).rstrip("/").rsplit("/", 1)[-1]
    return re.search(r"(?:^|[-_])dflash2(?:$|[-_])", name, re.IGNORECASE) is not None


def _is_quantized_target(model_ref: str | Path, config: dict[str, Any]) -> bool:
    if config.get("quantization") or config.get("quantization_config"):
        return True
    name = Path(str(model_ref).rstrip("/")).name
    return (
        re.search(
            r"(?:^|[-_])(\d+bit|int\d+|fp\d+|nvfp\d+|oq\d+)(?:$|[-_])",
            name,
            re.IGNORECASE,
        )
        is not None
    )


def load_models(
    target_ref: str,
    draft_ref: str,
    *,
    draft_quant_enabled: bool,
    draft_quant_weight_bits: int | None,
    draft_quant_activation_bits: int | None,
    draft_quant_group_size: int | None,
) -> tuple[Any, Any, Any, dict[str, Any], int | None]:
    """Load and optionally quantize an official DFlash 2 MLX model pair."""
    target_model, tokenizer = dflash2_mlx.load(target_ref)
    draft_model = dflash2_mlx.load_draft(draft_ref)

    if draft_quant_enabled:
        from mlx import nn

        weight_bits = draft_quant_weight_bits or 4
        group_size = draft_quant_group_size or 64
        if draft_quant_activation_bits not in (None, 16):
            logger.warning(
                "DFlash 2 supports weight-only draft quantization; "
                "ignoring activation_bits=%s",
                draft_quant_activation_bits,
            )
        nn.quantize(draft_model, group_size=group_size, bits=weight_bits)
        mx.eval(draft_model.parameters())

    target_config = read_model_config(target_ref)
    block_size = None
    if draft_quant_enabled or _is_quantized_target(target_ref, target_config):
        # The reference runtime recommends at most five candidates with MLX's
        # current quantized matmul kernel.
        block_size = min(5, int(draft_model.config.block_size))

    return target_model, tokenizer, draft_model, target_config, block_size


def stream_events(
    *,
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    prefill_step_size: int,
    block_size: int | None,
) -> Iterator[Any]:
    """Expose DFlash 2 responses through dflash-mlx's stable event protocol."""
    from dflash_mlx.engine.events import SummaryEvent, TokenEvent

    started_at = time.perf_counter()
    generated: list[int] = []
    accepted_from_draft = 0
    acceptance_history: list[int] = []
    cycles = 0
    peak_memory_gb: float | None = None

    responses = dflash2_mlx.stream_generate(
        target_model,
        draft_model,
        tokenizer,
        prompt_tokens,
        block_size=block_size,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        prefill_step_size=prefill_step_size,
    )

    for response in responses:
        response_tokens = [int(token) for token in response.tokens]
        if response.accepted is not None and response_tokens:
            # Each speculative cycle commits accepted draft tokens followed by
            # one target-owned bonus token. The reference response's
            # ``accepted`` field is the total committed width.
            accepted = max(0, min(len(response_tokens), int(response.accepted)) - 1)
            accepted_from_draft += accepted
            acceptance_history.append(accepted)
            cycles += 1

        peak_memory_gb = float(response.peak_memory)
        for token_id in response_tokens:
            generated.append(token_id)
            ratio = accepted_from_draft / len(generated) if generated else 0.0
            yield TokenEvent(
                token_id=token_id,
                generated_tokens=len(generated),
                acceptance_ratio=ratio,
                cycles_completed=cycles,
            )

        if response.finish_reason is not None:
            break

    elapsed_us = (time.perf_counter() - started_at) * 1e6
    generation_tokens = len(generated)
    acceptance_ratio = (
        accepted_from_draft / generation_tokens if generation_tokens else 0.0
    )
    yield SummaryEvent(
        elapsed_us=elapsed_us,
        prompt_token_count=len(prompt_tokens),
        generated_token_ids=tuple(generated),
        generation_tokens=generation_tokens,
        accepted_from_draft=accepted_from_draft,
        acceptance_ratio=acceptance_ratio,
        cycles_completed=cycles,
        phase_timings_us={},
        block_tokens=(
            block_size if block_size is not None else int(draft_model.config.block_size)
        ),
        tokens_per_cycle=(generation_tokens / cycles if cycles else 0.0),
        acceptance_history=tuple(acceptance_history),
        peak_memory_gb=peak_memory_gb,
    )
