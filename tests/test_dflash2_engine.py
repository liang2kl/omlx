# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for the DFlash 2 compatibility path."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx", reason="MLX is required for DFlash 2")

from omlx.engine import dflash2


def test_detects_dflash2_from_local_architecture(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["DFlash2DraftModel"]}),
        encoding="utf-8",
    )

    assert dflash2.is_dflash2_draft(tmp_path)


def test_local_architecture_wins_over_checkpoint_name(tmp_path):
    draft = tmp_path / "renamed-DFlash2"
    draft.mkdir()
    (draft / "config.json").write_text(
        json.dumps({"architectures": ["DFlashDraftModel"]}),
        encoding="utf-8",
    )

    assert not dflash2.is_dflash2_draft(draft)


def test_detects_remote_dflash2_name():
    assert dflash2.is_dflash2_draft("z-lab/Qwen3.8-27B-DFlash2")
    assert not dflash2.is_dflash2_draft("z-lab/Qwen3.6-27B-DFlash")


def test_stream_events_adapts_responses_and_sampling(monkeypatch):
    pytest.importorskip("dflash_mlx", reason="dflash-mlx is required by oMLX")
    captured = {}

    responses = [
        SimpleNamespace(
            tokens=[10], accepted=None, peak_memory=1.0, finish_reason=None
        ),
        SimpleNamespace(
            tokens=[11, 12, 13], accepted=3, peak_memory=1.5, finish_reason=None
        ),
        SimpleNamespace(
            tokens=[], accepted=None, peak_memory=1.5, finish_reason="length"
        ),
    ]

    def fake_stream_generate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return iter(responses)

    monkeypatch.setattr(dflash2.dflash2_mlx, "stream_generate", fake_stream_generate)
    draft = SimpleNamespace(config=SimpleNamespace(block_size=16))

    events = list(
        dflash2.stream_events(
            target_model="target",
            draft_model=draft,
            tokenizer="tokenizer",
            prompt_tokens=[1, 2, 3],
            max_tokens=32,
            temperature=0.7,
            top_p=0.95,
            top_k=20,
            prefill_step_size=1024,
            block_size=5,
        )
    )

    from dflash_mlx.engine.events import SummaryEvent, TokenEvent

    token_events = [event for event in events if isinstance(event, TokenEvent)]
    summary = next(event for event in events if isinstance(event, SummaryEvent))
    assert [event.token_id for event in token_events] == [10, 11, 12, 13]
    assert summary.prompt_token_count == 3
    assert summary.generation_tokens == 4
    assert summary.accepted_from_draft == 2
    assert summary.acceptance_ratio == 0.5
    assert summary.cycles_completed == 1
    assert summary.block_tokens == 5
    assert summary.acceptance_history == (2,)
    assert captured["kwargs"] == {
        "block_size": 5,
        "max_tokens": 32,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "prefill_step_size": 1024,
    }


def test_engine_disables_v1_snapshot_cache_reporting_for_dflash2(tmp_path):
    from omlx.engine.dflash import DFlashEngine
    from omlx.model_settings import ModelSettings

    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["DFlash2DraftModel"]}),
        encoding="utf-8",
    )
    engine = DFlashEngine(
        model_name="target",
        draft_model_path=str(tmp_path),
        model_settings=ModelSettings(
            dflash_in_memory_cache=True,
            dflash_ssd_cache=True,
        ),
        omlx_ssd_cache_dir=tmp_path,
    )

    stats = engine.get_stats()
    assert stats["dflash_version"] == 2
    assert stats["in_memory_cache"] is False
    assert stats["ssd_cache"] is False
    assert engine.get_runtime_cache_stats() is None


@pytest.mark.asyncio
async def test_start_selects_dflash2_loader_without_v1_runtime(monkeypatch, tmp_path):
    pytest.importorskip("dflash_mlx", reason="dflash-mlx is required by oMLX")
    from omlx.engine import dflash as dflash_engine_module
    from omlx.engine.dflash import DFlashEngine
    from omlx.patches import qwen35_moe_gate_up

    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["DFlash2DraftModel"]}),
        encoding="utf-8",
    )
    target = SimpleNamespace()
    tokenizer = SimpleNamespace(
        name_or_path="target",
        eos_token_id=2,
        eos_token_ids=[2],
    )
    draft = SimpleNamespace(config=SimpleNamespace(block_size=16))
    load_models = MagicMock(
        return_value=(target, tokenizer, draft, {"model_type": "qwen3_5"}, 5)
    )
    monkeypatch.setattr(dflash2, "load_models", load_models)
    monkeypatch.setattr(
        dflash_engine_module,
        "maybe_apply_pre_load_patches",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        dflash_engine_module,
        "load_generation_config_token_ids",
        lambda *args, **kwargs: set(),
    )
    monkeypatch.setattr(
        dflash_engine_module, "detect_output_parser", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        dflash_engine_module, "set_model_info_from_model", lambda *args: None
    )
    monkeypatch.setattr(
        qwen35_moe_gate_up,
        "apply_qwen35_moe_gate_up_fusion",
        lambda _model: None,
    )

    engine = DFlashEngine(
        model_name="target",
        draft_model_path=str(tmp_path),
        draft_quant_enabled=True,
        draft_quant_weight_bits=4,
        draft_quant_activation_bits=16,
        draft_quant_group_size=64,
    )
    engine._build_runtime_context = MagicMock(
        side_effect=AssertionError("DFlash 2 must not build the v1 runtime")
    )

    await engine.start()
    try:
        assert engine._loaded
        assert engine._dflash_version == 2
        assert engine._target_model is target
        assert engine._draft_model is draft
        assert engine._runtime_context is None
        assert engine._dflash2_block_size == 5
        load_models.assert_called_once_with(
            "target",
            str(tmp_path),
            draft_quant_enabled=True,
            draft_quant_weight_bits=4,
            draft_quant_activation_bits=16,
            draft_quant_group_size=64,
        )
        engine._build_runtime_context.assert_not_called()
    finally:
        await engine.stop()
