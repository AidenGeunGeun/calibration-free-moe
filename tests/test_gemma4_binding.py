import json
from pathlib import Path

from calibration_free_moe.model_config import model_override


def test_gemma_conditional_override_includes_inner_causal_model(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "text_config": {"top_k_experts": 8, "moe_intermediate_size": 2048},
            }
        )
    )
    assert model_override(str(tmp_path)) == {
        "Gemma4ForConditionalGeneration": (
            "calibration_free_moe.runtime.gemma4:ProgressiveGemma4ForConditionalGeneration"
        ),
        "Gemma4ForCausalLM": (
            "calibration_free_moe.runtime.gemma4:ProgressiveGemma4ForCausalLM"
        ),
    }
