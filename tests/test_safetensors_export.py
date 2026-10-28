# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from transformers import GenerationConfig, PretrainedConfig

from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer


class _BufferExportMapper:
    """Include native buffers through the export mapper's tensor iterator."""

    def __init__(self, model):
        """Bind the model whose buffers accompany the first parameter."""
        self.model = model

    def policy_map_local_key_to_hf_key(self, name):
        """Keep native tensor names in the serialized checkpoint."""
        return name

    def policy_map_local_key_for_export_tensor(self, name, param):
        """Yield parameters and include each buffer exactly once."""
        yield name, param
        if name == "weight":
            yield from self.model.named_buffers()


@pytest.mark.parametrize("export_dtype", [None, torch.float32, torch.bfloat16])
def test_safetensors_export_preserves_discrete_buffers(tmp_path, export_dtype):
    """Floating conversion must not round integer buffers or change boolean dtype."""
    model = torch.nn.Module()
    model.register_parameter(
        "weight", torch.nn.Parameter(torch.tensor([1.003, -0.127], dtype=torch.float32))
    )
    model.register_parameter(
        "low_precision",
        torch.nn.Parameter(torch.tensor([0.75, -2.5], dtype=torch.bfloat16)),
    )
    model.register_buffer(
        "num_batches_tracked", torch.tensor(2**40 + 1, dtype=torch.int64)
    )
    model.register_buffer("enabled", torch.tensor([True, False]))
    model.weight_mapper = _BufferExportMapper(model)
    initial_state = {
        name: tensor.clone() for name, tensor in model.state_dict().items()
    }

    source = tmp_path / "source"
    GenerationConfig().save_pretrained(source)
    trainer = object.__new__(GRPOTrainer)
    trainer.model = model
    trainer.hf_config = PretrainedConfig()
    trainer.data_packer = None
    trainer.upload_thread = None
    trainer.global_rank = 0
    trainer.parallel_dims = SimpleNamespace(
        dp_replicate_coord=(0, 1),
        dp_replicate_enabled=True,
        dp_shard_coord=(0, 1),
        cp_coord=(0, 1),
        tp_coord=(0, 1),
        pp_coord=(0, 1),
        pp_enabled=False,
    )
    trainer.config = SimpleNamespace(
        policy=SimpleNamespace(
            lora=None, model_name_or_path=str(source), model_revision=None
        ),
        train=SimpleNamespace(ckpt=SimpleNamespace(upload_hf=False, upload_s3=False)),
    )

    trainer.export_safetensors(str(tmp_path), "export", dtype=export_dtype)
    trainer.upload_thread.join(timeout=30)
    assert not trainer.upload_thread.is_alive()
    exported = load_file(tmp_path / "export" / "00000.safetensors")

    assert exported.keys() == initial_state.keys()
    for name, tensor in initial_state.items():
        expected = (
            tensor.to(export_dtype)
            if tensor.is_floating_point() and export_dtype is not None
            else tensor
        )
        torch.testing.assert_close(exported[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0)
