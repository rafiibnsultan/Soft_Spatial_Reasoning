#!/usr/bin/env python3
"""GRPO training entry point for AdaptSoft on the OmniSpatial dataset.

Holds the configuration used for the reported runs and launches `verl.trainer.main_ppo` with it.
Any additional argument is forwarded to verl unchanged.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

# The blend temperature is tau = TAU_BASE + TAU_DELTA * tanh(u); see adaptsoft/controller.py.
TAU_BASE, TAU_DELTA = 0.5, 0.4
# Whitening constants for the normalised top-k entropy fed to the controller, measured once on a
# held-out batch of the base model's rollouts.
H_MEAN, H_STD = 0.173, 0.224
# Qwen3-VL's </think>. The rollout leaves soft mode one step after emitting it.
THINK_END_ID = 151668


def verl_argv(a) -> list[str]:
    return [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.norm_adv_by_std_in_grpo=true",
        f"custom_reward_function.path={SRC_DIR / 'adaptsoft' / 'reward.py'}",
        "custom_reward_function.name=compute_score",

        f"data.train_files={a.train_file}",
        f"data.val_files={a.test_file}",
        "data.image_key=images",
        f"data.train_batch_size={a.train_batch_size}",
        "data.max_prompt_length=10240",
        f"data.max_response_length={a.max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",

        f"actor_rollout_ref.model.path={a.model_name_or_path}",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",

        "actor_rollout_ref.actor.strategy=fsdp2",
        f"actor_rollout_ref.actor.optim.lr={a.learning_rate}",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=cosine",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05",
        "actor_rollout_ref.actor.optim.min_lr_ratio=0.1",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",

        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        f"actor_rollout_ref.rollout.n={a.num_generations}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.6",
        f"actor_rollout_ref.rollout.val_kwargs.top_k={a.soft_top_k}",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",

        # Soft thinking, consumed by the patched SGLang server (engine/sglang/server_args.py).
        "+actor_rollout_ref.rollout.enable_soft_thinking=True",
        f"+actor_rollout_ref.rollout.soft_top_k={a.soft_top_k}",
        f"+actor_rollout_ref.rollout.soft_noise_factor={a.soft_noise_factor}",
        f"+actor_rollout_ref.rollout.soft_think_end_id={THINK_END_ID}",
        # False reproduces fixed-temperature soft thinking, the ablation in the paper.
        f"+actor_rollout_ref.rollout.soft_adaptive_temperature={a.adaptive_temperature}",

        "trainer.logger=console",
        "trainer.project_name=adaptsoft",
        "trainer.experiment_name=adaptsoft_8b",
        f"trainer.n_gpus_per_node={a.num_gpus}",
        f"trainer.nnodes={a.num_nodes}",
        f"trainer.default_local_dir={a.output_dir}",
        "trainer.resume_mode=disable",
        "trainer.save_freq=10",
        "trainer.test_freq=25",
        "trainer.val_before_train=True",
        # verl stops at whichever of the two binds first, so both carry the step budget.
        f"trainer.total_epochs={a.total_steps}",
        f"trainer.total_training_steps={a.total_steps}",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--train_file", required=True)
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--num_nodes", type=int, default=1)
    p.add_argument("--total_steps", type=int, default=200)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--max_response_length", type=int, default=2048)
    p.add_argument("--learning_rate", default="1e-6")
    p.add_argument("--soft_top_k", type=int, default=5)
    p.add_argument("--soft_noise_factor", default="1.0")
    p.add_argument("--adaptive_temperature", default="True",
                   help="False = fixed-temperature soft thinking (the ablation)")
    p.add_argument("--controller_lr", default="1e-3")
    a, extra = p.parse_known_args()

    # adaptsoft is imported by the patched verl FSDP engine and by the patched SGLang model runner,
    # so it has to be importable in both processes.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_DIR)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    os.environ.setdefault("ADAPTSOFT_LR", a.controller_lr)
    os.environ.setdefault("ADAPTSOFT_H_MEAN", str(H_MEAN))
    os.environ.setdefault("ADAPTSOFT_H_STD", str(H_STD))
    os.environ.setdefault("ADAPTSOFT_TAU_BASE", str(TAU_BASE))
    os.environ.setdefault("ADAPTSOFT_TAU_DELTA", str(TAU_DELTA))
    sys.path.insert(0, str(SRC_DIR))

    sys.argv = ["verl.trainer.main_ppo"] + verl_argv(a) + extra
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")


if __name__ == "__main__":
    main()
