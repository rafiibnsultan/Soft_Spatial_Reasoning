# AdaptSoft — Training Code

## Requirements

```bash
pip install -r requirements.txt
```

The method runs inside the trainer and the rollout engine, so the files in `engine/` replace their
upstream counterparts in the installed `verl` and `sglang` before training will run. See
`engine/README.md` for the path each one replaces.

## Contents

```
adaptsoft_omnispatial_train.py   training entry point
adaptsoft/
  controller.py                  the controller and the temperature map
  grad_align.py                  the per-step gradient alignment objective
  soft_forward.py                training forward with soft embeddings
  soft_logprobs.py               per-token log-probabilities for soft rollouts
  patch.py                       trainer integration: forward, controller update, checkpoints
  reward.py                      the reward function
  system_prompt.py               the system prompt stored in the parquets
engine/                          the modified verl and SGLang files
```

## Data

verl parquet files with the columns `data_source`, `prompt`, `images`, `ability`, `reward_model`
(`{"style": "rule", "ground_truth": "<A-D>"}`) and `extra_info` (`index`, `question`, `options`,
`answer`, `task_type`, `sub_task_type`). One image per row, capped at `128*28*28` pixels.

## Training

Single node, 8 GPUs with 80 GB:

```bash
python adaptsoft_omnispatial_train.py \
  --model_name_or_path Qwen/Qwen3-VL-8B-Thinking \
  --train_file  /path/to/OmniSpatial/train.parquet \
  --test_file   /path/to/OmniSpatial/test.parquet \
  --output_dir  /path/to/checkpoints/adaptsoft \
  --total_steps 200 \
  --num_generations 8
```

Checkpoints are written to `<output_dir>/global_step_<N>/`, holding the sharded policy under
`actor/` and the controller as `adaptsoft_controller.pt`.

Fixed-temperature soft thinking:

```bash
python adaptsoft_omnispatial_train.py ... --adaptive_temperature False
```

Discrete GRPO:

```bash
python adaptsoft_omnispatial_train.py ... +actor_rollout_ref.rollout.enable_soft_thinking=False
```

Any further argument is forwarded to `verl.trainer.main_ppo` unchanged.
