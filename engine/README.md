# Engine files

Modified copies of the verl and SGLang files used for the reported runs. To run the training
script, each one replaces the file at the path below in an installed `verl==0.8.0` /
`sglang==0.5.12`.

| File here | Replaces |
| --- | --- |
| `sglang/server_args.py` | `sglang/srt/server_args.py` |
| `sglang/model_executor/model_runner.py` | `sglang/srt/model_executor/model_runner.py` |
| `sglang/layers/sampler.py` | `sglang/srt/layers/sampler.py` |
| `sglang/managers/tp_worker.py` | `sglang/srt/managers/tp_worker.py` |
| `sglang/managers/mm_utils.py` | `sglang/srt/managers/mm_utils.py` |
| `sglang/managers/scheduler_output_processor_mixin.py` | `sglang/srt/managers/scheduler_output_processor_mixin.py` |
| `verl/fsdp_transformer_impl.py` | `verl/workers/engine/fsdp/transformer_impl.py` |
| `verl/rollout_config.py` | `verl/workers/config/rollout.py` |
| `verl/agent_loop.py` | `verl/experimental/agent_loop/agent_loop.py` |

The server is started with `--disable-overlap-schedule` and `--disable-cuda-graph`, set
automatically when soft thinking is enabled.
