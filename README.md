# Soft Spatial Reasoning

Official implementation of **Soft Spatial Reasoning**, a post-training framework that enables vision-language models to reason with adaptive continuous intermediate states on spatial tasks.

## Abstract

Large Vision-Language Models (LVLMs) commonly perform spatial reasoning through chain-of-thought, encoding intermediate reasoning as autoregressive sequences of discrete language tokens. This hard thinking requires committing to a single token at each step, even when the correct spatial interpretation remains uncertain. We propose **Soft Spatial Reasoning**, a post-training framework that introduces soft thinking for spatial tasks in LVLMs. At each intermediate reasoning step, the model forms a continuous soft state by mixing token embeddings rather than selecting a single token, allowing multiple candidate continuations to influence the next step.

The degree of softness can vary across reasoning steps. **AdaptSoft** uses the current hidden state and predictive uncertainty to set the temperature of each soft state. We train the controller with **gradient-alignment learning**, which provides step-specific supervision without requiring intermediate reasoning annotations. Across OmniSpatial, SpatiaLab, and MindCube, Soft Spatial Reasoning improves over hard and fixed-soft chain-of-thought baselines using the same backbone.

## Architecture

![Soft Spatial Reasoning overview](figures/figure2.png)

Soft thinking carries multiple candidate continuations through the reasoning trace, then switches to discrete generation for the final answer. AdaptSoft controls the token-mixture temperature using the current reasoning state and predictive uncertainty.

![Gradient-alignment learning](figures/figure3.png)

![Adaptive temperature example](figures/figure4.png)

## Results

The reported experiments use **Qwen3-VL-8B-Thinking** as the backbone and train with GRPO on OmniSpatial.

| Benchmark | Soft Spatial Reasoning | Hard Thinking + GRPO | Soft Thinking + GRPO |
|---|---:|---:|---:|
| OmniSpatial | **49.68** | 45.92 | 46.93 |
| SpatiaLab (zero-shot) | **48.71** | 46.21 | 46.86 |
| MindCube (zero-shot) | **38.13** | - | - |

On OmniSpatial, the method improves over basic soft thinking by 2.75 points. On the unseen SpatiaLab benchmark, it improves over the strongest prior open-weights comparison by 2.93 points and transfers to spatial mental modeling on MindCube.

Source code will be released upon acceptance.

## Citation

If you find this work useful, please cite:

```bibtex
@article{sultan2026softspatialreasoning,
	title={Soft Spatial Reasoning},
	author={Sultan, Rafi Ibn and Chowdhury, Md. Sajid Alam and Zare Zade, Saleh and Li, Chengyin and Khanduri, Prashant and Brocanelli, Marco and Zhu, Dongxiao},
	journal={arXiv preprint},
	year={2026}
}
```