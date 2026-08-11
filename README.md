# Stabilizing Physics-Informed Consistency Models via Structure-Preserving Training

## Abstract

We propose a physics-informed consistency modeling framework for solving partial differential equations (PDEs) via fast, few-step generative inference. We identify a key stability challenge in physics-constrained consistency training, where PDE residuals can drive the model toward trivial or degenerate solutions, degrading the learned data distribution. To address this, we introduce a structure-preserving two-stage training strategy that decouples distribution learning from physics enforcement by freezing the coefficient decoder during physics-informed fine-tuning. We further propose a two-step residual objective that enforces physical consistency on refined, structurally valid generative trajectories rather than noisy single-step predictions. The resulting framework enables stable, high-fidelity inference for both unconditional generation and forward problems. We demonstrate that forward solutions can be obtained via a projection-based zero-shot inpainting procedure, achieving consistent accuracy of diffusion baselines with orders of magnitude reduction in computational cost. 


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@inproceedings{10.1145/3770855.3819059,
author = {Chang, Che-Chia and Dai, Chen-Yang and Lin, Te-Sheng and Lai, Ming-Chih and Lai, Chieh-Hsin},
title = {Stabilizing Physics-Informed Consistency Models via Structure-Preserving Training},
year = {2026},
isbn = {9798400722592},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3770855.3819059},
doi = {10.1145/3770855.3819059},
abstract = {We propose a physics-informed consistency modeling framework for solving partial differential equations (PDEs) via fast, few-step generative inference. We identify a key stability challenge in physics-constrained consistency training, where PDE residuals can drive the model toward trivial or degenerate solutions, degrading the learned data distribution. To address this, we introduce a structure-preserving two-stage training strategy that decouples distribution learning from physics enforcement by freezing the coefficient decoder during physics-informed fine-tuning. We further propose a two-step residual objective that enforces physical consistency on refined, structurally valid generative trajectories rather than noisy single-step predictions. The resulting framework enables stable, high-fidelity inference for both unconditional generation and forward problems. We demonstrate that forward solutions can be obtained via a projection-based zero-shot inpainting procedure, achieving consistent accuracy of diffusion baselines with orders of magnitude reduction in computational cost.},
booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {10555–10566},
numpages = {12},
keywords = {physics-informed, scientific machine learning, partial differential equations, pde solving, consistency model, generative modeling},
location = {Republic of Korea},
series = {KDD '26}
}
```
