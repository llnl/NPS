# Neural Phase Simulation (NPS)
NPS is a package of codes for training and applying neural network models for materials science, especially for machine-learning based materials simulation. Its scope includes microstructure evolution, accelerated atomistic dynamics, generative AI models for atomic structures, and structure denoising, phase classification and order parameters.

# Features
## Capabilities
* 2D and 3D microstructure evolution, including grain growth, nucleation and growth, spinodal decomposition, and dendrite growth, including both deterministic and stochastic dynamics
* Data-driven coarse-graining via potential of mean force (PMF)
* Accelerated full-atom and coarse-grained molecular dynamics
* Dislocation dynamics and learned mobility laws
* Generative modeling of crystalline phases, disordered structures, and grain boundaries
* Crystal structure denoising, phase classification, and order-parameter analysis

## Training ground-truth or high-fidelity method
The NPS surrogate models can be trained from various ground truth simulation methods, which are supposed to be accurate but expensive, such as **molecular dynamics**, phase field methods, Kinetic Monte Carlo and discrete dislocation dynamics.

## Networks:
* Convolutional neural networks, convolutional RNN
* Graph neural networks and equivariant GNN
* Transformer
* Diffusion model

# Installation
NPS requires:
* Python >= 3.6
* PyTorch >= 1.9
* Torch Geometric, e3nn
* Numpy, Scipy, Matplotlib

# Getting started
The main entry is [NPS/main.py](NPS/main.py).
## Training
```
python -m NPS/main.py --mode=train ...
```

## Prediction/Simulation
```
python -m NPS/main.py --mode=predict ...
```

See the [tutorial](examples/tutorial.md)

# Publications that use this repository
1. K. Ji, L. Sun, S. Liu, C. A. Orme, G. Bucci, M. A. Worsley, F. Zhou, and T. W. Heo, "Scalable Autoregressive Deep Surrogates for Complex Microstructure Dynamics", [arXiv:2511.03884](http://arxiv.org/abs/2511.03884), Machine Learning: Science and Technology, accepted (2026). <span style="color: green; font-size: 200%;">[**>>CODE**](notebooks/dendrite/)</span>
1. B. Lei, E. Chen, H. Kwon, T. Hsu, B. Sadigh, V. Lordi, T. Frolov, and F. Zhou, "Grand canonical diffusion model for crystalline phases and grain boundaries", [arXiv:2408.15601](http://arxiv.org/abs/2408.15601).
1. L. Sun, V. H. Nguyen, S. Liu, J. Klepeis, and F. Zhou, "Learning Noisy Dynamics of Spinodal Decomposition from Stochastic Differential Equations", [arXiv:2604.09664](https://arxiv.org/abs/2604.09664).
1. H. Kwon, B. Sadigh, S. Hamel, V. Lordi, J. Klepeis, and F. Zhou, "A probabilistic framework for crystal structure denoising, phase classification, and order parameters", [arXiv:2512.11077](https://arxiv.org/abs/2512.11077).
1. K. Ji, L. Sun, S. Liu, F. Zhou, and T. W. Heo, "Scalable Autoregressive Deep Surrogates for Dendritic Microstructure Dynamics", [arXiv:2511.03884](http://arxiv.org/abs/2511.03884), submitted to Patterns (2025).
1. Z. Tian, E. Suwandi, T. Oppelstrup, V. V. Bulatov, J. B. Harley, and F. Zhou, "Scaling Kinetic Monte-Carlo Simulations of Grain Growth with Combined Convolutional and Graph Neural Networks", [Acta Materialia 311, 122153 (2026)](https://doi.org/10.1016/j.actamat.2026.122153). <span style="color: green; font-size: 200%;">[**>>CODE**](notebooks/AE_GNN/tutorial.ipynb)</span>
1. H. Kwon, T. Hsu, W. Sun, W. Jeong, F. Aydin, J. Chapman, X. Chen, M. R. Carbone, D. Lu, F. Zhou, and T. A. Pham, "Spectroscopy-Guided Discovery of Three-Dimensional Structures of Disordered Materials with Diffusion Models", [Machine Learning: Science and Technology 5, 045037 (2024)](https://iopscience.iop.org/article/10.1088/2632-2153/ad8c10).
1. N. Bertin, V. V. Bulatov, and F. Zhou, "Learning Dislocation Dynamics Mobility Laws from Large-Scale MD Simulations", [npj Computational Materials 10, 192 (2024)](https://doi.org/10.1038/s41524-024-01378-4).
1. H. Sun, S. Hamel, T. Hsu, B. Sadigh, V. Lordi, and F. Zhou, "Ice phase classification made easy with score-based denoising", [Journal of Chemical Information and Modeling 64, 6369 (2024)](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00822).
1. T. Hsu, B. Sadigh, N. Bertin, C.-W. Park, J. Chapman, V. Bulatov, and F. Zhou, "Score-based denoising for atomic structure identification", [npj Computational Materials 10, 155 (2024)](https://doi.org/10.1038/s41524-024-01337-z).
1. T. Hsu, B. Sadigh, V. Bulatov, and F. Zhou, "Score Dynamics: Scaling Molecular Dynamics with Picoseconds Timestep via Conditional Diffusion Model", [Journal of Chemical Theory and Computation 20, 2335 (2024)](https://doi.org/10.1021/acs.jctc.3c01361).
1. S. Fan, A. L. Hitt, M. Tang, B. Sadigh, and F. Zhou, "Accelerate Microstructure Evolution Simulation Using Graph Neural Networks with Adaptive Spatiotemporal Resolution", [Machine Learning: Science and Technology 5, 025027 (2024)](https://doi.org/10.1088/2632-2153/ad3e4b).
1. N. Bertin and F. Zhou, "Accelerating discrete dislocation dynamics simulations with graph neural networks", [Journal of Computational Physics 487, 112180 (2023)](https://doi.org/10.1016/j.jcp.2023.112180).
1. K. Yang, Y. Cao, Y. Zhang, M. Tang, B. Sadigh, D. Aberg, and F. Zhou, "Self-supervised learning and prediction of microstructure evolution with convolutional recurrent neural networks", [Patterns 2, 100243 (2021)](https://doi.org/10.1016/j.patter.2021.100243).

# Authors
NPS is being developed by [Fei Zhou](mailto:zhou6@llnl.gov)


# Getting Involved
Please contact [Fei Zhou](mailto:zhou6@llnl.gov) for questions.

# Contributing
The NPS package is intended to be a general and extensible framework to develop machine-learning surrogate models for materials simulation. Contributions are welcome.
Just send us a pull request. When you send your request, make `develop` the destination branch on the repository.

Users who want the latest package versions, features, etc. can use `develop`.


# License
NPS is distributed under the terms of the MIT license.

All new contributions must be made under the MIT license.

SPDX-License-Identifier: MIT

LLNL-CODE-842508
