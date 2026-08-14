# Tutorial notebooks

Lightweight demo for the autoregressive deep surrogate (ADS)
framework introduced in

> Ji et al., *Scalable Autoregressive Deep Surrogates for Dendritic
> Microstructure Dynamics*, arXiv:2511.03884 (2025).
> <https://arxiv.org/abs/2511.03884>

## Prerequisites

- Python 3.9+, NumPy, PyTorch (GPU preferred; CPU is fine, but could be very slow).
- A clone of the NPS package (<https://github.com/llnl/NPS>),
  either installed via `pip install -e .` from a clone or referenced by
  setting `NPS_PATH` at the top of the notebook.

Data is not included with the notebooks. Cells that need data first check for it
(`OK`/`MISS`) and skip themselves if it is missing.

## Running

```bash
jupyter lab isothermal_dendrite.ipynb
# or
jupyter lab directional_solidification.ipynb
```

## Licence

MIT.
