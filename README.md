# Home Energy Management under Tiered Peak Power Charges

Code and data to replicate all results in the
[paper](https://web.stanford.edu/~boyd/papers/hem.html).

## Setup

Requires [Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/)
and [MOSEK](https://www.mosek.com/products/academic-licenses/) licenses
(free for academic use).

```bash
uv sync
```

## Usage

```bash
uv run python run.py                    # Run everything
uv run python run.py --train            # Train forecasters only
uv run python run.py --experiments      # Run experiments only
uv run python run.py --figures          # Generate figures only
```

Individual experiments:
```bash
uv run python run.py --baseline         # Baseline policies
uv run python run.py --prescient        # Prescient optimization
uv run python run.py --mpc              # MPC policies
uv run python run.py --capacity         # Capacity sweep
uv run python run.py --sensitivity      # Sensitivity analysis
```

Results are saved to `results/`, figures to `figures/`.

## Citing

```bibtex
@misc{perezpineiro2026home,
    title         = {Home Energy Management under Tiered Peak Power Charges},
    author        = {David P\'erez-Pi\~neiro and Sigurd Skogestad and Stephen Boyd},
    year          = {2026},
    eprint        = {2307.07580},
    archivePrefix = {arXiv},
    primaryClass  = {math.OC}
}
```
