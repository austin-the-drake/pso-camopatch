# pso-camopatch

Particle Swarm Optimization for camouflaged adversarial patch generation. An extension of Williams and Li's CamoPatch (2023) that evaluates PSO as an alternative to the original Evolutionary Strategy + Simulated Annealing search method. Completed for an MS-level course taught by one of the original authors and reviewed favorably.

See [RESULTS.md](RESULTS.md) for quantitative performance metrics.

## Overview

Adversarial patches, which are small, semi-transparent perturbations applied to images, can fool image classifiers while remaining visually inconspicuous to humans. CamoPatch (Williams & Li, 2023) uses a computational art framework to achieve this; instead of optimizing millions of pixel parameters directly, patches are composed of overlapping semi-transparent circles. This reduces the search space to only a few hundred parameters (position, radius, color, alpha) while making the results more interpretable.

This project substitutes PSO for CamoPatch's original search method (Evolutionary Strategy with Simulated Annealing). This extension is motivated by PSO's population-based approach that might explore the joint optimization space of patch content and placement more effectively. The implementation keeps CamoPatch's computational art approach, while swapping the optimization algorithm along with the addition of a palette-based color gravity mechanism and a soft pbest threshold.

In practice, PSO finds patches with substantially better visual camouflage (10x lower L2 distance from background) than Random Search when successful. It also exhibits more consistent patch placement across runs than the original CamoPatch, suggesting more stable convergence to optimal positions.

Reference: [CamoPatch](https://github.com/phoenixwilliams/CamoPatch)

## Design Choices

**Soft Misclassification Threshold**: The fitness function uses a configurable margin threshold to allow particles to update their personal best even when a patch hasn't yet achieved misclassification. If the original class's logit is close to the runner-up, the patch is accepted as pbest, allowing the swarm to explore the boundary region more freely without requiring every update to be a confirmed attack. This prevents the search from getting stuck waiting for hard misclassification and keeps gbest uncorrupted; only verified misclassifications are reported as final results. This can be disabled for use in black-box classifier scenarios.

**Color Palette Gravity**: The implementation extends PSO's velocity update with an optional gravity term that pulls patch colors toward dominant colors in the target image. Extracted with K-means clustering, this approach helps the swarm escape local minima in the color space by biasing the search toward naturally camouflaged solutions. The mechanism trades some exploration freedom for better visual similarity to the background.

## Results

When successful, PSO generates adversarial patches with substantially higher visual camouflage (10x lower L2 distance) compared to Random Search, indicating superior patch quality. Full benchmark results and statistical analysis are in [RESULTS.md](RESULTS.md).

To reproduce: `python experiments/run_comparison.py --runs 5 --gens 100 --pop 100`

## Quick Start

Prerequisites: Python 3.8+, PyTorch, torchvision. Install with:

```bash
pip install -r requirements.txt
```

Run a single experiment:

```python
from src import PSOConfig, run

cfg = PSOConfig(image_source='https://example.com/image.jpg', pop_size=50, generations=50)
success, best_l2, _, _ = run(cfg, plot=True)
print(f"Attack {'succeeded' if success else 'failed'} (L2: {best_l2:.2f})")
```

Or run the interactive demo:

```bash
jupyter notebook demo.ipynb
```

For multi-image benchmarking against baselines:

```bash
python experiments/run_comparison.py --runs 5 --gens 100 --pop 100
```

Results and figures are saved to `results/`.

## Reproducibility

All experiments are deterministic given a fixed random seed. No external data is required; test images are fetched from a public ImageNet sample repository. Full configuration is exposed in `src/config.py`, and all results are logged to CSV for analysis.

To reproduce the main results, run `experiments/run_comparison.py` with default arguments (5 runs, 50 generations, 50 population size). With a GPU, the full benchmark completes in approximately 45 minutes.

## Future Work

There are a few natural extensions worth exploring:

**Linearly Transformed Gaussians**: The circle primitive is effective but limited. Switching to linearly transformed Gaussians would allow for ellipses, rotations, and smoother camouflage while staying in a low-dimensional parameter space. This could allow for more complex patch geometry without exploding the search space.

**Hybrid Search**: PSO converges to high-quality patches but has lower overall success rates than Random Search due to more subtle visual differences. Combining PSO's population-based exploration with local refinement might push success rates up while keeping patch quality high.

**Semantically-Guided Placement**: PSO's patch placement is fairly consistent across runs. This might indicate the swarm is finding meaningfully vulnerable locations to attack. Experiments with attention mechanisms could test whether we can guide PSO toward more effective attack locations.

**Cross-Model Generalization**: Do PSO patches transfer across classification model architectures differently than the original ES+SA patches? Different optimization behavior might produce patches that generalize to varying degrees.