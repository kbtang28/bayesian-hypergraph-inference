# Bayesian inference of hypergraph structure from dynamics

This repository accompanies the preprint "Bayesian inference of hypergraph structure from scarce and noisy dynamical observations" by Katerina Tang, Vivek Srikrishnan, and Jackson Kulik.

## Dependencies

This code is based on Julia 1.12.5. Relevant dependencies are in the `Project.toml` and `Manifest.toml` files. (The `Manifest.toml` file specifies the particular versions; this file should be kept as-is for perfect reproducibility but may need to be deleted and rebuilt with `Pkg.instantiate()` for different Julia versions.)

Bayes-THIS, our inference method, is based on sparse Bayesian regression with automatic relevance determination (ARD); we use [`SparseBayes.jl`](https://github.com/kbtang28/SparseBayes.jl), our Julia port of the [SparseBayes v2.0 MATLAB library]((https://www.miketipping.com/sparsebayes.htm)) that implements sparse Bayesian regression with ARD via the fast marginal likelihood maximization algorithm of Tipping and Faul (2003).

## Reproduction

After cloning this repository, install the necessary packages:
```julia
import Pkg
Pkg.activate(".") # from cloned root directory
Pkg.instantiate()
```

### Repository structure:
The structure of this repository is as follows:
* `figs` - PNG files for each of the figures in the paper.
* `hyperg-models` - adjacency matrices, tensors, and edge lists for toy hypergraph models used in some experiments
* `scripts` - scripts to run the experiments described in the paper and generate corresponding figures.
* `src` - utility library (hypergraph generation, inference methods, evaluation metrics) loaded as the `BayesTHIS` module

Below is a more detailed mapping between results in the paper and experiment/plotting scripts.

| Figure | Scripts |
| ------ | ------- |
| Fig. 1 | `test-robustness.jl`<br>`plot-robustness-comparison.jl` |
| Fig. 2 | `test-robustness-near-sync.jl`<br>`plot-robustness-near-sync.jl` |
| Fig. 3 | `demo-decision-space.jl` |
| Fig. 4 | `test-filtering.jl`<br>`plot-filtering.jl` |
| Fig. 5 | `test-ppc-scatter.jl`<br>`test-ppc-trajectories-fd.jl`<br>`plot-ppcs.jl` |
| Fig. 6 | `test-node-swap.jl`<br>`plot-structure-effects.jl` |
| Fig. 7 | `test-robustness-num-nodes.jl`<br>`plot-performance-num-nodes.jl` |
| Fig. 8 | `test-robustness-sparsity.jl`<br>`plot-performance-sparsity.jl` |