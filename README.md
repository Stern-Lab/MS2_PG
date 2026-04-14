# MS2 Public Goods Inference

Code and data for inferring protein-specific public goods (complementation) effects in MS2 bacteriophage using experimental evolution and simulation-based inference.

---

## Overview

RNA viruses frequently experience co-infection, allowing gene products to be shared between genomes. This repository contains the code used to infer which MS2 bacteriophage proteins behave as public goods under co-infection.

We combine:

* Experimental evolution under low and high multiplicity of infection (MOI)
* A stochastic evolutionary model
* Simulation-based inference using Neural Posterior Estimation (NPE)

to quantify protein-specific complementation effects.

---

## Repository structure

* `data/` - processed sequencing data and summary statistics TODO!!
* `analysis/` – notebooks for preprocessing, empirical data analysis, and plotting
* `model/` – simulation, parameter definition, training, and evaluation code
  * `parameters_model_AB.py` – model parameter definitions and configuration
  * `evolutionary_model_AB.py` – core evolutionary model
  * `simulator_model_AB.py` – simulation framework
  * `sbi_simulate_AB.py` – generation of simulated datasets for inference
  * `train_AB.py` – training of the NPE model
  * `test_empirical_AB.py` – inference on empirical data
  * `test_nn_performance_AB.py` – validation and diagnostics

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Stern-Lab/MS2_PG.git
cd MS2_PG
```

Create environment:

```bash
conda env create -f environment.yml
conda activate ms2_pg
```

---

## Data

This repository includes processed data used for analysis.

Raw sequencing data is available at: [ADD LINK] TODO!!

---

## Evolutionary model

We use a stochastic Wright–Fisher model to simulate viral population dynamics across serial passages.

Key processes:

* Mutation (genome-wide rate μ)
* Selection (gene-specific fitness effects)
* Drift (finite population sampling)
* Complementation (protein-specific “recessiveness” parameters)

Genotypes are represented in a reduced form capturing mutation counts per gene and mutation type.

See `model/` for implementation details.

---

## Inference

Parameters are inferred using Neural Posterior Estimation (NPE), a simulation-based inference method.

Workflow:

1. Simulate data under candidate parameters
2. Compare simulated and empirical summary statistics
3. Train neural density estimator
4. Infer posterior distributions of parameters

See `model/` for implementation details.

---

## Reproducing results

To reproduce the main results:

### 1. Generate simulations
TODO!!
```bash
python ?
```

### 2. Train inference model

```bash
python ?
```

---

## Model parameters

| Parameter | Meaning                        |
| --------- | ------------------------------ |
| μ         | mutation rate                  |
| ω_ns^(i)  | nonsynonymous fitness per gene |
| p_rec^(i) | probability of complementation |

---

## Citation

If you use this code, please cite:

[Your paper / preprint]

---

## Acknowledgments

Developed in the Stern Lab.
