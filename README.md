# MS2 Public Goods Inference

Code and data for inferring protein-specific public goods (complementation) effects in MS2 bacteriophage using experimental evolution and simulation-based inference.

---

## Abstract
Interactions among individuals in structured populations can alter fitness effects of mutations and reshape evolutionary processes. In many systems, including bacteria, yeast, and viruses, such interactions often result in public goods: gene products that are costly to produce yet exploitable by others. During viral coinfection of the same cell, gene products from one genome may complement deleterious mutations in another, allowing defective genomes to persist. Yet it remains difficult to infer which proteins are shareable from population sequencing data, because mutation, selection, drift, and complementation are intertwined. Here, we developed a quantitative framework to infer protein-specific public goods in the RNA bacteriophage MS2, which encodes only four proteins. We analyzed experimental evolution data generated under two multiplicity-of-infection (MOI) regimes: low MOI, where coinfection is rare, and high MOI, where coinfection is common. We first compared empirical mutation patterns between regimes and then applied a Wright-Fisher model combined with simulation-based Bayesian inference using neural posterior estimation. In a two-stage strategy, gene-specific fitness effects were inferred from low-MOI data and subsequently used to estimate protein sharing under high-MOI conditions. Across two statistical inference frameworks, lysis emerged as the strongest public-good candidate, replicase and coat showed an intermediate signal, and maturation showed the weakest evidence for sharing. Together, our results show that viral proteins differ markedly in their propensity to act as public goods. More broadly, they illustrate how coinfection can generate density-dependent selection, a general feature of social evolution that may shape evolutionary dynamics.


## Repository structure

* `data/` - processed sequencing data and summary statistics
* `data_analysis` - preprocessing and data analysis
* `model/` – evolutionary model simulation, parameter definition, training and evaluation code
  * `parameters_model_AB.py` – model parameter definitions and configuration
  * `evolutionary_model_AB.py` – core evolutionary model
  * `simulator_model_AB.py` – simulation framework
  * `sbi_simulate_AB.py` – generation of simulated datasets for inference
  * `train_AB.py` – training of the NPE model
  * `test_empirical_AB.py` – inference on empirical data
  * `test_nn_performance_AB.py` – neural network performance and diagnostics
* `visualizations` - notebook for visalizations and plotting

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

---

## Evolutionary model

We use a stochastic Wright–Fisher model to simulate viral population dynamics across serial passages.

Key processes:

* Mutation (genome-wide rate μ)
* Selection (gene-specific fitness effects)
* Drift (finite population sampling)
* Complementation (protein-specific "recessiveness" parameters)

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



---

## Acknowledgments

Developed in the Stern Lab.
