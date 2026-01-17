# How Is Uncertainty Propagated in Knowledge Distillation?

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the official implementation for the paper **_How Is Uncertainty Propagated in Knowledge Distillation?_**

The goal of this repo is to systematically study how different sources of uncertainty propagate through the knowledge distillation pipeline and to provide practical, variance-aware distillation methods that better preserve teacher uncertainty. For more details please consult the paper.



## Installation
   ```bash
   pip install -r requirements.txt
   ```

## Datasets

### BioASQ (LLM experiments)

The LLM experiments use the **BioASQ** biomedical question answering dataset.

- **Dataset access:** https://participants-area.bioasq.org/datasets/

### Other datasets

All other datasets used in the paper (e.g., Boston Housing, MNIST, Wine, Breast Cancer, Covertype) are loaded automatically via the provided scripts.


## Repository Structure

The repository is organized by model class and source of uncertainty, mirroring the structure of the paper.

```
root
├─ simple_model_experiments
│  ├─ teacher_output_uncertainty
│  ├─ student_initialization_uncertainty
│  │  └─ model_driven
│  │  └─ data_driven
│  └─ student_output_uncertainty
│  └─ variance_aware_distillation
├─ llm_experiments
│  ├─ teacher_output_uncertainty
│  ├─ student_initialization_uncertainty
│  └─ student_output_uncertainty
│  └─ variance_aware_distillation
```

Each directory corresponds directly to sections and results in the paper:

- **Teacher output uncertainty**
  - Linear regression, neural networks, and LLM experiments
  - Corresponds to **Section 4**

- **Student initialization uncertainty**
  - Model-driven perturbations and data-driven bootstrap analyses
  - Corresponds to **Section 5**

- **Student output uncertainty**
  - Predictive entropy, intra-student, and inter-student analyses
  - Corresponds to **Section 6**

- **Variance-aware distillation**
  - Averaging and variance-weighting approaches
  - Corresponds to **Section 7**

Each experiment directory contains scripts to reproduce the corresponding results.
