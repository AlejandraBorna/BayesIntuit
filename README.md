# BayesIntuit: A reliability-aware neural framework integrating dynamic memory and epistemic uncertainty for interpretable decision-making.

BayesIntuit is a reliability-aware neural framework that integrates dynamic memory, epistemic uncertainty, and interpretable reasoning into supervised learning models. The framework is inspired by human intuition: selectively blending current perception with prior experience, modulated by learned confidence.

This repository provides a **reference implementation** of the core BayesIntuit architecture as published in:

Bornacelly, M. (2025). _BayesIntuit: A Neural Framework for Intuition-Based Reasoning_.  
Proceedings of the International Conference on Industrial Engineering and Operations Management (IEOM). Springer.
---

## Overview

BayesIntuit introduces a modular reasoning layer that can be attached to attention-based neural architectures (e.g., LSTM, Transformer). The framework combines:

- **Dynamic Memory Bank**  
  Stores and retrieves latent representations from prior instances using learned semantic projections and clustering.

- **Epistemic Gating via Alpha (α)**  
  A learned, stochastic gating coefficient modeled with a Beta distribution that controls memory reuse based on semantic similarity and uncertainty.

- **Bayesian Output Layer**  
  Provides calibrated predictive uncertainty while preserving computational efficiency.

Together, these components enable **semantic regularization in representation space**, yielding improved robustness, interpretability, and reliability under distributional shift.
---

## What This Repository Contains

This repository focuses on **illustrative, self-contained components** of BayesIntuit, including:

- Core memory bank logic (storage, retrieval, semantic projection)
- Alpha estimation and Bayesian sampling mechanisms
- Integration with attention-based encoders (LSTM / Transformer)
- Minimal training and evaluation pipelines for demonstration

> Note: Full experimental pipelines, large-scale benchmarks, and extended variants are part of ongoing journal submissions and are therefore not included here.
---
## Scope and Intended Use

This code is intended for:
- Research inspection and methodological understanding
- Reproducibility of core architectural ideas
- Extension into reliability-aware and interpretable ML research

It is **not** designed as a production-ready package.
---

## Related Work and Extensions

BayesIntuit has been extended in ongoing work to study:
- Reliability manifolds under distribution shift
- Informative post-hoc diagnostics for interpretability
- Transformer-based variants for semantic reasoning

Representative code snippets for these extensions can be shared upon request.
---
## Citation

If you use or reference this work, please cite: 

Bornacelly, M. (2026). BayesIntuit: A Neural Framework for Intuition-Based Reasoning. In: Florez, H., Rabelo, L., Diaz, C. (eds) Industrial Engineering and Operations Management. IEOM-CS 2025. Communications in Computer and Information Science, vol 2557. Springer, Cham. https://doi.org/10.1007/978-3-031-98235-4_9
---
## Contact
For questions related to this repository or research collaborations, please contact:

**Mayra Bornacelly**  
PhD Student, Industrial Engineering  
University of Central Florida

