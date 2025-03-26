# ReLapse

A reinforcement learning framework for medical relapse prediction and intervention modeling.

## Overview

ReLapse uses reinforcement learning to model the decision process of when to take measurements and when to intervene in a medical context. The agent learns to balance:

- When to take measurements (with associated costs)
- How long to wait between measurements
- When to intervene based on predicted relapse patterns

The environment simulates patient biomarkers that follow exponential growth patterns, where measurements can be noisy and the goal is to intervene just before a relapse occurs.

## Installation

Install the package locally:

```bash
# Using pip
pip install -e .
```

## Usage

### Training
Train the model with:

```bash
relapse --train
```

### Validation
Validate the model with:

```bash
relapse --validate --weights <path_to_weights>
``` 