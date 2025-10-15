# Action Surface Fitting Code

This repository contains the implementation of the input-modulated dose-response function fitting (referred to as "action surfaces" in the manuscript) used in our optogenetics analysis.

## Overview

The action surface model describes how optogenetic responses vary as a function of both light intensity (irradiance) and wavelength. The model uses a Hill function modulated by a Gaussian spectral sensitivity curve:

```
Response = A_max * (I_scaled^n) / (I_scaled^n + K^n) + baseline

where:
  I_scaled = I * exp(-(λ - λ_max)² / (2σ²))
```

Parameters:
- `A_max`: Maximum response amplitude
- `K`: Half-saturation constant
- `n`: Hill coefficient
- `λ_max`: Peak wavelength sensitivity
- `σ`: Spectral width
- `baseline`: Baseline response

## Installation



```bash
# Clone the repository
git clone <repository-url>
cd wachr_public_code

# Create virtual environment and install dependencies
uv venv
uv sync

source .venv/bin/activate
```
