
## Action Surface Fitting Code
This repository contains the Python implementation for the "action surface" analysis described in the manuscript "WAChRs are excitatory opsins sensitive to indoor lighting." The code facilitates the fitting of optogenetic response data to a function of both light intensity and wavelength.

## The Action Surface Model
Our model characterizes an opsin's activation profile across a two-dimensional input space of irradiance (intensity) and wavelength (spectrum). It achieves this by modulating the dose-response relationship with a spectral sensitivity function.

The model combines a sigmoidal Hill function with a Gaussian curve that represents the opsin's action spectrum. The response R for a given irradiance I and wavelength λ is defined by the following equation:

$$R(I, \lambda) = A_{max} \frac{I_{eff}(I, \lambda)^n}{I_{eff}(I, \lambda)^n + K^n} + R_{baseline} $$The key component of this model is the **effective irradiance** ($I_{eff}$), which scales the incident irradiance based on its proximity to the opsin's peak sensitivity: $$I\_{eff}(I, \lambda) = I \cdot \exp\left(-\frac{(\lambda - \lambda\_{max})^2}{2\sigma^2}\right) $$

### Model Parameters 

- **$A_{max}$**: The maximum response amplitude above baseline. 
- **$K$**: The half-maximal effective irradiance. This constant represents the value of $I_{eff}$ that elicits a half-maximal response. 
- **$n$**: The Hill coefficient, describing the steepness or cooperativity of the response. 
- **$\lambda_{max}$**: The wavelength of peak sensitivity. 
- **$\sigma$**: The standard deviation of the Gaussian spectral sensitivity curve, which defines the spectral bandwidth of the opsin.
- **$R_{baseline}$**: The baseline physiological response in the absence of light stimulation.

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