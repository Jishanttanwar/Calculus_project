# Leslie Matrix Analysis for Population Dynamics

This project implements Leslie matrices for age-structured population modeling using matrix multiplication and eigenvalue analysis.

## Overview

Leslie matrices model population dynamics by incorporating fertility and survival rates across age classes. Matrix multiplication projects populations forward in time, while eigenvalue analysis determines long-term growth rates and stable age distributions.

## Features

- **Matrix Construction**: Build Leslie matrices from fertility and survival rates
- **Population Projection**: Project populations using matrix multiplication (n(t+1) = L × n(t))
- **Eigenvalue Analysis**: Calculate growth rates and stable age distributions
- **Sensitivity Analysis**: Determine parameter importance for population growth
- **Visualization**: Plot population dynamics, age distributions, and growth patterns
- **Built-in Examples**: Human, mammal, bird, and fish population models

## Requirements

```bash
pip install numpy matplotlib scipy pandas
```

## Quick Start

```bash
# Run the demonstration
python leslie_matrix.py
```

## Usage

```python
from leslie_matrix import LeslieMatrix

# Create Leslie matrix
leslie = LeslieMatrix(
    fertility_rates=[0.0, 1.5, 2.0],  
    survival_rates=[0.8, 0.6]        
)

# Analyze population
leslie.summary_statistics()
populations = leslie.project_population([1000, 500, 300], 20)
leslie.plot_projection([1000, 500, 300], 20)

# Eigenvalue analysis
analysis = leslie.eigenvalue_analysis()
print(f"Growth rate: {analysis['population_growth_rate']:.4f}")

# Built-in examples
from leslie_matrix import create_example_matrices
examples = create_example_matrices()
examples['human'].summary_statistics()
```

## Mathematical Background

### Leslie Matrix Structure
```
L = [f₀  f₁  f₂  ...  fₙ₋₁]    where fᵢ = fertility rate
    [s₀   0   0  ...   0 ]          sᵢ = survival rate
    [ 0  s₁   0  ...   0 ]
    [ ⋮   ⋮   ⋮  ⋱    ⋮ ]
```

### Key Concepts
- **Population Projection**: n(t+1) = L × n(t)
- **Growth Rate**: Dominant eigenvalue λ₁ determines if population grows (λ₁>1), declines (λ₁<1), or stays stable (λ₁=1)
- **Stable Age Distribution**: Eigenvector shows eventual age structure
- **Sensitivity**: ∂λ₁/∂parameter shows which rates most affect population growth

## Built-in Examples
- **Human**: Low fertility, high survival
- **Small Mammal**: Moderate rates
- **Bird**: Variable fertility by age
- **Fish**: High late fertility, low early survival

## Sample Output

```
EIGENVALUE ANALYSIS:
Dominant eigenvalue (λ₁): 0.9695
Population growth rate: -3.05% per time step
Population is DECLINING

STABLE AGE DISTRIBUTION:
Age class 0: 45.57%
Age class 1: 36.46%
Age class 2: 17.98%
```

## Applications
- Population viability analysis
- Conservation planning  
- Life history comparisons
- Parameter sensitivity analysis

This implementation provides educational and research tools for understanding matrix-based population models in biology.
