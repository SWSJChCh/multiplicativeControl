# Optimal control in combination therapy for heterogeneous cell populations with drug synergies

## Overview
This repository contains Python scripts used to generate data for Figure 2,3,5 and 6 in the publication:
_Optimal control in combination therapy for heterogeneous cell populations with drug synergies_

### Code Authors
- Samuel Johnson
- Simon Martina-Perez

### Date
- 06/10/2024

### Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib

The required libraries can be installed using pip:

```bash
pip install numpy scipy matplotlib

## Script Descriptions

### Figure2.py
`Figure2.py` simulates and analyzes optimal combination therapy for heterogeneous cell populations using drug synergies.
It models the dynamics of non-proliferative and proliferative cells influenced by cisplatin and paclitaxel using coupled
semi-linear ODEs. Optimal adminstration is then compared with administration at a constant rate, and the ratio of final
cell counts are calculated in both regimes.

#### Model Description
The system is described by the following differential equations:

1. **Non-Proliferative Cells (Na):**
   \[
   \frac{dNa}{dt} = 2(1 - Up) \cdot Nb - Na \cdot (1 - Uc) - \alpha \cdot Uc \cdot Na
   \]

2. **Proliferative Cells (Nb):**
   \[
   \frac{dNb}{dt} = -(1 - Up) \cdot Nb + Na \cdot (1 - Uc) - \beta \cdot Up \cdot Nb
   \]

Where:
- \( Na \) = Non-proliferative cells (G1-phase)
- \( Nb \) = Proliferative cells (S-/G2-phase)
- \( Uc \) = Relative concentration of cisplatin
- \( Up \) = Relative concentration of paclitaxel

#### State and Drug Vectors
- **State Vector:** \([Na, Nb]\)
- **Drug Vector:** \([Uc, Up]\)


#### How to Run
To execute the script, use the following command in your terminal:

```bash
python Figure2.py <alpha> <beta>
