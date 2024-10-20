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
```
## Script Descriptions

### Figure2.py
`Figure2.py` simulates and analyzes optimal combination therapy for heterogeneous cell populations using drug synergies.
It models the dynamics of non-proliferative and proliferative cervical cancer cells influenced by cisplatin and paclitaxel 
using coupled semi-linear ODEs. Optimal adminstration is then compared with administration at a constant rate, and the 
ratio of final cell counts are calculated as a metric of efficacy.

#### Model Description
The system is described by the following differential equations:

1. **Non-Proliferative Cells ($$N_a$$):**
   $$\frac{dNa}{dt} = 2(1 - Up) \cdot Nb - Na \cdot (1 - Uc) - \alpha \cdot Uc \cdot Na$$


2. **Proliferative Cells ($$N_b$$):**
   $$\frac{dNb}{dt} = -(1 - Up) \cdot Nb + Na \cdot (1 - Uc) - \beta \cdot Up \cdot Nb$$

Where:
- $$\( N_a\)$$ = Non-proliferative cells (G1-phase)
- $$\( N_b \)$$ = Proliferative cells (S-/G2-phase)
- $$\( U_c \)$$ = Relative concentration of cisplatin
- $$\( U_p \)$$ = Relative concentration of paclitaxel

#### State and Drug Vectors
- **State Vector:** $$\(N_a, N_b\)$$
- **Drug Vector:** $$\(U_c, U_p\)$$


#### How to Run
To execute the script, use the following command in your terminal:

```bash
python Figure2.py <alpha> <beta>
```
#### Outputs
The script will generate an output file named `a=<alpha>-b=<beta>.txt`, which contains:

- Total cell count at the end of the optimal regime
- Total cell count at the end of the mean administration regime
- Ratio of final cell counts
- Total cost of drug administration
- Individual drug costs for cisplatin and paclitaxel

### Figure3.py
`Figure3.py` simulates the same system, but also allows entries of the state cost matrix along the leading
diagonal to be varied independently to study how optimality varies with the cost of drugs relative to one-
another. 

#### How to Run
To execute the script, use the following command in your terminal:

```bash
python example1-Fig2.py <alpha> <beta> <control_cost>
```
#### Outputs
The script will generate an output file named `a=<alpha>-b=<beta>-R11=<control_cost>.txt`, which contains:

- Total cell count at the end of the optimal regime
- Total cell count at the end of the mean administration regime
- Ratio of final cell counts
- Total cost of drug administration
- Individual drug costs for cisplatin and paclitaxel
