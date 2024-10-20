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
`Figure2.py` models the dynamics of non-proliferative and proliferative cervical cancer cells influenced 
by cisplatin and paclitaxel using coupled semi-linear ODEs. Optimal adminstration is then compared with 
administration at a constant rate, and the ratio of final cell counts are calculated as a metric of efficacy.

#### Model Description
The system is described by the following differential equations:

1. **Non-Proliferative Cells ($$N_a$$):**
   $$\frac{dNa}{dt} = 2(1 - Up) \cdot Nb - Na \cdot (1 - Uc) - \alpha \cdot Uc \cdot Na$$


2. **Proliferative Cells ($$N_b$$):**
   $$\frac{dNb}{dt} = -(1 - Up) \cdot Nb + Na \cdot (1 - Uc) - \beta \cdot Up \cdot Nb$$

Where:
- $$\( N_a\)$$ = Non-proliferative cervical cancer cells (G1-phase)
- $$\( N_b \)$$ = Proliferative cervical cancer cells (S-/G2-phase)
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

### Figure5+6.py
`Figure5+6.py` models the dynamics of sympathoblasts, adrenergic cells, and mesenchymal cells influenced by 
retinoic acid, a chemotherapeutic agent, a track inhibitor, and nerve growth factor using coupled semi-linear
ODEs. The script evaluates the effectiveness of drug combinations by calculating the total cost of drug 
administration along with the respective individual drug costs.

#### Model Description
The system is described by the following differential equations:

1. **Sympathoblasts ($$n_I$$):**
   $$\frac{dn_I}{dt} = (\lambda - 2) n_I + n_N + n_S - (\delta \cdot u_{RA} + 2r \cdot u_{chemo} + \delta \cdot (1 - u_{trk}) \cdot u_{NGF}) n_I + \delta \cdot (1 - u_{trk}) n_N$$

2. **Adrenergic Cells ($$n_N$$):**
   $$\frac{dn_N}{dt} = -2n_N + n_I + n_S + \delta \cdot (u_{RA} + (1 - u_{trk}) \cdot u_{NGF}) n_I - (\delta_{apop} \cdot (1 - u_{NGF}) \cdot (1 - u_{trk}) + 2\delta \cdot (1 - u_{trk})) n_N$$

3. **Mesenchymal Cells ($$n_S$$):**
   $$\frac{dn_S}{dt} = -2n_S + n_I + n_N + \delta \cdot (1 - u_{trk}) n_N$$

Where:
- $$\( n_I \)$$ = Sympathoblasts
- $$\( n_N \)$$ = Adrenergic cells
- $$\( n_S \)$$ = Mesenchymal cells
- $$\( u_{RA} \)$$ = Relative concentration of retinoic acid
- $$\( u_{chemo} \)$$ = Relative concentration of chemotherapeutic agent
- $$\( u_{trk} \)$$ = Relative concentration of track inhibitor 
- $$\( u_{NGF} \)$$ = Relative concentration of nerve growth factor

#### State and Drug Vectors
- **State Vector:** $$\( n_I, n_N, n_S \)$$
- **Drug Vector:** $$\(u_{RA}, u_{chemo}, u_{trk}, u_{NGF} \)$$

#### How to Run
To execute the script, use the following command in your terminal:

```bash
python Figure5+6.py <lambda> <delta> <delta_apop>
Outputs
```
The script will generate an output file named `lmbd=<lambda>-delta=<delta>-deltaAPOP=<delta_apop>`.txt, which contains:

- Total dosage of retinoic acid administered
- Total dosage of chemotherapeutic agent administered
- Total dosage of track inhibitor administered
- Total dosage of nerve growth factor administered
- Total cost of retinoic acid administration
- Total cost of chemotherapeutic agent administration
- Total cost of track inhibitor administration
- Total cost of nerve growth factor administration
