# Figure2.py

## Overview
`Figure2.py` is a Python script designed to simulate and analyze optimal combination therapy for heterogeneous cell populations in the context of drug synergies. The model is based on differential equations representing the dynamics of non-proliferative and proliferative cells under the influence of two drugs, cisplatin and paclitaxel.

### Authors
- Samuel Johnson
- Simon Martina-Perez

### Date
- 06/10/2024

## Purpose
This script is used in the publication titled **"Optimal combination therapy for heterogeneous cell populations with drug synergies."** It implements a mathematical model to optimize drug administration for achieving desired therapeutic outcomes.

## Model Description
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

### State and Drug Vectors
- **State Vector:** \([Na, Nb]\)
- **Drug Vector:** \([Uc, Up]\)

## Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib

You can install the required libraries using pip:

```bash
pip install numpy scipy matplotlib
Usage
To run the script, execute the following command in your terminal:

bash
Copy code
python Figure2.py <alpha> <beta>
Replace <alpha> and <beta> with the desired parameters for the control cost matrix.

Output
The script will generate an output file named a=<alpha>-b=<beta>.txt, which contains:

Total cell count at the end of the optimal regime
Total cell count at the end of the mean administration regime
The ratio of final cell counts
Total cost of drug administration
Individual drug costs for cisplatin and paclitaxel
Code Structure
The code consists of several main functions:

Dynamics Functions: Define the linear and non-linear dynamics of the system.
Control Functions: Calculate the optimal control strategy based on the state and costate dynamics.
Boundary Conditions: Evaluate the boundary conditions for the boundary value problem (BVP).
Solvers: Functions to solve the optimal control problem using solve_bvp from SciPy.
Example Output
Upon running the script, the output file will contain space-separated values corresponding to the metrics defined above. Here’s an example of how the data might look:

Copy code
50.0 48.0 1.04 0.5 0.3 0.2
This output represents:

Total cell count in mean administration regime
Total cell count in optimal regime
Ratio of final cell counts
Total cost
Total cost for cisplatin
Total cost for paclitaxel
Conclusion
This script provides a framework for analyzing drug synergies in heterogeneous cell populations through optimal control theory. It can be modified or extended to accommodate different models or parameters based on research needs.

Copy code


You said:
I want to copy and paste one chunk of text, so don't break things up. 
ChatGPT said:
ChatGPT
markdown
Copy code
# Figure2.py

## Overview
`Figure2.py` is a Python script designed to simulate and analyze optimal combination therapy for heterogeneous cell populations in the context of drug synergies. The model is based on differential equations representing the dynamics of non-proliferative and proliferative cells under the influence of two drugs, cisplatin and paclitaxel.

### Authors
- Samuel Johnson
- Simon Martina-Perez

### Date
- 06/10/2024

## Purpose
This script is used in the publication titled **"Optimal combination therapy for heterogeneous cell populations with drug synergies."** It implements a mathematical model to optimize drug administration for achieving desired therapeutic outcomes.

## Model Description
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

### State and Drug Vectors
- **State Vector:** \([Na, Nb]\)
- **Drug Vector:** \([Uc, Up]\)

## Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib

You can install the required libraries using pip:

```bash
pip install numpy scipy matplotlib
Usage
To run the script, execute the following command in your terminal:

bash
Copy code
python Figure2.py <alpha> <beta>
Replace <alpha> and <beta> with the desired parameters for the control cost matrix.

Output
The script will generate an output file named a=<alpha>-b=<beta>.txt, which contains:

Total cell count at the end of the optimal regime
Total cell count at the end of the mean administration regime
The ratio of final cell counts
Total cost of drug administration
Individual drug costs for cisplatin and paclitaxel
Code Structure
The code consists of several main functions:

Dynamics Functions: Define the linear and non-linear dynamics of the system.
Control Functions: Calculate the optimal control strategy based on the state and costate dynamics.
Boundary Conditions: Evaluate the boundary conditions for the boundary value problem (BVP).
Solvers: Functions to solve the optimal control problem using solve_bvp from SciPy.
Example Output
Upon running the script, the output file will contain space-separated values corresponding to the metrics defined above. Here’s an example of how the data might look:

Copy code
50.0 48.0 1.04 0.5 0.3 0.2
This output represents:

Total cell count in mean administration regime
Total cell count in optimal regime
Ratio of final cell counts
Total cost
Total cost for cisplatin
Total cost for paclitaxel
Conclusion
This script provides a framework for analyzing drug synergies in heterogeneous cell populations through optimal control theory. It can be modified or extended to accommodate different models or parameters based on research needs.
