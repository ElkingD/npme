# NPME: Non-Periodic Particle Mesh Ewald

https://doi.org/10.1016/j.cpc.2025.109739
C++ implementation of the Non-Periodic Particle Mesh Ewald (NPME) method for radially symmetric kernels.

---

## Description
The NPME method extends the smooth Particle Mesh Ewald (PME) algorithm to **non-periodic charge systems** interacting via a radially symmetric kernel \( f(r) \). 
It replaces periodic FFTs with sine transforms and introduces **derivative-matched kernel splitting** for improved performance and flexibility. 
NPME supports predefined kernels (\(1/r, r^\alpha, e^{ik_0 r}/r\)) and user-defined kernels via C++ classes.

---

## References and Resources
- **Accepted manuscript (CPC preprint PDF):** [📄 Download](docs/NPME_CPC_Preprint.pdf) 
- **Published in Computer Physics Communications:** [DOI: 10.1016/j.cpc.2025.109739](https://doi.org/10.1016/j.cpc.2025.109739) 
- **Cassyni Seminar Talk:** [Watch here](https://cassyni.com/events/3gMtbmEfjR8JvWTEEEbkay) 
- **Intel oneAPI Compiler:** [Download here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)

---

## Installation
The NPME code requires the Intel C++ compiler and associated Math Kernel Library (MKL). 
As of **npme v1.2**, the code has been updated to support the free Intel oneAPI compiler suite.

To compile using 4 threads:
```bash
make -j4

