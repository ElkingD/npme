# NPME: Non-Periodic Particle Mesh Ewald    

## Description
npme is a C++ implementation of the non-Periodic Particle Mesh Ewald (NPME) method - a fast algorithm for calculating pairwise potentials for a set of charges interacting via a radially symmetric kernel $f(r)$ in free space.

NPME extends the smooth Particle Mesh Ewald (PME) algorithm to non-periodic charge systems with arbitrary radially symmetric kernels by splitting the kernel $f(r)$ into:
- Short-range component $f_{s}(r)$
- Smooth long-range component $f_{l}(r)$

The smooth long-range component $f_{l}(r)$ is represented numerically as a Fourier extension, computed via discrete Fourier interpolation. This numerical representation provides flexibility in:
- Kernel choice
- Kernel splitting strategy
- Application to anisotropic rectangular volumes

The derivative match (DM) splitting is applicable to arbitrary radially symmetric kernels and offers additional performance improvements.

npme is open source and currently supports:
- Predefined kernels: $1/r$, $r^{\alpha}$, $\exp(ik_{0}r)/r$
- User-defined kernels via C++ classes
---

## References and Resources
- Accepted manuscript (CPC preprint PDF): [Download](docs/npme_preprint.pdf) 
- Published in Computer Physics Communications: [DOI: 10.1016/j.cpc.2025.109739](https://doi.org/10.1016/j.cpc.2025.109739) 
- Cassyni Seminar Talk: [Watch here](https://cassyni.com/events/3gMtbmEfjR8JvWTEEEbkay) 
- Intel oneAPI Compiler: [Download here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)

---

## Installation
The NPME code requires the Intel C++ compiler and associated Math Kernel Library (MKL). 
As of npme v1.2, the code has been updated to support the free Intel oneAPI compiler suite.

To compile using 4 threads:  
```
make -j4
```
---

## Directory Structure
/src    - NPME core library source code  
/app    - Command-line applications  
/test   - Test cases and example input scripts  
/doc    - User manual and CPC preprint PDF

---

## Example Test Run
```
cd ./test/01_npme_laplaceDM
./run.sh
```
---

## Compatibility Notes
v1.2 and later: Compatible with Intel oneAPI compiler  
v1.0, v1.1: Require Intel Classic C++ Compiler (not supported in v1.2)

---

## Citation
When using npme, please cite the following references:  
[1] D. M. Elking, “A non-periodic particle mesh Ewald method for radially symmetric kernels in free space”, Comput. Phys. Comm. 315, 109739 (2025). https://doi.org/10.1016/j.cpc.2025.109739  
[2] U. Essmann, L. Perera, M. L. Berkowitz, T. Darden, H. Lee, and L. G. Pedersen, “A smooth particle mesh Ewald method”, J. Chem. Phys. 103, 8577 (1995). https://doi.org/10.1063/1.470117  

---

## Contact
Dennis M. Elking  
FieldDyne, L.L.C.  
delking@fielddyne.com

---


