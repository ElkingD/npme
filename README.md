# NPME: Non-Periodic Particle Mesh Ewald
C++ implementation of the Non-Periodic Particle Mesh Ewald (NPME) method for radially symmetric kernels.  
---

# Description
The non-periodic particle mesh Ewald (NPME) method is a fast method for calculating pair-wise potentials for a set charges interacting via radially symmetric kernel $f(r)$ in free space.  NPME extends the smooth Particle Mesh Ewald (PME) algorithm to non-periodic charge systems with arbitrary radially symmetric kernels by first splitting the kernel $f(r)$ into short-range $f_{s}(r)$ and smooth long-range $f_{l}(r)$ components.  $f_{l}(r)$ is represented numerically as a Fourier extension calculated with discrete Fourier interpolation.  This numerical representation for $f_{l}(r)$ leads to flexibility in the kernel, kernel splitting, and application to anisotropic rectangular volumes.  In particular, the derivative match (DM) splitting is applicable to arbitrary radially symmetric kernels and has additional performance capabilities.  npme is an open source implementation of the NPME method.  The current npme version supports predefined kernels $1/r, r^{\alpha}, \exp(ik_{0}r)/r$ and also user-defined kernels via C++ classes.

---

# References and Resources
- Accepted manuscript (CPC preprint PDF): [📄 Download](docs/npme_preprint.pdf) 
- Published in Computer Physics Communications: [DOI: 10.1016/j.cpc.2025.109739](https://doi.org/10.1016/j.cpc.2025.109739) 
- Cassyni Seminar Talk: [Watch here](https://cassyni.com/events/3gMtbmEfjR8JvWTEEEbkay) 
- Intel oneAPI Compiler: [Download here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)

---

# Installation
The NPME code requires the Intel C++ compiler and associated Math Kernel Library (MKL). 
As of npme v1.2, the code has been updated to support the free Intel oneAPI compiler suite.

To compile using 4 threads:  
>make -j4

---

## Directory Structure
/src    - NPME core library source code  
/app    - Command-line applications  
/test   - Test cases and example input scripts  
/doc    - User manual and CPC preprint PDF

---

## Example Test Run
cd ./test/01_npme_laplaceDM  
./run.sh

---

## Compatibility Notes
v1.2 and later: Compatible with Intel oneAPI compiler  
v1.0, v1.1: Require Intel Classic C++ Compiler (not supported in v1.2)

---

## Example Test Run
cd ./test/01_npme_laplaceDM  
./run.sh

---

## Citation
When using npme, please cite the following references:  
[1] D. M. Elking, “A non-periodic particle mesh Ewald method for radially symmetric kernels in free space”, Comput. Phys. Comm. 315, 109739 (2025). https://doi.org/10.1016/j.cpc.2025.109739  
[2] U. Essmann, L. Perera, M. L. Berkowitz, T. Darden, H. Lee, and L. G. Pedersen, “A smooth particle mesh Ewald method”, J. Chem. Phys. 103, 8577 (1995). https://doi.org/10.1063/1.470117  

---

## Contact
Dennis M. Elking  
FieldDyne L.L.C.  
delking@fielddyne.com

---


