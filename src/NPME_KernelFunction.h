//Copyright (c) 2025, Dennis M. Elking
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

#ifndef NPME_KERNEL_FUNCTION_H
#define NPME_KERNEL_FUNCTION_H

#include "NPME_Constant.h"

#include <immintrin.h>
#include <iostream> 

namespace NPME_Library
{
class NPME_KfuncReal
//input:  x[N], y[N], z[N] 
//output: f0[N], fX[N], fY[N], fZ[N]
//        f0 = f(x, y, z)
//        fX = df0/dx, fY = df0/dy, fZ = df0/dZ
//        x_f0[N] stores both x[N] (input) and f0[N] (output)
//        x_fX[N] stores both x[N] (input) and fX[N] (output)
//        y_fY[N] stores both y[N] (input) and fY[N] (output)
//        z_fZ[N] stores both z[N] (input) and fZ[N] (output)
{
public:
  NPME_KfuncReal () { } 
  virtual ~NPME_KfuncReal () { }
  NPME_KfuncReal (const NPME_KfuncReal& rhs)  { }
  NPME_KfuncReal& operator= (const NPME_KfuncReal& rhs) { return *this; }

  virtual void Print (std::ostream& os) const = 0;

  virtual void Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const = 0;
  virtual void Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const = 0;

  //AVX intrinsic functions of above 
  //arrays are aligned 32 byte arrays and N is a multiple of 4
  #if NPME_USE_AVX
  virtual void CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const = 0;
  virtual void CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const = 0;
  #endif

  //AVX_512 intrinsic functions of above 
  //arrays are aligned 64 byte arrays and N is a multiple of 8
  #if NPME_USE_AVX_512
  virtual void CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const = 0;
  virtual void CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const = 0;
  #endif
};

class NPME_KfuncComplex
//input:  x[N], y[N], z[N]
//output: f0_r[N], fX_r[N], fY_r[N], fZ_r[N]
//        f0_i[N], fX_i[N], fY_i[N], fZ_i[N]
//        f0 = f0_r + I*f0_i
//        fX = fX_r + I*fX_i
//        fY = fY_r + I*fY_i
//        fZ = fZ_r + I*fZ_i
//        fX = df0/dx, fY = df0/dy, fZ = df0/dZ
//        x_f0_r[N] stores both x[N] (input) and f0_r[N] (output)
//        x_fX_r[N] stores both x[N] (input) and fX_r[N] (output)
//        y_fY_r[N] stores both y[N] (input) and fY_r[N] (output)
//        z_fZ_r[N] stores both z[N] (input) and fZ_r[N] (output)
{
public:
  NPME_KfuncComplex () { } 
  virtual ~NPME_KfuncComplex () { }
  NPME_KfuncComplex (const NPME_KfuncComplex& rhs)  { }
  NPME_KfuncComplex& operator= (const NPME_KfuncComplex& rhs) { return *this; }

  virtual void Print (std::ostream& os) const = 0;

  virtual void Calc (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const = 0;
  virtual void Calc (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const = 0;

  #if NPME_USE_AVX
  virtual void CalcAVX (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const = 0;
  virtual void CalcAVX (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const = 0;
  #endif

  #if NPME_USE_AVX_512
  virtual void CalcAVX_512 (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const = 0;
  virtual void CalcAVX_512 (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const = 0;
  #endif
};

bool NPME_KernelFuncCheck (NPME_Library::NPME_KfuncReal& func, 
  const size_t N, const char *funcName, const double Xmin, const double Xmax, 
  int vecOption, bool PRINT, bool PRINT_ALL, std::ostream& os);
bool NPME_KernelFuncCheck (NPME_Library::NPME_KfuncComplex& func, 
  const size_t N, const char *funcName, const double Xmin, const double Xmax, 
  int vecOption, bool PRINT, bool PRINT_ALL, std::ostream& os);
//tests func with numerical derivatives and 
//compares AVX and AVX_512 implementation with scalar implementation
//N is the array size to test on
//random x,y,z coordinates between (Xmin,Xmax)
//vecOption = 0, 1, 2 for no vectorization, AVX, AVX-512




bool NPME_KernelFuncCheck (NPME_Library::NPME_KfuncReal& func, 
  const size_t N, const char *funcName, const double Xmin, const double Xmax, 
  bool PRINT, bool PRINT_ALL, std::ostream& os);
bool NPME_KernelFuncCheck (NPME_Library::NPME_KfuncComplex& func, 
  const size_t N, const char *funcName, const double Xmin, const double Xmax, 
  bool PRINT, bool PRINT_ALL, std::ostream& os);
//determines vecOption = 0, 1, 2 from NPME_USE_AVX and NPME_USE_AVX_512 flags



double NPME_KernelFunc_GetTime (NPME_Library::NPME_KfuncReal& func, 
  const size_t N, int vecOption, 
  const double Xmin = -1.0, const double Xmax = 1.0);
//Gets CPU Time for calculating f[N], dfdx[N], dfdy[N], dfdz[N]
//vecOption = 0, 1, 2 for no vectorization, AVX, AVX-512


}//end namespace NPME_Library


#endif // NPME_KERNEL_FUNCTION_H



