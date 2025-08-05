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

#ifndef NPME_KERNEL_FUNCTION_R_ALPHA_H
#define NPME_KERNEL_FUNCTION_R_ALPHA_H

#include "Constant.h"

#include <immintrin.h>

#include "KernelFunction.h"

namespace NPME_Library
{
//DM = derivative match
//LR = long  range part of Ewald splitting 
//SR = short range part of Ewald splitting 

class NPME_Kfunc_Ralpha : public NPME_KfuncReal
//calculates f0 = r^alpha
{
public:
  NPME_Kfunc_Ralpha  () : NPME_KfuncReal()  { }
  NPME_Kfunc_Ralpha  (double alpha) : NPME_KfuncReal()
  {
    SetParm (alpha);
  }
  virtual ~NPME_Kfunc_Ralpha ()  { } 

  NPME_Kfunc_Ralpha (const NPME_Kfunc_Ralpha& rhs) : 
    NPME_KfuncReal(rhs)
  {
    _alpha  = rhs._alpha;
  }
  NPME_Kfunc_Ralpha& operator= (const NPME_Kfunc_Ralpha& rhs)
  {
    NPME_KfuncReal::operator= (rhs);
    if (this != &rhs)
    {
      _alpha  = rhs._alpha;
    }
    return *this;
  }

  bool SetParm (const double alpha);

  void Print (std::ostream& os) const;

  void Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;

  //AVX intrinsic functions of above 
  //arrays are aligned 32 byte arrays and N is a multiple of 4
  #if NPME_USE_AVX
  void CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

  #if NPME_USE_AVX_512
  //AVX_512 intrinsic functions of above 
  //arrays are aligned 64 byte arrays and N is a multiple of 8
  void CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

private:
  double _alpha;
};

class NPME_Kfunc_Ralpha_LR_DM : public NPME_KfuncReal
//calculates f(r) = r^alpha   for r >= Rdir
//           f(r) = a[n]*r^2n for r <= Rdir
//where a[n] (n = 0, 1,.. N) coefficients are determined by matching derivatives
//at r = Rdir
{
public:
  NPME_Kfunc_Ralpha_LR_DM  () : NPME_KfuncReal()  { }
  NPME_Kfunc_Ralpha_LR_DM  (double alpha, int Nder, double Rdir) 
    : NPME_KfuncReal()
  {
    SetParm (alpha, Nder, Rdir, 0, std::cout);
  }
  virtual ~NPME_Kfunc_Ralpha_LR_DM ()  { } 

  NPME_Kfunc_Ralpha_LR_DM (const NPME_Kfunc_Ralpha_LR_DM& rhs) : 
    NPME_KfuncReal(rhs)
  {
    _alpha  = rhs._alpha;
    _Nder   = rhs._Nder;
    _Rdir   = rhs._Rdir;
    memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(double));
    memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(double));
  }
  NPME_Kfunc_Ralpha_LR_DM& operator= (const NPME_Kfunc_Ralpha_LR_DM& rhs)
  {
    NPME_KfuncReal::operator= (rhs);
    if (this != &rhs)
    {
      _alpha  = rhs._alpha;
      _Nder   = rhs._Nder;
      _Rdir   = rhs._Rdir;
      memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(double));
      memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(double));
    }
    return *this;
  }


  bool SetParm (const double alpha, const int Nder, 
    const double Rdir, bool PRINT, std::ostream& os);

  void Print (std::ostream& os) const;

  void Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;

  //AVX intrinsic functions of above 
  //arrays are aligned 32 byte arrays and N is a multiple of 4
  #if NPME_USE_AVX
  void CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

  #if NPME_USE_AVX_512
  //AVX_512 intrinsic functions of above 
  //arrays are aligned 64 byte arrays and N is a multiple of 8
  void CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

private:
  double _alpha;
  int    _Nder;
  double _Rdir;
  double _a[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
  double _b[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
};

class NPME_Kfunc_Ralpha_SR_DM : public NPME_KfuncReal
//calculates f(r) = 0                   for r >= Rdir
//           f(r) = r^alpha - a[n]*r^2n for r <= Rdir
//where a[n] (n = 0, 1,.. N) coefficients are determined by matching derivatives
//at r = Rdir
{
public:
  NPME_Kfunc_Ralpha_SR_DM  () : NPME_KfuncReal()  { }
  NPME_Kfunc_Ralpha_SR_DM  (double alpha, int Nder, double Rdir) 
    : NPME_KfuncReal()
  {
    SetParm (alpha, Nder, Rdir, 0, std::cout);
  }
  virtual ~NPME_Kfunc_Ralpha_SR_DM ()  { } 

  NPME_Kfunc_Ralpha_SR_DM (const NPME_Kfunc_Ralpha_SR_DM& rhs) : 
    NPME_KfuncReal(rhs)
  {
    _alpha  = rhs._alpha;
    _Nder   = rhs._Nder;
    _Rdir   = rhs._Rdir;
    memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(double));
    memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(double));
  }
  NPME_Kfunc_Ralpha_SR_DM& operator= (const NPME_Kfunc_Ralpha_SR_DM& rhs)
  {
    NPME_KfuncReal::operator= (rhs);
    if (this != &rhs)
    {
      _alpha  = rhs._alpha;
      _Nder   = rhs._Nder;
      _Rdir   = rhs._Rdir;
      memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(double));
      memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(double));
    }
    return *this;
  }

  bool SetParm (const double alpha, const int Nder, 
    const double Rdir, bool PRINT, std::ostream& os);

  void Print (std::ostream& os) const;

  void Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;

  //AVX intrinsic functions of above 
  //arrays are aligned 32 byte arrays and N is a multiple of 4
  #if NPME_USE_AVX
  void CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

  #if NPME_USE_AVX_512
  //AVX_512 intrinsic functions of above 
  //arrays are aligned 64 byte arrays and N is a multiple of 8
  void CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

private:
  double _alpha;
  int    _Nder;
  double _Rdir;
  double _a[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
  double _b[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
};

}//end namespace NPME_Library


#endif // NPME_KERNEL_FUNCTION_R_ALPHA_H



