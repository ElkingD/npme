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

#ifndef NPME_KERNEL_FUNCTION_INV_R_H
#define NPME_KERNEL_FUNCTION_INV_R_H

#include <cstdlib> 
#include <iostream> 
#include <vector> 

#include <immintrin.h>

#include "NPME_Constant.h"
#include "NPME_KernelFunction.h"

namespace NPME_Library
{
//DM = derivative match
//LR = long  range part of Ewald splitting 
//SR = short range part of Ewald splitting 


class NPME_Kfunc_Laplace : public NPME_KfuncReal
//calculates 1.0/r
{
public:
  NPME_Kfunc_Laplace  () : NPME_KfuncReal()  { }
  virtual ~NPME_Kfunc_Laplace ()  { } 

  NPME_Kfunc_Laplace (const NPME_Kfunc_Laplace& rhs) : NPME_KfuncReal(rhs)
  {
  }
  NPME_Kfunc_Laplace& operator= (const NPME_Kfunc_Laplace& rhs)
  {
    NPME_KfuncReal::operator= (rhs);
    return *this;
  }

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

};


class NPME_Kfunc_Laplace_LR_DM : public NPME_KfuncReal
//calculates f(r) = 1/r       for r >= Rdir
//           f(r) = a[n]*r^2n for r <= Rdir
//where a[n] (n = 0, 1,.. N) coefficients are determined by matching derivatives
//at r = Rdir
{
public:
  NPME_Kfunc_Laplace_LR_DM  () : NPME_KfuncReal()  { }
  NPME_Kfunc_Laplace_LR_DM  (int Nder, double Rdir) : NPME_KfuncReal()
  {
    SetParm (Nder, Rdir, 0, std::cout);
  }
  virtual ~NPME_Kfunc_Laplace_LR_DM ()  { } 

  NPME_Kfunc_Laplace_LR_DM (const NPME_Kfunc_Laplace_LR_DM& rhs) : 
    NPME_KfuncReal(rhs)
  {
    _Nder = rhs._Nder;
    _Rdir = rhs._Rdir;
    memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(double));
    memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(double));
  }
  NPME_Kfunc_Laplace_LR_DM& operator= (const NPME_Kfunc_Laplace_LR_DM& rhs)
  {
    NPME_KfuncReal::operator= (rhs);
    if (this != &rhs)
    {
      _Nder = rhs._Nder;
      _Rdir = rhs._Rdir;
      memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(double));
      memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(double));
    }
    return *this;
  }

  
  bool SetParm (const int Nder, const double Rdir, 
      bool PRINT, std::ostream& os);

  void Print (std::ostream& os) const;

  void Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;

  //AVX intrinsic functions of above
  #if NPME_USE_AVX
  void CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

  #if NPME_USE_AVX_512
  void CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

private:
  int    _Nder;
  double _Rdir;
  double _a[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
  double _b[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
};

class NPME_Kfunc_Laplace_SR_DM : public NPME_KfuncReal
//calculates f(r) = 0               for r >= Rdir
//           f(r) = 1/r - a[n]*r^2n for r <= Rdir
//where a[n] (n = 0, 1,.. N) coefficients are determined by matching derivatives
//at r = Rdir
{
public:
  NPME_Kfunc_Laplace_SR_DM  () : NPME_KfuncReal()  { }
  NPME_Kfunc_Laplace_SR_DM  (int Nder, double Rdir) : NPME_KfuncReal()
  {
    SetParm (Nder, Rdir, 0, std::cout);
  }
  virtual ~NPME_Kfunc_Laplace_SR_DM ()  { } 

  NPME_Kfunc_Laplace_SR_DM (const NPME_Kfunc_Laplace_SR_DM& rhs) : 
    NPME_KfuncReal(rhs)
  {
    _Nder = rhs._Nder;
    _Rdir = rhs._Rdir;
    memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(double));
    memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(double));
  }
  NPME_Kfunc_Laplace_SR_DM& operator= (const NPME_Kfunc_Laplace_SR_DM& rhs)
  {
    NPME_KfuncReal::operator= (rhs);
    if (this != &rhs)
    {
      _Nder = rhs._Nder;
      _Rdir = rhs._Rdir;
      memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(double));
      memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(double));
    }
    return *this;
  }

  bool SetParm (const int Nder, const double Rdir, 
          bool PRINT, std::ostream& os);

  void Print (std::ostream& os) const;

  void Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;

  #if NPME_USE_AVX
  void CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif
  #if NPME_USE_AVX_512
  void CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

private:
  int    _Nder;
  double _Rdir;
  double _a[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
  double _b[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
};


class NPME_Kfunc_Laplace_LR_Original : public NPME_KfuncReal
//calculates f(r) = erf(beta*r)/r
{
public:
  NPME_Kfunc_Laplace_LR_Original  () : NPME_KfuncReal()  { }
  NPME_Kfunc_Laplace_LR_Original  (double beta) : NPME_KfuncReal()
  {
    SetParm (beta);
  }
  virtual ~NPME_Kfunc_Laplace_LR_Original ()  { } 

  NPME_Kfunc_Laplace_LR_Original (const NPME_Kfunc_Laplace_LR_Original& rhs) : 
    NPME_KfuncReal(rhs)
  {
    _beta   = rhs._beta;
    _beta3  = rhs._beta3;
  }
  NPME_Kfunc_Laplace_LR_Original& operator= 
    (const NPME_Kfunc_Laplace_LR_Original& rhs)
  {
    NPME_KfuncReal::operator= (rhs);
    if (this != &rhs)
    {
      _beta   = rhs._beta;
      _beta3  = rhs._beta3;
    }
    return *this;
  }

  void SetParm (const double beta);

  void Print (std::ostream& os) const;

  void Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;

  #if NPME_USE_AVX
  void CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif
  #if NPME_USE_AVX_512
  void CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

private:
  double _beta, _beta3;
};

class NPME_Kfunc_Laplace_SR_Original : public NPME_KfuncReal
//calculates f(r) = erf(beta*r)/r
{
public:
  NPME_Kfunc_Laplace_SR_Original  () : NPME_KfuncReal()  { }
  NPME_Kfunc_Laplace_SR_Original  (double beta) : NPME_KfuncReal()
  {
    SetParm (beta);
  }
  virtual ~NPME_Kfunc_Laplace_SR_Original ()  { } 

  NPME_Kfunc_Laplace_SR_Original (const NPME_Kfunc_Laplace_SR_Original& rhs) : 
    NPME_KfuncReal(rhs)
  {
    _beta   = rhs._beta;
    _beta3  = rhs._beta3;
  }
  NPME_Kfunc_Laplace_SR_Original& operator= 
    (const NPME_Kfunc_Laplace_SR_Original& rhs)
  {
    NPME_KfuncReal::operator= (rhs);
    if (this != &rhs)
    {
      _beta   = rhs._beta;
      _beta3  = rhs._beta3;
    }
    return *this;
  }

  void SetParm (const double beta);

  void Print (std::ostream& os) const;

  void Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;

  #if NPME_USE_AVX
  void CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif
  #if NPME_USE_AVX_512
  void CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

private:
  double _beta, _beta3;
};


double NPME_EwaldSplitOrig_Rdir2Beta (const double Rdir, const double tol);

}//end namespace NPME_Library


#endif // NPME_KERNEL_FUNCTION_INV_R_H



