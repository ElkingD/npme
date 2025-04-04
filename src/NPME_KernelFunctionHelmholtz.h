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

#ifndef NPME_FUNCTION_HELMHOLTZ_H
#define NPME_FUNCTION_HELMHOLTZ_H

#include <iostream> 

#include <immintrin.h>

#include "NPME_KernelFunction.h"

namespace NPME_Library
{
//DM = derivative match
//LR = long  range part of Ewald splitting 
//SR = short range part of Ewald splitting 

class NPME_Kfunc_Helmholtz : public NPME_KfuncComplex
//calculates cexp(I*k0*r)/r
{
public:
  NPME_Kfunc_Helmholtz  () : NPME_KfuncComplex()  { }
  NPME_Kfunc_Helmholtz  (_Complex double k0) : NPME_KfuncComplex()
  {
    SetParm (k0);
  }
  virtual ~NPME_Kfunc_Helmholtz ()  { } 
  NPME_Kfunc_Helmholtz (const NPME_Kfunc_Helmholtz& rhs) : 
    NPME_KfuncComplex(rhs)
  {
    _k0 = rhs._k0;
  }
  NPME_Kfunc_Helmholtz& operator= (const NPME_Kfunc_Helmholtz& rhs)
  {
    NPME_KfuncComplex::operator= (rhs);
    if (this != &rhs)
    {
      _k0 = rhs._k0;
    }
    return *this;
  }

  bool SetParm (const _Complex double k0);

  void Print (std::ostream& os) const;

  void Calc (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const;
  void Calc (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const;

  #if NPME_USE_AVX
  void CalcAVX (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const;
  void CalcAVX (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const;
  #endif

  #if NPME_USE_AVX_512
  void CalcAVX_512 (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const;
  void CalcAVX_512 (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const;
  #endif

private:
  _Complex double _k0;
};





class NPME_Kfunc_Helmholtz_LR_DM : public NPME_KfuncComplex
//calculates f(r) = cexp(I*k0*r)/r for r >= Rdir
//           f(r) = a[n]*r^2n      for r <= Rdir
//where a[n] (n = 0, 1,.. N) coefficients are determined by matching derivatives
//at r = Rdir
{
public:
  NPME_Kfunc_Helmholtz_LR_DM  () : NPME_KfuncComplex()  { }
  NPME_Kfunc_Helmholtz_LR_DM  (_Complex double k0, int Nder, double Rdir) 
    : NPME_KfuncComplex()
  {
    SetParm (k0, Nder, Rdir, 0, std::cout);
  }
  virtual ~NPME_Kfunc_Helmholtz_LR_DM ()  { } 
  NPME_Kfunc_Helmholtz_LR_DM (const NPME_Kfunc_Helmholtz_LR_DM& rhs) : 
    NPME_KfuncComplex(rhs)
  {
    _k0   = rhs._k0;
    _Nder = rhs._Nder;
    _Rdir = rhs._Rdir;
    memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(_Complex double));
    memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(_Complex double));
  }
  NPME_Kfunc_Helmholtz_LR_DM& operator= (const NPME_Kfunc_Helmholtz_LR_DM& rhs)
  {
    NPME_KfuncComplex::operator= (rhs);
    if (this != &rhs)
    {
      _k0   = rhs._k0;
      _Nder = rhs._Nder;
      _Rdir = rhs._Rdir;
      memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(_Complex double));
      memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(_Complex double));
    }
    return *this;
  }


  bool SetParm (const _Complex double k0, 
    const int Nder, const double Rdir, bool PRINT, std::ostream& os);

  void Print (std::ostream& os) const;

  void Calc (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const;
  void Calc (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const;

  #if NPME_USE_AVX
  void CalcAVX (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const;
  void CalcAVX (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const;
  #endif

  #if NPME_USE_AVX_512
  void CalcAVX_512 (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const;
  void CalcAVX_512 (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const;
  #endif

private:
  _Complex double _k0;
  int    _Nder;
  double _Rdir;
  _Complex double _a[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
  _Complex double _b[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
};


class NPME_Kfunc_Helmholtz_SR_DM : public NPME_KfuncComplex
//calculates f(r) = 0                           for r >= Rdir
//           f(r) = cexp(I*k0*r)/r - a[n]*r^2n  for r <= Rdir
//where a[n] (n = 0, 1,.. N) coefficients are determined by matching derivatives
//at r = Rdir
{
public:
  NPME_Kfunc_Helmholtz_SR_DM  () : NPME_KfuncComplex()  { }
  NPME_Kfunc_Helmholtz_SR_DM  (_Complex double k0, int Nder, double Rdir) 
    : NPME_KfuncComplex()
  {
    SetParm (k0, Nder, Rdir, 0, std::cout);
  }
  virtual ~NPME_Kfunc_Helmholtz_SR_DM ()  { } 
  NPME_Kfunc_Helmholtz_SR_DM (const NPME_Kfunc_Helmholtz_SR_DM& rhs) : 
    NPME_KfuncComplex(rhs)
  {
    _k0   = rhs._k0;
    _Nder = rhs._Nder;
    _Rdir = rhs._Rdir;
    memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(_Complex double));
    memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(_Complex double));
  }
  NPME_Kfunc_Helmholtz_SR_DM& operator= (const NPME_Kfunc_Helmholtz_SR_DM& rhs)
  {
    NPME_KfuncComplex::operator= (rhs);
    if (this != &rhs)
    {
      _k0   = rhs._k0;
      _Nder = rhs._Nder;
      _Rdir = rhs._Rdir;
      memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(_Complex double));
      memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(_Complex double));
    }
    return *this;
  }

  bool SetParm (const _Complex double k0, 
    const int Nder, const double Rdir, bool PRINT, std::ostream& os);

  void Print (std::ostream& os) const;

  void Calc (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const;
  void Calc (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const;

  #if NPME_USE_AVX
  void CalcAVX (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const;
  void CalcAVX (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const;
  #endif

  #if NPME_USE_AVX_512
  void CalcAVX_512 (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const;
  void CalcAVX_512 (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const;
  #endif

private:
  _Complex double _k0;
  int    _Nder;
  double _Rdir;
  _Complex double _a[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
  _Complex double _b[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
};

class NPME_Kfunc_Helmholtz_LR_Alt : public NPME_KfuncComplex
//calculates f(r) = erf(beta*r)/r*cos(k0*r) + I*sin(k0*r)/r
//k0 must be real
//note that NPME_Kfunc_Helmholtz_LR_Alt is complex, but
//          NPME_Kfunc_Helmholtz_SR_Alt is real
{
public:
  NPME_Kfunc_Helmholtz_LR_Alt  () : NPME_KfuncComplex()  { }
  NPME_Kfunc_Helmholtz_LR_Alt  (double k0, double beta) 
    : NPME_KfuncComplex()
  {
    SetParm (k0, beta);
  }
  virtual ~NPME_Kfunc_Helmholtz_LR_Alt ()  { } 
  NPME_Kfunc_Helmholtz_LR_Alt (const NPME_Kfunc_Helmholtz_LR_Alt& rhs) : 
    NPME_KfuncComplex(rhs)
  {
    _k0         = rhs._k0;
    _k03        = rhs._k03;
    _k02_beta   = rhs._k02_beta;
    _beta       = rhs._beta;
    _beta3      = rhs._beta3;
  }
  NPME_Kfunc_Helmholtz_LR_Alt& operator= 
    (const NPME_Kfunc_Helmholtz_LR_Alt& rhs)
  {
    NPME_KfuncComplex::operator= (rhs);
    if (this != &rhs)
    {
      _k0         = rhs._k0;
      _k03        = rhs._k03;
      _k02_beta   = rhs._k02_beta;
      _beta       = rhs._beta;
      _beta3      = rhs._beta3;
    }
    return *this;
  }

  bool SetParm (const double k0, const double beta);

  void Print (std::ostream& os) const;

  void Calc (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const;
  void Calc (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const;

  #if NPME_USE_AVX
  void CalcAVX (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const;
  void CalcAVX (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const;
  #endif

  #if NPME_USE_AVX_512
  void CalcAVX_512 (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const;
  void CalcAVX_512 (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const;
  #endif

private:
  double _k0, _k03, _k02_beta;
  double _beta, _beta3;
};

class NPME_Kfunc_Helmholtz_SR_Alt : public NPME_KfuncReal
//calculates f(r) = erfc(beta*r)/r*cos(k0*r)
//k0 must be real
//note that NPME_Kfunc_Helmholtz_LR_Alt is complex, but
//          NPME_Kfunc_Helmholtz_SR_Alt is real
{
public:
  NPME_Kfunc_Helmholtz_SR_Alt  () : NPME_KfuncReal()  { }
  NPME_Kfunc_Helmholtz_SR_Alt  (double k0, double beta) 
    : NPME_KfuncReal()
  {
    SetParm (k0, beta);
  }
  virtual ~NPME_Kfunc_Helmholtz_SR_Alt ()  { } 
  NPME_Kfunc_Helmholtz_SR_Alt (const NPME_Kfunc_Helmholtz_SR_Alt& rhs) : 
    NPME_KfuncReal(rhs)
  {
    _k0         = rhs._k0;
    _k03        = rhs._k03;
    _k02_beta   = rhs._k02_beta;
    _beta       = rhs._beta;
    _beta3      = rhs._beta3;
  }
  NPME_Kfunc_Helmholtz_SR_Alt& operator= 
    (const NPME_Kfunc_Helmholtz_SR_Alt& rhs)
  {
    NPME_KfuncReal::operator= (rhs);
    if (this != &rhs)
    {
      _k0         = rhs._k0;
      _k03        = rhs._k03;
      _k02_beta   = rhs._k02_beta;
      _beta       = rhs._beta;
      _beta3      = rhs._beta3;
    }
    return *this;
  }

  bool SetParm (const double k0, const double beta);

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
  double _k0, _k03, _k02_beta;
  double _beta, _beta3;
};
}//end namespace NPME_Library


#endif // NPME_FUNCTION_HELMHOLTZ_H



