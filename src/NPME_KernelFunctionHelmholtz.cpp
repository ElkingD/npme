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

#include <cstdio>
#include <cstdlib> 
#include <cstring> 
#include <cmath> 
#include <cstdio>

#include <iostream> 
#include <vector>

#include <immintrin.h>


#include "NPME_Constant.h"
#include "NPME_KernelFunction.h"
#include "NPME_KernelFunctionHelmholtz.h"
#include "NPME_FunctionDerivMatch.h"
#include "NPME_MathFunctions.h"

namespace NPME_Library
{
//******************************************************************************
//******************************************************************************
//**************************Some Support Functions******************************
//******************************************************************************
//******************************************************************************

#if NPME_USE_AVX
void NPME_KfuncHelmholtzRadial_AVX (__m256d& f0rVec, __m256d& f0iVec,
  const __m256d& rVec, const __m256d& k0rVec256, const __m256d& nk0iVec256)
{
  __m256d  k0r_r_Vec  = _mm256_mul_pd ( k0rVec256, rVec);
  __m256d nk0i_r_Vec  = _mm256_mul_pd (nk0iVec256, rVec);

  __m256d sinVec, cosVec, expVec;
  sinVec    = _mm256_sincos_pd (&cosVec, k0r_r_Vec);
  expVec    = _mm256_exp_pd    (nk0i_r_Vec);
  expVec    = _mm256_div_pd  (expVec, rVec);

  f0rVec    = _mm256_mul_pd  (expVec,    cosVec);
  f0iVec    = _mm256_mul_pd  (expVec,    sinVec);
}

void NPME_KfuncHelmholtzRadial_AVX (__m256d& f0rVec, __m256d& f0iVec,
  __m256d& f1rVec, __m256d& f1iVec, const __m256d& rVec, 
  const __m256d& k0rVec256, const __m256d& nk0iVec256)
{
  __m256d sinVec, cosVec, expVec;
  sinVec  = _mm256_sincos_pd (&cosVec, _mm256_mul_pd (k0rVec256, rVec));

  expVec  = _mm256_exp_pd  (_mm256_mul_pd (nk0iVec256, rVec));
  expVec  = _mm256_div_pd  (expVec, rVec);

  f0rVec  = _mm256_mul_pd  (expVec,    cosVec);
  f0iVec  = _mm256_mul_pd  (expVec,    sinVec);


  //g = f0/r
  __m256d grVec = _mm256_div_pd  (f0rVec , rVec);
  __m256d giVec = _mm256_div_pd  (f0iVec , rVec);

  //f1 = 1/r df0/dr
  #if NPME_USE_AVX_FMA
  {
    f1rVec  = _mm256_fmadd_pd  ( k0rVec256, f0iVec,  grVec);
    f1rVec  = _mm256_fmsub_pd  (nk0iVec256, f0rVec, f1rVec);
    f1iVec  = _mm256_fmsub_pd  (nk0iVec256, f0iVec,  giVec);
    f1iVec  = _mm256_fmadd_pd  ( k0rVec256, f0rVec, f1iVec);
  }
  #else
  {
    f1rVec  = _mm256_add_pd  (_mm256_mul_pd ( k0rVec256, f0iVec),  grVec);
    f1rVec  = _mm256_sub_pd  (_mm256_mul_pd (nk0iVec256, f0rVec), f1rVec);
    f1iVec  = _mm256_sub_pd  (_mm256_mul_pd (nk0iVec256, f0iVec),  giVec);
    f1iVec  = _mm256_add_pd  (_mm256_mul_pd ( k0rVec256, f0rVec), f1iVec);
  }
  #endif

  f1rVec = _mm256_div_pd  (f1rVec, rVec);
  f1iVec = _mm256_div_pd  (f1iVec, rVec);
}
#endif

#if NPME_USE_AVX_512

void NPME_KfuncHelmholtzRadial_AVX_512 (__m512d& f0rVec, __m512d& f0iVec,
  const __m512d& rVec, const __m512d& k0rVec512, const __m512d& nk0iVec512)
{
  __m512d  k0r_r_Vec  = _mm512_mul_pd ( k0rVec512, rVec);
  __m512d nk0i_r_Vec  = _mm512_mul_pd (nk0iVec512, rVec);

  __m512d sinVec, cosVec, expVec;
  sinVec    = _mm512_sincos_pd (&cosVec, k0r_r_Vec);
  expVec    = _mm512_exp_pd    (nk0i_r_Vec);
  expVec    = _mm512_div_pd  (expVec, rVec);

  f0rVec    = _mm512_mul_pd  (expVec,    cosVec);
  f0iVec    = _mm512_mul_pd  (expVec,    sinVec);
}

void NPME_KfuncHelmholtzRadial_AVX_512 (__m512d& f0rVec, __m512d& f0iVec,
  __m512d& f1rVec, __m512d& f1iVec, const __m512d& rVec, 
  const __m512d& k0rVec512, const __m512d& nk0iVec512)
{
  __m512d sinVec, cosVec, expVec;
  sinVec  = _mm512_sincos_pd (&cosVec, _mm512_mul_pd (k0rVec512, rVec));

  expVec  = _mm512_exp_pd  (_mm512_mul_pd (nk0iVec512, rVec));
  expVec  = _mm512_div_pd  (expVec, rVec);

  f0rVec  = _mm512_mul_pd  (expVec,    cosVec);
  f0iVec  = _mm512_mul_pd  (expVec,    sinVec);

  //g = f0/r
  __m512d grVec = _mm512_div_pd  (f0rVec , rVec);
  __m512d giVec = _mm512_div_pd  (f0iVec , rVec);

  //f1 = 1/r df0/dr
  f1rVec  = _mm512_fmadd_pd  ( k0rVec512, f0iVec,  grVec);
  f1rVec  = _mm512_fmsub_pd  (nk0iVec512, f0rVec, f1rVec);
  f1iVec  = _mm512_fmsub_pd  (nk0iVec512, f0iVec,  giVec);
  f1iVec  = _mm512_fmadd_pd  ( k0rVec512, f0rVec, f1iVec);

  f1rVec = _mm512_div_pd  (f1rVec, rVec);
  f1iVec = _mm512_div_pd  (f1iVec, rVec);
}
#endif

//******************************************************************************
//******************************************************************************
//*****************************NPME_Kfunc_Helmholtz_****************************
//******************************************************************************
//******************************************************************************
bool NPME_Kfunc_Helmholtz::SetParm (const _Complex double k0)
{
  _k0 = k0;
  return true;
}

void NPME_Kfunc_Helmholtz::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Helmholtz::Print\n";
  sprintf(str, "  k0    = %.3f + I*%.3f\n", creal(_k0), cimag(_k0));
  os << str;
  os.flush();
}

void NPME_Kfunc_Helmholtz::Calc (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_f0_r[i]*x_f0_r[i] + y[i]*y[i] + z[i]*z[i];
    const double r  = sqrt(fabs(r2));

    _Complex double f0_tmp  = cexp(I*_k0*r)/r;
    x_f0_r[i]               = creal(f0_tmp);
    f0_i[i]                 = cimag(f0_tmp);
  }
}
void NPME_Kfunc_Helmholtz::Calc (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX_r[i]*x_fX_r[i] + 
                      y_fY_r[i]*y_fY_r[i] + 
                      z_fZ_r[i]*z_fZ_r[i];
    const double r  = sqrt(fabs(r2));

    _Complex double f0_tmp  = cexp(I*_k0*r)/r;
    _Complex double f1_tmp  = (I*_k0*f0_tmp - f0_tmp/r)/r;
    double f1_r             = creal(f1_tmp);
    double f1_i             = cimag(f1_tmp);

    f0_i[i]                 = cimag(f0_tmp);
    fX_i[i]                 = x_fX_r[i]*f1_i;
    fY_i[i]                 = y_fY_r[i]*f1_i;
    fZ_i[i]                 = z_fZ_r[i]*f1_i;

    f0_r[i]                 = creal(f0_tmp);
    x_fX_r[i]               = x_fX_r[i]*f1_r;
    y_fY_r[i]               = y_fY_r[i]*f1_r;
    z_fZ_r[i]               = z_fZ_r[i]*f1_r;
  }
}


#if NPME_USE_AVX
void NPME_Kfunc_Helmholtz::CalcAVX (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop        = N/4;
  const __m256d _k0rVec256  = _mm256_set1_pd( creal(_k0) );
  const __m256d _nk0iVec256 = _mm256_set1_pd(-cimag(_k0) );

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm256_load_pd (&x_f0_r[count]);
    yVec  = _mm256_load_pd (&y[count]);
    zVec  = _mm256_load_pd (&z[count]);

    r2Vec = _mm256_mul_pd  (xVec, xVec);
    #if NPME_USE_AVX_FMA
    {
      r2Vec  = _mm256_fmadd_pd  (yVec, yVec, r2Vec);
      r2Vec  = _mm256_fmadd_pd  (zVec, zVec, r2Vec);
    }
    #else
    {
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (yVec, yVec), r2Vec);
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (zVec, zVec), r2Vec);
    }
    #endif
    rVec  = _mm256_sqrt_pd (r2Vec);

    __m256d f0rVec, f0iVec;
    NPME_KfuncHelmholtzRadial_AVX (f0rVec, f0iVec, rVec, 
      _k0rVec256, _nk0iVec256);

    _mm256_store_pd (&x_f0_r[count], f0rVec);
    _mm256_store_pd (  &f0_i[count], f0iVec);

    count += 4;
  }
}

void NPME_Kfunc_Helmholtz::CalcAVX (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop        = N/4;
  const __m256d _k0rVec256  = _mm256_set1_pd( creal(_k0) );
  const __m256d _nk0iVec256 = _mm256_set1_pd(-cimag(_k0) );

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm256_load_pd (&x_fX_r[count]);
    yVec  = _mm256_load_pd (&y_fY_r[count]);
    zVec  = _mm256_load_pd (&z_fZ_r[count]);

    r2Vec = _mm256_mul_pd  (xVec, xVec);
    #if NPME_USE_AVX_FMA
    {
      r2Vec  = _mm256_fmadd_pd  (yVec, yVec, r2Vec);
      r2Vec  = _mm256_fmadd_pd  (zVec, zVec, r2Vec);
    }
    #else
    {
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (yVec, yVec), r2Vec);
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (zVec, zVec), r2Vec);
    }
    #endif
    rVec  = _mm256_sqrt_pd (r2Vec);


    __m256d f0rVec, f0iVec, f1rVec, f1iVec;
    NPME_KfuncHelmholtzRadial_AVX (f0rVec, f0iVec, f1rVec, f1iVec, rVec, 
      _k0rVec256, _nk0iVec256);

    __m256d fXrVec, fYrVec, fZrVec;
    __m256d fXiVec, fYiVec, fZiVec;
    fXrVec = _mm256_mul_pd  (f1rVec, xVec);
    fYrVec = _mm256_mul_pd  (f1rVec, yVec);
    fZrVec = _mm256_mul_pd  (f1rVec, zVec);

    fXiVec = _mm256_mul_pd  (f1iVec, xVec);
    fYiVec = _mm256_mul_pd  (f1iVec, yVec);
    fZiVec = _mm256_mul_pd  (f1iVec, zVec);

    _mm256_store_pd (  &f0_r[count], f0rVec);
    _mm256_store_pd (  &f0_i[count], f0iVec);

    _mm256_store_pd (&x_fX_r[count], fXrVec);
    _mm256_store_pd (&y_fY_r[count], fYrVec);
    _mm256_store_pd (&z_fZ_r[count], fZrVec);

    _mm256_store_pd (  &fX_i[count], fXiVec);
    _mm256_store_pd (  &fY_i[count], fYiVec);
    _mm256_store_pd (  &fZ_i[count], fZiVec);

    count += 4;
  }
}

#endif


#if NPME_USE_AVX_512
void NPME_Kfunc_Helmholtz::CalcAVX_512 (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop          = N/8;
  const __m512d _k0rVec512    = _mm512_set1_pd( creal(_k0) );
  const __m512d _nk0iVec512   = _mm512_set1_pd(-cimag(_k0) );

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm512_load_pd (&x_f0_r[count]);
    yVec  = _mm512_load_pd (&y[count]);
    zVec  = _mm512_load_pd (&z[count]);

    r2Vec = _mm512_mul_pd  (xVec, xVec);
    r2Vec = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm512_fmadd_pd  (zVec, zVec, r2Vec);

    rVec  = _mm512_sqrt_pd (r2Vec);

    __m512d f0rVec, f0iVec;
    NPME_KfuncHelmholtzRadial_AVX_512 (f0rVec, f0iVec, rVec, 
      _k0rVec512, _nk0iVec512);

    _mm512_store_pd (&x_f0_r[count], f0rVec);
    _mm512_store_pd (  &f0_i[count], f0iVec);

    count += 8;
  }
}

void NPME_Kfunc_Helmholtz::CalcAVX_512 (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop          = N/8;
  const __m512d _k0rVec512    = _mm512_set1_pd( creal(_k0) );
  const __m512d _nk0iVec512   = _mm512_set1_pd(-cimag(_k0) );

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm512_load_pd (&x_fX_r[count]);
    yVec  = _mm512_load_pd (&y_fY_r[count]);
    zVec  = _mm512_load_pd (&z_fZ_r[count]);

    r2Vec = _mm512_mul_pd  (xVec, xVec);
    r2Vec = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm512_fmadd_pd  (zVec, zVec, r2Vec);
    rVec  = _mm512_sqrt_pd (r2Vec);


    __m512d f0rVec, f0iVec, f1rVec, f1iVec;
    NPME_KfuncHelmholtzRadial_AVX_512 (f0rVec, f0iVec, f1rVec, f1iVec, rVec, 
      _k0rVec512, _nk0iVec512);

    __m512d fXrVec, fYrVec, fZrVec;
    __m512d fXiVec, fYiVec, fZiVec;
    fXrVec = _mm512_mul_pd  (f1rVec, xVec);
    fYrVec = _mm512_mul_pd  (f1rVec, yVec);
    fZrVec = _mm512_mul_pd  (f1rVec, zVec);

    fXiVec = _mm512_mul_pd  (f1iVec, xVec);
    fYiVec = _mm512_mul_pd  (f1iVec, yVec);
    fZiVec = _mm512_mul_pd  (f1iVec, zVec);

    _mm512_store_pd (  &f0_r[count], f0rVec);
    _mm512_store_pd (  &f0_i[count], f0iVec);

    _mm512_store_pd (&x_fX_r[count], fXrVec);
    _mm512_store_pd (&y_fY_r[count], fYrVec);
    _mm512_store_pd (&z_fZ_r[count], fZrVec);

    _mm512_store_pd (  &fX_i[count], fXiVec);
    _mm512_store_pd (  &fY_i[count], fYiVec);
    _mm512_store_pd (  &fZ_i[count], fZiVec);

    count += 8;
  }
}

#endif





//******************************************************************************
//******************************************************************************
//***********************NPME_Kfunc_Helmholtz_LR_DM*****************************
//******************************************************************************
//******************************************************************************



bool NPME_Kfunc_Helmholtz_LR_DM::SetParm (const _Complex double k0, 
  const int Nder, const double Rdir, bool PRINT, std::ostream& os)
{
  if (Nder > NPME_MaxDerivMatchOrder)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_LR_DM::SetParm\n";
    char str[2000];
    sprintf(str, "  Nder = %d > %d = NPME_MaxDerivMatchOrder\n", Nder,
      NPME_MaxDerivMatchOrder);
    std::cout << str;
    return false;
  }

  _k0   = k0;
  _Nder = Nder;
  _Rdir = Rdir;

  std::vector<_Complex double> fHelm(_Nder+1);
  NPME_FunctionDerivMatch_HelmholtzRadialDeriv (&fHelm[0], _Nder, _Rdir, _k0);
  if (!NPME_FunctionDerivMatch_CalcEvenSeries (&_a[0], &_b[0], 
    &fHelm[0], _Nder, _Rdir))
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_LR_DM::SetParm\n";
    std::cout << "NPME_FunctionDerivMatch_CalcEvenSeries failed\n";
    return false;
  }

  if (PRINT)
  {
    char str[2000];
    os << "\n\nNPME_Kfunc_Helmholtz_LR_DM::SetParm\n";
    sprintf(str, "   k0    = %10.6f + %10.6fi\n", creal(_k0), cimag(_k0));
    os << str;
    sprintf(str, "   Rdir  = %10.6f\n", _Rdir);
    os << str;
    sprintf(str, "   Nder  = %d\n", _Nder);
    os << str;
    os << "\n";
    for (int i = 0; i <= _Nder; i++)
    {
      sprintf(str, "      a[%4d] = %25.15le + %25.15lei\n", 
        i, creal(_a[i]), cimag(_a[i]));
      os << str;
    }
    os << "\n";
    for (int i = 0; i <= _Nder; i++)
    {
      sprintf(str, "      b[%4d] = %25.15le + %25.15lei\n", 
        i, creal(_b[i]), cimag(_b[i]));
      os << str;
    }
  }
  return true;
}

void NPME_Kfunc_Helmholtz_LR_DM::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Helmholtz_LR_DM::Print\n";
  sprintf(str, "  k0    = %.3f + I*%.3f\n", creal(_k0), cimag(_k0));
  os << str;
  sprintf(str, "  Rdir  = %.3f\n", _Rdir);    os << str;
  sprintf(str, "  Nder  = %3d\n", _Nder);     os << str;
  os.flush();
}

void NPME_Kfunc_Helmholtz_LR_DM::Calc (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_f0_r[i]*x_f0_r[i] + y[i]*y[i] + z[i]*z[i];
    const double r  = sqrt(fabs(r2));

    _Complex double f0_tmp;
    if (r > _Rdir)
      f0_tmp = cexp(I*_k0*r)/r;
    else
      f0_tmp = NPME_FunctionDerivMatch_EvenSeriesComplex (_Nder, &_a[0], r2);

    x_f0_r[i]               = creal(f0_tmp);
    f0_i[i]                 = cimag(f0_tmp);
  }
}
void NPME_Kfunc_Helmholtz_LR_DM::Calc (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX_r[i]*x_fX_r[i] + 
                      y_fY_r[i]*y_fY_r[i] + 
                      z_fZ_r[i]*z_fZ_r[i];
    const double r  = sqrt(fabs(r2));

    _Complex double f0_tmp, f1_tmp;
    if (r > _Rdir)
    {
      f0_tmp  = cexp(I*_k0*r)/r;
      f1_tmp  = (I*_k0*f0_tmp - f0_tmp/r)/r;
    }
    else
    {
      f0_tmp = NPME_FunctionDerivMatch_EvenSeriesComplex (f1_tmp, _Nder, 
            &_a[0], &_b[0], r2);
    }

    double f1_r             = creal(f1_tmp);
    double f1_i             = cimag(f1_tmp);

    f0_i[i]                 = cimag(f0_tmp);
    fX_i[i]                 = x_fX_r[i]*f1_i;
    fY_i[i]                 = y_fY_r[i]*f1_i;
    fZ_i[i]                 = z_fZ_r[i]*f1_i;

    f0_r[i]                 = creal(f0_tmp);
    x_fX_r[i]               = x_fX_r[i]*f1_r;
    y_fY_r[i]               = y_fY_r[i]*f1_r;
    z_fZ_r[i]               = z_fZ_r[i]*f1_r;
  }
}

#if NPME_USE_AVX
void NPME_Kfunc_Helmholtz_LR_DM::CalcAVX (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_LR_DM::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop          = N/4;
  const __m256d _k0rVec256    = _mm256_set1_pd( creal(_k0) );
  const __m256d _nk0iVec256   = _mm256_set1_pd(-cimag(_k0) );
  const __m256d _RdirVec256   = _mm256_set1_pd( _Rdir );

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm256_load_pd (&x_f0_r[count]);
    yVec  = _mm256_load_pd (&y[count]);
    zVec  = _mm256_load_pd (&z[count]);

    r2Vec = _mm256_mul_pd  (xVec, xVec);
    #if NPME_USE_AVX_FMA
    {
      r2Vec  = _mm256_fmadd_pd  (yVec, yVec, r2Vec);
      r2Vec  = _mm256_fmadd_pd  (zVec, zVec, r2Vec);
    }
    #else
    {
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (yVec, yVec), r2Vec);
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (zVec, zVec), r2Vec);
    }
    #endif
    rVec  = _mm256_sqrt_pd (r2Vec);



    __m256d f0rVec, f0iVec;
    {
      __m256d f0r_AVec, f0i_AVec;
      __m256d f0r_BVec, f0i_BVec;

      NPME_FunctionDerivMatch_EvenSeriesComplex_AVX (f0r_AVec, f0i_AVec, 
        r2Vec, _Nder, &_a[0]);
      NPME_KfuncHelmholtzRadial_AVX (f0r_BVec, f0i_BVec, rVec, 
        _k0rVec256, _nk0iVec256);

      //use (f0r_AVec, f0i_AVec) if r < Rdir
      //use (f0r_BVec, f0i_BVec) if r > Rdir
      {
        __m256d t0, dless, dmore;
        t0      = _mm256_cmp_pd (rVec, _RdirVec256, 1);

        dless   = _mm256_and_pd    (t0, f0r_AVec);
        dmore   = _mm256_andnot_pd (t0, f0r_BVec);
        f0rVec  = _mm256_add_pd (dless, dmore);

        dless   = _mm256_and_pd    (t0, f0i_AVec);
        dmore   = _mm256_andnot_pd (t0, f0i_BVec);
        f0iVec  = _mm256_add_pd (dless, dmore);
      }
    }
    _mm256_store_pd (&x_f0_r[count], f0rVec);
    _mm256_store_pd (  &f0_i[count], f0iVec);

    count += 4;
  }
}

void NPME_Kfunc_Helmholtz_LR_DM::CalcAVX (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop          = N/4;
  const __m256d _k0rVec256    = _mm256_set1_pd( creal(_k0) );
  const __m256d _nk0iVec256   = _mm256_set1_pd(-cimag(_k0) );
  const __m256d _RdirVec256   = _mm256_set1_pd( _Rdir );

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm256_load_pd (&x_fX_r[count]);
    yVec  = _mm256_load_pd (&y_fY_r[count]);
    zVec  = _mm256_load_pd (&z_fZ_r[count]);

    r2Vec = _mm256_mul_pd  (xVec, xVec);
    #if NPME_USE_AVX_FMA
    {
      r2Vec  = _mm256_fmadd_pd  (yVec, yVec, r2Vec);
      r2Vec  = _mm256_fmadd_pd  (zVec, zVec, r2Vec);
    }
    #else
    {
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (yVec, yVec), r2Vec);
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (zVec, zVec), r2Vec);
    }
    #endif
    rVec  = _mm256_sqrt_pd (r2Vec);

    __m256d f0rVec, f0iVec, f1rVec, f1iVec;
    __m256d fXrVec, fYrVec, fZrVec;
    __m256d fXiVec, fYiVec, fZiVec;

    {
      __m256d f0r_AVec, f0i_AVec;
      __m256d f0r_BVec, f0i_BVec;
      __m256d f1r_AVec, f1i_AVec;
      __m256d f1r_BVec, f1i_BVec;

      NPME_FunctionDerivMatch_EvenSeriesComplex_AVX (f0r_AVec, f0i_AVec, 
        f1r_AVec, f1i_AVec, r2Vec, _Nder, &_a[0], &_b[0]);
      NPME_KfuncHelmholtzRadial_AVX (f0r_BVec, f0i_BVec, 
        f1r_BVec, f1i_BVec, rVec, _k0rVec256, _nk0iVec256);

      //use (f0r_AVec, f0i_AVec) if r < Rdir
      //use (f0r_BVec, f0i_BVec) if r > Rdir
      {
        __m256d t0, dless, dmore;
        t0      = _mm256_cmp_pd (rVec, _RdirVec256, 1);

        dless   = _mm256_and_pd    (t0, f0r_AVec);
        dmore   = _mm256_andnot_pd (t0, f0r_BVec);
        f0rVec  = _mm256_add_pd (dless, dmore);

        dless   = _mm256_and_pd    (t0, f0i_AVec);
        dmore   = _mm256_andnot_pd (t0, f0i_BVec);
        f0iVec  = _mm256_add_pd (dless, dmore);

        dless   = _mm256_and_pd    (t0, f1r_AVec);
        dmore   = _mm256_andnot_pd (t0, f1r_BVec);
        f1rVec  = _mm256_add_pd (dless, dmore);

        dless   = _mm256_and_pd    (t0, f1i_AVec);
        dmore   = _mm256_andnot_pd (t0, f1i_BVec);
        f1iVec  = _mm256_add_pd (dless, dmore);
      }
    }


    fXrVec = _mm256_mul_pd  (f1rVec, xVec);
    fYrVec = _mm256_mul_pd  (f1rVec, yVec);
    fZrVec = _mm256_mul_pd  (f1rVec, zVec);

    fXiVec = _mm256_mul_pd  (f1iVec, xVec);
    fYiVec = _mm256_mul_pd  (f1iVec, yVec);
    fZiVec = _mm256_mul_pd  (f1iVec, zVec);

    _mm256_store_pd (  &f0_r[count], f0rVec);
    _mm256_store_pd (  &f0_i[count], f0iVec);

    _mm256_store_pd (&x_fX_r[count], fXrVec);
    _mm256_store_pd (&y_fY_r[count], fYrVec);
    _mm256_store_pd (&z_fZ_r[count], fZrVec);

    _mm256_store_pd (  &fX_i[count], fXiVec);
    _mm256_store_pd (  &fY_i[count], fYiVec);
    _mm256_store_pd (  &fZ_i[count], fZiVec);

    count += 4;
  }
}


#endif



#if NPME_USE_AVX_512
void NPME_Kfunc_Helmholtz_LR_DM::CalcAVX_512 (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_LR_DM::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop          = N/8;
  const __m512d _k0rVec512    = _mm512_set1_pd( creal(_k0) );
  const __m512d _nk0iVec512   = _mm512_set1_pd(-cimag(_k0) );
  const __m512d _RdirVec512   = _mm512_set1_pd( _Rdir );

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm512_load_pd (&x_f0_r[count]);
    yVec  = _mm512_load_pd (&y[count]);
    zVec  = _mm512_load_pd (&z[count]);

    r2Vec = _mm512_mul_pd  (xVec, xVec);
    r2Vec = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm512_fmadd_pd  (zVec, zVec, r2Vec);

    rVec  = _mm512_sqrt_pd (r2Vec);

    __m512d f0rVec, f0iVec;
    {

      __m512d f0r_AVec, f0i_AVec;
      __m512d f0r_BVec, f0i_BVec;

      NPME_FunctionDerivMatch_EvenSeriesComplex_AVX_512 (f0r_AVec, f0i_AVec, 
        r2Vec, _Nder, &_a[0]);
      NPME_KfuncHelmholtzRadial_AVX_512 (f0r_BVec, f0i_BVec, rVec, 
        _k0rVec512, _nk0iVec512);

      //use (f0r_AVec, f0i_AVec) if r < Rdir
      //use (f0r_BVec, f0i_BVec) if r > Rdir
      {
        __mmask8 maskVec = _mm512_cmp_pd_mask (rVec, _RdirVec512, 1);
        f0rVec = _mm512_mask_mov_pd (f0r_BVec, maskVec, f0r_AVec);
        f0iVec = _mm512_mask_mov_pd (f0i_BVec, maskVec, f0i_AVec);
      }
    }

    _mm512_store_pd (&x_f0_r[count], f0rVec);
    _mm512_store_pd (  &f0_i[count], f0iVec);

    count += 8;
  }
}

void NPME_Kfunc_Helmholtz_LR_DM::CalcAVX_512 (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_LR_DM::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop          = N/8;
  const __m512d _k0rVec512    = _mm512_set1_pd( creal(_k0) );
  const __m512d _nk0iVec512   = _mm512_set1_pd(-cimag(_k0) );
  const __m512d _RdirVec512   = _mm512_set1_pd( _Rdir );

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm512_load_pd (&x_fX_r[count]);
    yVec  = _mm512_load_pd (&y_fY_r[count]);
    zVec  = _mm512_load_pd (&z_fZ_r[count]);

    r2Vec = _mm512_mul_pd  (xVec, xVec);
    r2Vec = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm512_fmadd_pd  (zVec, zVec, r2Vec);
    rVec  = _mm512_sqrt_pd (r2Vec);


    __m512d f0rVec, f0iVec, f1rVec, f1iVec;
    {
      __m512d f0r_AVec, f0i_AVec;
      __m512d f0r_BVec, f0i_BVec;
      __m512d f1r_AVec, f1i_AVec;
      __m512d f1r_BVec, f1i_BVec;

      NPME_FunctionDerivMatch_EvenSeriesComplex_AVX_512 (f0r_AVec, f0i_AVec, 
        f1r_AVec, f1i_AVec, r2Vec, _Nder, &_a[0], &_b[0]);
      NPME_KfuncHelmholtzRadial_AVX_512 (f0r_BVec, f0i_BVec, 
        f1r_BVec, f1i_BVec, rVec, _k0rVec512, _nk0iVec512);

      //use (f0r_AVec, f0i_AVec) if r < Rdir
      //use (f0r_BVec, f0i_BVec) if r > Rdir
      {
        __mmask8 maskVec = _mm512_cmp_pd_mask (rVec, _RdirVec512, 1);
        f0rVec = _mm512_mask_mov_pd (f0r_BVec, maskVec, f0r_AVec);
        f0iVec = _mm512_mask_mov_pd (f0i_BVec, maskVec, f0i_AVec);
        f1rVec = _mm512_mask_mov_pd (f1r_BVec, maskVec, f1r_AVec);
        f1iVec = _mm512_mask_mov_pd (f1i_BVec, maskVec, f1i_AVec);
      }
    }

    __m512d fXrVec, fYrVec, fZrVec;
    __m512d fXiVec, fYiVec, fZiVec;
    fXrVec = _mm512_mul_pd  (f1rVec, xVec);
    fYrVec = _mm512_mul_pd  (f1rVec, yVec);
    fZrVec = _mm512_mul_pd  (f1rVec, zVec);

    fXiVec = _mm512_mul_pd  (f1iVec, xVec);
    fYiVec = _mm512_mul_pd  (f1iVec, yVec);
    fZiVec = _mm512_mul_pd  (f1iVec, zVec);

    _mm512_store_pd (  &f0_r[count], f0rVec);
    _mm512_store_pd (  &f0_i[count], f0iVec);

    _mm512_store_pd (&x_fX_r[count], fXrVec);
    _mm512_store_pd (&y_fY_r[count], fYrVec);
    _mm512_store_pd (&z_fZ_r[count], fZrVec);

    _mm512_store_pd (  &fX_i[count], fXiVec);
    _mm512_store_pd (  &fY_i[count], fYiVec);
    _mm512_store_pd (  &fZ_i[count], fZiVec);

    count += 8;
  }
}


#endif


//******************************************************************************
//******************************************************************************
//***********************NPME_Kfunc_Helmholtz_SR_DM*****************************
//******************************************************************************
//******************************************************************************



bool NPME_Kfunc_Helmholtz_SR_DM::SetParm (const _Complex double k0, 
  const int Nder, const double Rdir, bool PRINT, std::ostream& os)
{
  if (Nder > NPME_MaxDerivMatchOrder)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_SR_DM::SetParm\n";
    char str[2000];
    sprintf(str, "  Nder = %d > %d = NPME_MaxDerivMatchOrder\n", Nder,
      NPME_MaxDerivMatchOrder);
    std::cout << str;
    return false;
  }

  _k0   = k0;
  _Nder = Nder;
  _Rdir = Rdir;


  std::vector<_Complex double> fHelm(_Nder+1);
  NPME_FunctionDerivMatch_HelmholtzRadialDeriv (&fHelm[0], _Nder, _Rdir, _k0);
  if (!NPME_FunctionDerivMatch_CalcEvenSeries (&_a[0], &_b[0], 
    &fHelm[0], _Nder, _Rdir))
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_SR_DM::SetParm\n";
    std::cout << "NPME_FunctionDerivMatch_CalcEvenSeries failed\n";
    return false;
  }

  if (PRINT)
  {
    char str[2000];
    os << "\n\nNPME_Kfunc_Helmholtz_SR_DM::SetParm\n";
    sprintf(str, "   k0    = %10.6f + %10.6fi\n", creal(_k0), cimag(_k0));
    os << str;
    sprintf(str, "   Rdir  = %10.6f\n", _Rdir);
    os << str;
    sprintf(str, "   Nder  = %d\n", _Nder);
    os << str;
    os << "\n";
    for (int i = 0; i <= _Nder; i++)
    {
      sprintf(str, "      a[%4d] = %25.15le + %25.15lei\n", 
        i, creal(_a[i]), cimag(_a[i]));
      os << str;
    }
    os << "\n";
    for (int i = 0; i <= _Nder; i++)
    {
      sprintf(str, "      b[%4d] = %25.15le + %25.15lei\n", 
        i, creal(_b[i]), cimag(_b[i]));
      os << str;
    }

  }

  return true;
}

void NPME_Kfunc_Helmholtz_SR_DM::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Helmholtz_SR_DM::Print\n";
  sprintf(str, "  k0    = %.3f + I*%.3f\n", creal(_k0), cimag(_k0));
  os << str;
  sprintf(str, "  Rdir  = %.3f\n", _Rdir);    os << str;
  sprintf(str, "  Nder  = %3d\n", _Nder);     os << str;
  os.flush();
}

void NPME_Kfunc_Helmholtz_SR_DM::Calc (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_f0_r[i]*x_f0_r[i] + y[i]*y[i] + z[i]*z[i];
    const double r  = sqrt(fabs(r2));

    _Complex double f0_tmp;

    if (r < _Rdir)
    {
      f0_tmp = cexp(I*_k0*r)/r -
          NPME_FunctionDerivMatch_EvenSeriesComplex (_Nder, &_a[0], r2);
    }
    else
      f0_tmp = 0;

    x_f0_r[i]               = creal(f0_tmp);
    f0_i[i]                 = cimag(f0_tmp);
  }
}
void NPME_Kfunc_Helmholtz_SR_DM::Calc (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX_r[i]*x_fX_r[i] + 
                      y_fY_r[i]*y_fY_r[i] + 
                      z_fZ_r[i]*z_fZ_r[i];
    const double r  = sqrt(fabs(r2));

    _Complex double f0_tmp, f1_tmp;
    if (r < _Rdir)
    {
      _Complex double f0_exact  = cexp(I*_k0*r)/r;
      _Complex double f1_exact  = (I*_k0*f0_exact - f0_exact/r)/r;

      f0_tmp = f0_exact - NPME_FunctionDerivMatch_EvenSeriesComplex (f1_tmp, 
                            _Nder, &_a[0], &_b[0], r2);
      f1_tmp = f1_exact - f1_tmp;
    }
    else
    {
      f0_tmp = 0;
      f1_tmp = 0;
    }

    double f1_r             = creal(f1_tmp);
    double f1_i             = cimag(f1_tmp);

    f0_i[i]                 = cimag(f0_tmp);
    fX_i[i]                 = x_fX_r[i]*f1_i;
    fY_i[i]                 = y_fY_r[i]*f1_i;
    fZ_i[i]                 = z_fZ_r[i]*f1_i;

    f0_r[i]                 = creal(f0_tmp);
    x_fX_r[i]               = x_fX_r[i]*f1_r;
    y_fY_r[i]               = y_fY_r[i]*f1_r;
    z_fZ_r[i]               = z_fZ_r[i]*f1_r;
  }
}


#if NPME_USE_AVX
void NPME_Kfunc_Helmholtz_SR_DM::CalcAVX (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const
{
  const __m256d zeroVec = _mm256_set1_pd(0.0);

  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_SR_DM::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop          = N/4;
  const __m256d _k0rVec256    = _mm256_set1_pd( creal(_k0) );
  const __m256d _nk0iVec256   = _mm256_set1_pd(-cimag(_k0) );
  const __m256d _RdirVec256   = _mm256_set1_pd( _Rdir );

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm256_load_pd (&x_f0_r[count]);
    yVec  = _mm256_load_pd (&y[count]);
    zVec  = _mm256_load_pd (&z[count]);

    r2Vec = _mm256_mul_pd  (xVec, xVec);
    #if NPME_USE_AVX_FMA
    {
      r2Vec  = _mm256_fmadd_pd  (yVec, yVec, r2Vec);
      r2Vec  = _mm256_fmadd_pd  (zVec, zVec, r2Vec);
    }
    #else
    {
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (yVec, yVec), r2Vec);
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (zVec, zVec), r2Vec);
    }
    #endif
    rVec  = _mm256_sqrt_pd (r2Vec);

    __m256d f0rVec, f0iVec;
    {
      __m256d f0r_AVec, f0i_AVec;
      __m256d f0r_BVec, f0i_BVec;

      NPME_FunctionDerivMatch_EvenSeriesComplex_AVX (f0r_AVec, f0i_AVec, 
        r2Vec, _Nder, &_a[0]);
      NPME_KfuncHelmholtzRadial_AVX (f0r_BVec, f0i_BVec, rVec, 
        _k0rVec256, _nk0iVec256);

      f0r_BVec = _mm256_sub_pd (f0r_BVec, f0r_AVec);
      f0i_BVec = _mm256_sub_pd (f0i_BVec, f0i_AVec);

      //use (f0r_BVec, f0i_BVec) if r < Rdir
      //use (zeroVec,  zeroVec)  if r > Rdir
      {
        __m256d t0, dless, dmore;
        t0      = _mm256_cmp_pd (rVec, _RdirVec256, 1);

        dless   = _mm256_and_pd    (t0, f0r_BVec);
        dmore   = _mm256_andnot_pd (t0, zeroVec);
        f0rVec  = _mm256_add_pd (dless, dmore);

        dless   = _mm256_and_pd    (t0, f0i_BVec);
        dmore   = _mm256_andnot_pd (t0, zeroVec);
        f0iVec  = _mm256_add_pd (dless, dmore);
      }
    }

    _mm256_store_pd (&x_f0_r[count], f0rVec);
    _mm256_store_pd (  &f0_i[count], f0iVec);

    count += 4;
  }
}

void NPME_Kfunc_Helmholtz_SR_DM::CalcAVX (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const
{
  const __m256d zeroVec = _mm256_set1_pd(0.0);

  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop          = N/4;
  const __m256d _k0rVec256    = _mm256_set1_pd( creal(_k0) );
  const __m256d _nk0iVec256   = _mm256_set1_pd(-cimag(_k0) );
  const __m256d _RdirVec256   = _mm256_set1_pd( _Rdir );

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm256_load_pd (&x_fX_r[count]);
    yVec  = _mm256_load_pd (&y_fY_r[count]);
    zVec  = _mm256_load_pd (&z_fZ_r[count]);

    r2Vec = _mm256_mul_pd  (xVec, xVec);
    #if NPME_USE_AVX_FMA
    {
      r2Vec  = _mm256_fmadd_pd  (yVec, yVec, r2Vec);
      r2Vec  = _mm256_fmadd_pd  (zVec, zVec, r2Vec);
    }
    #else
    {
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (yVec, yVec), r2Vec);
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (zVec, zVec), r2Vec);
    }
    #endif
    rVec  = _mm256_sqrt_pd (r2Vec);

    __m256d f0rVec, f0iVec, f1rVec, f1iVec;
    __m256d fXrVec, fYrVec, fZrVec;
    __m256d fXiVec, fYiVec, fZiVec;

    {
      __m256d f0r_AVec, f0i_AVec;
      __m256d f0r_BVec, f0i_BVec;
      __m256d f1r_AVec, f1i_AVec;
      __m256d f1r_BVec, f1i_BVec;

      NPME_FunctionDerivMatch_EvenSeriesComplex_AVX (f0r_AVec, f0i_AVec, 
        f1r_AVec, f1i_AVec, r2Vec, _Nder, &_a[0], &_b[0]);
      NPME_KfuncHelmholtzRadial_AVX (f0r_BVec, f0i_BVec, 
        f1r_BVec, f1i_BVec, rVec, _k0rVec256, _nk0iVec256);

      f0r_BVec = _mm256_sub_pd (f0r_BVec, f0r_AVec);
      f0i_BVec = _mm256_sub_pd (f0i_BVec, f0i_AVec);
      f1r_BVec = _mm256_sub_pd (f1r_BVec, f1r_AVec);
      f1i_BVec = _mm256_sub_pd (f1i_BVec, f1i_AVec);

      //use (f0r_BVec, f0i_BVec) if r < Rdir
      //use (zeroVec,  zeroVec)  if r > Rdir
      {
        __m256d t0, dless, dmore;
        t0      = _mm256_cmp_pd (rVec, _RdirVec256, 1);

        dless   = _mm256_and_pd    (t0, f0r_BVec);
        dmore   = _mm256_andnot_pd (t0, zeroVec);
        f0rVec  = _mm256_add_pd (dless, dmore);

        dless   = _mm256_and_pd    (t0, f0i_BVec);
        dmore   = _mm256_andnot_pd (t0, zeroVec);
        f0iVec  = _mm256_add_pd (dless, dmore);

        dless   = _mm256_and_pd    (t0, f1r_BVec);
        dmore   = _mm256_andnot_pd (t0, zeroVec);
        f1rVec  = _mm256_add_pd (dless, dmore);

        dless   = _mm256_and_pd    (t0, f1i_BVec);
        dmore   = _mm256_andnot_pd (t0, zeroVec);
        f1iVec  = _mm256_add_pd (dless, dmore);
      }
    }


    fXrVec = _mm256_mul_pd  (f1rVec, xVec);
    fYrVec = _mm256_mul_pd  (f1rVec, yVec);
    fZrVec = _mm256_mul_pd  (f1rVec, zVec);

    fXiVec = _mm256_mul_pd  (f1iVec, xVec);
    fYiVec = _mm256_mul_pd  (f1iVec, yVec);
    fZiVec = _mm256_mul_pd  (f1iVec, zVec);

    _mm256_store_pd (  &f0_r[count], f0rVec);
    _mm256_store_pd (  &f0_i[count], f0iVec);

    _mm256_store_pd (&x_fX_r[count], fXrVec);
    _mm256_store_pd (&y_fY_r[count], fYrVec);
    _mm256_store_pd (&z_fZ_r[count], fZrVec);

    _mm256_store_pd (  &fX_i[count], fXiVec);
    _mm256_store_pd (  &fY_i[count], fYiVec);
    _mm256_store_pd (  &fZ_i[count], fZiVec);

    count += 4;
  }
}


#endif


#if NPME_USE_AVX_512
void NPME_Kfunc_Helmholtz_SR_DM::CalcAVX_512 (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const
{
  const __m512d zeroVec = _mm512_set1_pd(0.0);

  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_SR_DM::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop          = N/8;
  const __m512d _k0rVec512    = _mm512_set1_pd( creal(_k0) );
  const __m512d _nk0iVec512   = _mm512_set1_pd(-cimag(_k0) );
  const __m512d _RdirVec512   = _mm512_set1_pd( _Rdir );

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm512_load_pd (&x_f0_r[count]);
    yVec  = _mm512_load_pd (&y[count]);
    zVec  = _mm512_load_pd (&z[count]);

    r2Vec = _mm512_mul_pd  (xVec, xVec);
    r2Vec = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm512_fmadd_pd  (zVec, zVec, r2Vec);
    rVec  = _mm512_sqrt_pd (r2Vec);

    __m512d f0rVec, f0iVec;
    {
      __m512d f0r_AVec, f0i_AVec;
      __m512d f0r_BVec, f0i_BVec;

      NPME_FunctionDerivMatch_EvenSeriesComplex_AVX_512 (f0r_AVec, f0i_AVec, 
        r2Vec, _Nder, &_a[0]);
      NPME_KfuncHelmholtzRadial_AVX_512 (f0r_BVec, f0i_BVec, rVec, 
        _k0rVec512, _nk0iVec512);

      f0r_BVec = _mm512_sub_pd (f0r_BVec, f0r_AVec);
      f0i_BVec = _mm512_sub_pd (f0i_BVec, f0i_AVec);


      //use (f0_BVec) if r < Rdir
      //use (zeroVec) if r > Rdir
      {
        __mmask8 maskVec = _mm512_cmp_pd_mask (rVec, _RdirVec512, 1);
        f0rVec = _mm512_mask_mov_pd (zeroVec, maskVec, f0r_BVec);
        f0iVec = _mm512_mask_mov_pd (zeroVec, maskVec, f0i_BVec);
      }
    }

    _mm512_store_pd (&x_f0_r[count], f0rVec);
    _mm512_store_pd (  &f0_i[count], f0iVec);

    count += 8;
  }
}

void NPME_Kfunc_Helmholtz_SR_DM::CalcAVX_512 (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const
{
  const __m512d zeroVec = _mm512_set1_pd(0.0);

  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop          = N/8;
  const __m512d _k0rVec512    = _mm512_set1_pd( creal(_k0) );
  const __m512d _nk0iVec512   = _mm512_set1_pd(-cimag(_k0) );
  const __m512d _RdirVec512   = _mm512_set1_pd( _Rdir );

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm512_load_pd (&x_fX_r[count]);
    yVec  = _mm512_load_pd (&y_fY_r[count]);
    zVec  = _mm512_load_pd (&z_fZ_r[count]);

    r2Vec = _mm512_mul_pd  (xVec, xVec);
    r2Vec = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm512_fmadd_pd  (zVec, zVec, r2Vec);
    rVec  = _mm512_sqrt_pd (r2Vec);

    __m512d f0rVec, f0iVec, f1rVec, f1iVec;
    __m512d fXrVec, fYrVec, fZrVec;
    __m512d fXiVec, fYiVec, fZiVec;

    {
      __m512d f0r_AVec, f0i_AVec;
      __m512d f0r_BVec, f0i_BVec;
      __m512d f1r_AVec, f1i_AVec;
      __m512d f1r_BVec, f1i_BVec;

      NPME_FunctionDerivMatch_EvenSeriesComplex_AVX_512 (f0r_AVec, f0i_AVec, 
        f1r_AVec, f1i_AVec, r2Vec, _Nder, &_a[0], &_b[0]);
      NPME_KfuncHelmholtzRadial_AVX_512 (f0r_BVec, f0i_BVec, 
        f1r_BVec, f1i_BVec, rVec, _k0rVec512, _nk0iVec512);

      f0r_BVec = _mm512_sub_pd (f0r_BVec, f0r_AVec);
      f0i_BVec = _mm512_sub_pd (f0i_BVec, f0i_AVec);
      f1r_BVec = _mm512_sub_pd (f1r_BVec, f1r_AVec);
      f1i_BVec = _mm512_sub_pd (f1i_BVec, f1i_AVec);

      //use (f0_BVec) if r < Rdir
      //use (zeroVec) if r > Rdir
      {
        __mmask8 maskVec = _mm512_cmp_pd_mask (rVec, _RdirVec512, 1);
        f0rVec = _mm512_mask_mov_pd (zeroVec, maskVec, f0r_BVec);
        f0iVec = _mm512_mask_mov_pd (zeroVec, maskVec, f0i_BVec);
        f1rVec = _mm512_mask_mov_pd (zeroVec, maskVec, f1r_BVec);
        f1iVec = _mm512_mask_mov_pd (zeroVec, maskVec, f1i_BVec);
      }
    }

    fXrVec = _mm512_mul_pd  (f1rVec, xVec);
    fYrVec = _mm512_mul_pd  (f1rVec, yVec);
    fZrVec = _mm512_mul_pd  (f1rVec, zVec);

    fXiVec = _mm512_mul_pd  (f1iVec, xVec);
    fYiVec = _mm512_mul_pd  (f1iVec, yVec);
    fZiVec = _mm512_mul_pd  (f1iVec, zVec);

    _mm512_store_pd (  &f0_r[count], f0rVec);
    _mm512_store_pd (  &f0_i[count], f0iVec);

    _mm512_store_pd (&x_fX_r[count], fXrVec);
    _mm512_store_pd (&y_fY_r[count], fYrVec);
    _mm512_store_pd (&z_fZ_r[count], fZrVec);

    _mm512_store_pd (  &fX_i[count], fXiVec);
    _mm512_store_pd (  &fY_i[count], fYiVec);
    _mm512_store_pd (  &fZ_i[count], fZiVec);

    count += 8;
  }
}


#endif





//******************************************************************************
//******************************************************************************
//***********************NPME_Kfunc_Helmholtz_LR_Alt****************************
//******************************************************************************
//******************************************************************************

bool NPME_Kfunc_Helmholtz_LR_Alt::SetParm (const double k0, const double beta)
{
  _k0       = k0;
  _beta     = beta;
  _beta3    = beta*beta*beta;
  _k03      = k0*k0*k0;
  _k02_beta = k0*k0*_beta;

  return true;
}

void NPME_Kfunc_Helmholtz_LR_Alt::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Helmholtz_LR_DM::Print\n";
  sprintf(str, "  k0    = %.3f (real only)\n", _k0);  os << str;
  sprintf(str, "  beta  = %.3f\n", _beta);            os << str;
  os.flush();
}

void NPME_Kfunc_Helmholtz_LR_Alt::Calc (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_f0_r[i]*x_f0_r[i] + y[i]*y[i] + z[i]*z[i];
    const double r  = sqrt(fabs(r2));
    const double v  = r*_k0;

    x_f0_r[i]     = _beta*NPME_Berf_0 (r*_beta)*cos(v);
    f0_i[i]       = _k0*NPME_sinx_x (v);
  }
}
void NPME_Kfunc_Helmholtz_LR_Alt::Calc (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX_r[i]*x_fX_r[i] + 
                      y_fY_r[i]*y_fY_r[i] + 
                      z_fZ_r[i]*z_fZ_r[i];
    const double r  = sqrt(fabs(r2));

    double B0, B1;
    B0 = NPME_Berf_1 (B1, _beta*r);

    double C0, C1;
    C0 = NPME_sinx_x (C1, _k0*r);

    double v      = r*_k0;
    double cos_v  = cos(v);

    f0_r[i]       = _beta*B0*cos_v;
    f0_i[i]       = _k0*NPME_sinx_x (v);

    double f1_r   = _beta3*B1*cos_v - _k02_beta*B0*C0;
    double f1_i   = _k03*C1;

    fX_i[i]                 = x_fX_r[i]*f1_i;
    fY_i[i]                 = y_fY_r[i]*f1_i;
    fZ_i[i]                 = z_fZ_r[i]*f1_i;

    x_fX_r[i]               = x_fX_r[i]*f1_r;
    y_fY_r[i]               = y_fY_r[i]*f1_r;
    z_fZ_r[i]               = z_fZ_r[i]*f1_r;
  }
}

#if NPME_USE_AVX
void NPME_Kfunc_Helmholtz_LR_Alt::CalcAVX (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const
{
  const __m256d zeroVec = _mm256_set1_pd(0.0);

  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_LR_Alt::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop              = N/4;
  const __m256d _k0_Vec256        = _mm256_set1_pd(_k0);
  const __m256d _k03_Vec256       = _mm256_set1_pd(_k03);
  const __m256d _k02_beta_Vec256  = _mm256_set1_pd(_k02_beta);
  const __m256d _beta_Vec256      = _mm256_set1_pd(_beta);
  const __m256d _beta3_Vec256     = _mm256_set1_pd(_beta3);

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm256_load_pd (&x_f0_r[count]);
    yVec  = _mm256_load_pd (&y[count]);
    zVec  = _mm256_load_pd (&z[count]);

    r2Vec = _mm256_mul_pd  (xVec, xVec);
    #if NPME_USE_AVX_FMA
    {
      r2Vec  = _mm256_fmadd_pd  (yVec, yVec, r2Vec);
      r2Vec  = _mm256_fmadd_pd  (zVec, zVec, r2Vec);
    }
    #else
    {
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (yVec, yVec), r2Vec);
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (zVec, zVec), r2Vec);
    }
    #endif
    rVec  = _mm256_sqrt_pd (r2Vec);

    __m256d f0rVec, f0iVec;
    {
      __m256d B0_vec;
      NPME_Berf_0_AVX (B0_vec, _mm256_mul_pd  (_beta_Vec256, rVec));

      __m256d C0_vec, cos_vec;
      NPME_sinx_x_AVX (C0_vec, cos_vec, _mm256_mul_pd  (_k0_Vec256, rVec));

      f0rVec = _mm256_mul_pd  (_beta_Vec256, B0_vec);
      f0rVec = _mm256_mul_pd  (cos_vec,      f0rVec);

      f0iVec = _mm256_mul_pd  (_k0_Vec256,   C0_vec);
    }

    _mm256_store_pd (&x_f0_r[count], f0rVec);
    _mm256_store_pd (  &f0_i[count], f0iVec);

    count += 4;
  }
}

void NPME_Kfunc_Helmholtz_LR_Alt::CalcAVX (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const
{
  const __m256d zeroVec = _mm256_set1_pd(0.0);

  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_LR_Alt::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop              = N/4;
  const __m256d _k0_Vec256        = _mm256_set1_pd(_k0);
  const __m256d _k03_Vec256       = _mm256_set1_pd(_k03);
  const __m256d _k02_beta_Vec256  = _mm256_set1_pd(_k02_beta);
  const __m256d _beta_Vec256      = _mm256_set1_pd(_beta);
  const __m256d _beta3_Vec256     = _mm256_set1_pd(_beta3);

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm256_load_pd (&x_fX_r[count]);
    yVec  = _mm256_load_pd (&y_fY_r[count]);
    zVec  = _mm256_load_pd (&z_fZ_r[count]);

    r2Vec = _mm256_mul_pd  (xVec, xVec);
    #if NPME_USE_AVX_FMA
    {
      r2Vec  = _mm256_fmadd_pd  (yVec, yVec, r2Vec);
      r2Vec  = _mm256_fmadd_pd  (zVec, zVec, r2Vec);
    }
    #else
    {
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (yVec, yVec), r2Vec);
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (zVec, zVec), r2Vec);
    }
    #endif
    rVec  = _mm256_sqrt_pd (r2Vec);

    __m256d f0rVec, f0iVec, f1rVec, f1iVec;
    __m256d fXrVec, fYrVec, fZrVec;
    __m256d fXiVec, fYiVec, fZiVec;

    {
      __m256d B0_vec, B1_vec;
      NPME_Berf_1_AVX (B0_vec, B1_vec, _mm256_mul_pd  (_beta_Vec256, rVec));

      __m256d C0_vec, C1_vec, cos_vec;
      NPME_sinx_x_AVX (C0_vec, C1_vec, cos_vec, _mm256_mul_pd  (_k0_Vec256, rVec));

      //f0_r = _beta*B0*cos(v);
      //f0_i = _k0*NPME_sinx_x (v);
      f0rVec        = _mm256_mul_pd  (_beta_Vec256, B0_vec);
      f0rVec        = _mm256_mul_pd  (cos_vec,      f0rVec);
      f0iVec        = _mm256_mul_pd  (_k0_Vec256,   C0_vec);

      //f1_r = _beta3*B1*cos(v) - _k02_beta*B0*C0;
      //f1_i = _k03*C1
      __m256d B1_beta3_vec;
      B1_beta3_vec  = _mm256_mul_pd  (_beta3_Vec256, B1_vec);
      f1rVec        = _mm256_mul_pd  (_k02_beta_Vec256, B0_vec);
      f1rVec        = _mm256_mul_pd  (C0_vec,           f1rVec);

      #if NPME_USE_AVX_FMA
      {
        f1rVec = _mm256_fmsub_pd  (B1_beta3_vec, cos_vec, f1rVec);
      }
      #else
      {
        f1rVec = _mm256_sub_pd  (_mm256_mul_pd (B1_beta3_vec, cos_vec), f1rVec);
      }
      #endif

      f1iVec = _mm256_mul_pd  (_k03_Vec256, C1_vec);
    }


    fXrVec = _mm256_mul_pd  (f1rVec, xVec);
    fYrVec = _mm256_mul_pd  (f1rVec, yVec);
    fZrVec = _mm256_mul_pd  (f1rVec, zVec);

    fXiVec = _mm256_mul_pd  (f1iVec, xVec);
    fYiVec = _mm256_mul_pd  (f1iVec, yVec);
    fZiVec = _mm256_mul_pd  (f1iVec, zVec);

    _mm256_store_pd (  &f0_r[count], f0rVec);
    _mm256_store_pd (  &f0_i[count], f0iVec);

    _mm256_store_pd (&x_fX_r[count], fXrVec);
    _mm256_store_pd (&y_fY_r[count], fYrVec);
    _mm256_store_pd (&z_fZ_r[count], fZrVec);

    _mm256_store_pd (  &fX_i[count], fXiVec);
    _mm256_store_pd (  &fY_i[count], fYiVec);
    _mm256_store_pd (  &fZ_i[count], fZiVec);

    count += 4;
  }
}


#endif

#if NPME_USE_AVX_512
void NPME_Kfunc_Helmholtz_LR_Alt::CalcAVX_512 (const size_t N, 
    double *x_f0_r, double *f0_i, const double *y, const double *z) const
{
  const __m512d zeroVec = _mm512_set1_pd(0.0);

  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_LR_Alt::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop              = N/8;
  const __m512d _k0_Vec512        = _mm512_set1_pd(_k0);
  const __m512d _k03_Vec512       = _mm512_set1_pd(_k03);
  const __m512d _k02_beta_Vec512  = _mm512_set1_pd(_k02_beta);
  const __m512d _beta_Vec512      = _mm512_set1_pd(_beta);
  const __m512d _beta3_Vec512     = _mm512_set1_pd(_beta3);

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm512_load_pd (&x_f0_r[count]);
    yVec  = _mm512_load_pd (&y[count]);
    zVec  = _mm512_load_pd (&z[count]);

    r2Vec = _mm512_mul_pd  (xVec, xVec);
    r2Vec = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm512_fmadd_pd  (zVec, zVec, r2Vec);
    rVec  = _mm512_sqrt_pd (r2Vec);

    __m512d f0rVec, f0iVec;
    {
      __m512d B0_vec;
      NPME_Berf_0_AVX_512 (B0_vec, _mm512_mul_pd  (_beta_Vec512, rVec));

      __m512d C0_vec, cos_vec;
      NPME_sinx_x_AVX_512 (C0_vec, cos_vec, _mm512_mul_pd  (_k0_Vec512, rVec));

      f0rVec = _mm512_mul_pd  (_beta_Vec512, B0_vec);
      f0rVec = _mm512_mul_pd  (cos_vec,      f0rVec);

      f0iVec = _mm512_mul_pd  (_k0_Vec512,   C0_vec);
    }

    _mm512_store_pd (&x_f0_r[count], f0rVec);
    _mm512_store_pd (  &f0_i[count], f0iVec);

    count += 8;
  }
}

void NPME_Kfunc_Helmholtz_LR_Alt::CalcAVX_512 (const size_t N, 
    double *f0_r,   double *f0_i, 
    double *x_fX_r, double *fX_i,
    double *y_fY_r, double *fY_i,
    double *z_fZ_r, double *fZ_i) const
{
  const __m512d zeroVec = _mm512_set1_pd(0.0);

  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_LR_Alt::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop              = N/8;
  const __m512d _k0_Vec512        = _mm512_set1_pd(_k0);
  const __m512d _k03_Vec512       = _mm512_set1_pd(_k03);
  const __m512d _k02_beta_Vec512  = _mm512_set1_pd(_k02_beta);
  const __m512d _beta_Vec512      = _mm512_set1_pd(_beta);
  const __m512d _beta3_Vec512     = _mm512_set1_pd(_beta3);

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm512_load_pd (&x_fX_r[count]);
    yVec  = _mm512_load_pd (&y_fY_r[count]);
    zVec  = _mm512_load_pd (&z_fZ_r[count]);

    r2Vec = _mm512_mul_pd  (xVec, xVec);
    r2Vec = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm512_fmadd_pd  (zVec, zVec, r2Vec);
    rVec  = _mm512_sqrt_pd (r2Vec);

    __m512d f0rVec, f0iVec, f1rVec, f1iVec;
    __m512d fXrVec, fYrVec, fZrVec;
    __m512d fXiVec, fYiVec, fZiVec;

    {
      __m512d B0_vec, B1_vec;
      NPME_Berf_1_AVX_512 (B0_vec, B1_vec, _mm512_mul_pd  (_beta_Vec512, rVec));

      __m512d C0_vec, C1_vec, cos_vec;
      NPME_sinx_x_AVX_512 (C0_vec, C1_vec, cos_vec, _mm512_mul_pd  (_k0_Vec512, rVec));

      //f0_r = _beta*B0*cos(v);
      //f0_i = _k0*NPME_sinx_x (v);
      f0rVec        = _mm512_mul_pd  (_beta_Vec512, B0_vec);
      f0rVec        = _mm512_mul_pd  (cos_vec,      f0rVec);
      f0iVec        = _mm512_mul_pd  (_k0_Vec512,   C0_vec);

      //f1_r = _beta3*B1*cos(v) - _k02_beta*B0*C0;
      //f1_i = _k03*C1
      __m512d B1_beta3_vec;
      B1_beta3_vec  = _mm512_mul_pd  (_beta3_Vec512, B1_vec);
      f1rVec        = _mm512_mul_pd  (_k02_beta_Vec512, B0_vec);
      f1rVec        = _mm512_mul_pd  (C0_vec,           f1rVec);

      #if NPME_USE_AVX_512_FMA
      {
        f1rVec = _mm512_fmsub_pd  (B1_beta3_vec, cos_vec, f1rVec);
      }
      #else
      {
        f1rVec = _mm512_sub_pd  (_mm512_mul_pd (B1_beta3_vec, cos_vec), f1rVec);
      }
      #endif

      f1iVec = _mm512_mul_pd  (_k03_Vec512, C1_vec);
    }


    fXrVec = _mm512_mul_pd  (f1rVec, xVec);
    fYrVec = _mm512_mul_pd  (f1rVec, yVec);
    fZrVec = _mm512_mul_pd  (f1rVec, zVec);

    fXiVec = _mm512_mul_pd  (f1iVec, xVec);
    fYiVec = _mm512_mul_pd  (f1iVec, yVec);
    fZiVec = _mm512_mul_pd  (f1iVec, zVec);

    _mm512_store_pd (  &f0_r[count], f0rVec);
    _mm512_store_pd (  &f0_i[count], f0iVec);

    _mm512_store_pd (&x_fX_r[count], fXrVec);
    _mm512_store_pd (&y_fY_r[count], fYrVec);
    _mm512_store_pd (&z_fZ_r[count], fZrVec);

    _mm512_store_pd (  &fX_i[count], fXiVec);
    _mm512_store_pd (  &fY_i[count], fYiVec);
    _mm512_store_pd (  &fZ_i[count], fZiVec);

    count += 8;
  }
}


#endif
//******************************************************************************
//******************************************************************************
//***********************NPME_Kfunc_Helmholtz_SR_Alt****************************
//******************************************************************************
//******************************************************************************

bool NPME_Kfunc_Helmholtz_SR_Alt::SetParm (const double k0, const double beta)
{
  _k0       = k0;
  _beta     = beta;
  _beta3    = beta*beta*beta;
  _k03      = k0*k0*k0;
  _k02_beta = k0*k0*_beta;

  return true;
}

void NPME_Kfunc_Helmholtz_SR_Alt::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Helmholtz_SR_Alt::Print\n";
  sprintf(str, "  beta  = %.3f\n", _beta);            os << str;
  sprintf(str, "  k0    = %.3f (real only)\n", _k0);  os << str;
  os.flush();
}


void NPME_Kfunc_Helmholtz_SR_Alt::Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2   = x_f0[i]*x_f0[i] + y[i]*y[i] + z[i]*z[i];
    const double r    = sqrt(fabs(r2));
    const double v    = r*_k0;
    x_f0[i]           = _beta*NPME_Berfc_0 (r*_beta)*cos(v);
  }
}

void NPME_Kfunc_Helmholtz_SR_Alt::Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX[i]*x_fX[i] + y_fY[i]*y_fY[i] + z_fZ[i]*z_fZ[i];
    const double r  = sqrt(fabs(r2));

    double B0, B1;
    B0 = NPME_Berfc_1 (B1, _beta*r);

    double C0;
    C0 = NPME_sinx_x (_k0*r);

    double v      = r*_k0;
    double cos_v  = cos(v);
    f0[i]         = _beta* B0*cos_v;
    double f1     = _beta3*B1*cos_v - _k02_beta*B0*C0;


    x_fX[i]         = x_fX[i]*f1;
    y_fY[i]         = y_fY[i]*f1;
    z_fZ[i]         = z_fZ[i]*f1;
  }
}



#if NPME_USE_AVX
//AVX intrinsic functions of above
void NPME_Kfunc_Helmholtz_SR_Alt::CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_SR_Alt::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop              = N/4;
  const __m256d _k0_Vec256        = _mm256_set1_pd(_k0);
  const __m256d _k03_Vec256       = _mm256_set1_pd(_k03);
  const __m256d _k02_beta_Vec256  = _mm256_set1_pd(_k02_beta);
  const __m256d _beta_Vec256      = _mm256_set1_pd(_beta);
  const __m256d _beta3_Vec256     = _mm256_set1_pd(_beta3);

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm256_load_pd (&x_f0[count]);
    yVec  = _mm256_load_pd (&y[count]);
    zVec  = _mm256_load_pd (&z[count]);

    r2Vec = _mm256_mul_pd  (xVec, xVec);
    #if NPME_USE_AVX_FMA
    {
      r2Vec  = _mm256_fmadd_pd  (yVec, yVec, r2Vec);
      r2Vec  = _mm256_fmadd_pd  (zVec, zVec, r2Vec);
    }
    #else
    {
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (yVec, yVec), r2Vec);
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (zVec, zVec), r2Vec);
    }
    #endif
    rVec  = _mm256_sqrt_pd (r2Vec);

    __m256d f0Vec;
    {
      __m256d B0_vec;
      NPME_Berfc_0_AVX (B0_vec, _mm256_mul_pd  (_beta_Vec256, rVec));

      __m256d sin_vec, cos_vec;
      sin_vec = _mm256_sincos_pd (&cos_vec, _mm256_mul_pd  (_k0_Vec256, rVec));

      f0Vec   = _mm256_mul_pd  (_beta_Vec256, B0_vec);
      f0Vec   = _mm256_mul_pd  (cos_vec,      f0Vec);
    }
    _mm256_store_pd (&x_f0[count], f0Vec);

    count += 4;
  }
}

void NPME_Kfunc_Helmholtz_SR_Alt::CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_SR_Alt::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop              = N/4;
  const __m256d _k0_Vec256        = _mm256_set1_pd(_k0);
  const __m256d _k03_Vec256       = _mm256_set1_pd(_k03);
  const __m256d _k02_beta_Vec256  = _mm256_set1_pd(_k02_beta);
  const __m256d _beta_Vec256      = _mm256_set1_pd(_beta);
  const __m256d _beta3_Vec256     = _mm256_set1_pd(_beta3);

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm256_load_pd (&x_fX[count]);
    yVec  = _mm256_load_pd (&y_fY[count]);
    zVec  = _mm256_load_pd (&z_fZ[count]);

    r2Vec = _mm256_mul_pd  (xVec, xVec);
    #if NPME_USE_AVX_FMA
    {
      r2Vec  = _mm256_fmadd_pd  (yVec, yVec, r2Vec);
      r2Vec  = _mm256_fmadd_pd  (zVec, zVec, r2Vec);
    }
    #else
    {
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (yVec, yVec), r2Vec);
      r2Vec  = _mm256_add_pd  (_mm256_mul_pd (zVec, zVec), r2Vec);
    }
    #endif
    rVec  = _mm256_sqrt_pd (r2Vec);

    __m256d f0Vec, f1Vec, fXVec, fYVec, fZVec;

    {
      __m256d B0_vec, B1_vec;
      NPME_Berfc_1_AVX (B0_vec, B1_vec, _mm256_mul_pd  (_beta_Vec256, rVec));

      __m256d C0_vec, cos_vec;
      NPME_sinx_x_AVX (C0_vec, cos_vec, _mm256_mul_pd  (_k0_Vec256, rVec));

      //f0    = _beta*B0*cos(v);
      f0Vec   = _mm256_mul_pd  (_beta_Vec256, B0_vec);
      f0Vec   = _mm256_mul_pd  (cos_vec,      f0Vec);

      //f1    = _beta3*B1*cos_v - _k02_beta*B0*C0
      __m256d B1_beta3_vec;
      B1_beta3_vec  = _mm256_mul_pd  (_beta3_Vec256, B1_vec);
      f1Vec         = _mm256_mul_pd  (_k02_beta_Vec256, B0_vec);
      f1Vec         = _mm256_mul_pd  (C0_vec,           f1Vec);

      #if NPME_USE_AVX_FMA
      {
        f1Vec = _mm256_fmsub_pd  (B1_beta3_vec, cos_vec, f1Vec);
      }
      #else
      {
        f1Vec = _mm256_sub_pd  (_mm256_mul_pd (B1_beta3_vec, cos_vec), f1Vec);
      }
      #endif
    }

    fXVec = _mm256_mul_pd  (f1Vec, xVec);
    fYVec = _mm256_mul_pd  (f1Vec, yVec);
    fZVec = _mm256_mul_pd  (f1Vec, zVec);

    _mm256_store_pd (  &f0[count], f0Vec);
    _mm256_store_pd (&x_fX[count], fXVec);
    _mm256_store_pd (&y_fY[count], fYVec);
    _mm256_store_pd (&z_fZ[count], fZVec);

    count += 4;
  }
}
#endif




#if NPME_USE_AVX_512
//AVX_512 intrinsic functions of above
void NPME_Kfunc_Helmholtz_SR_Alt::CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_SR_Alt::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop              = N/8;
  const __m512d _k0_Vec512        = _mm512_set1_pd(_k0);
  const __m512d _k03_Vec512       = _mm512_set1_pd(_k03);
  const __m512d _k02_beta_Vec512  = _mm512_set1_pd(_k02_beta);
  const __m512d _beta_Vec512      = _mm512_set1_pd(_beta);
  const __m512d _beta3_Vec512     = _mm512_set1_pd(_beta3);

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm512_load_pd (&x_f0[count]);
    yVec  = _mm512_load_pd (&y[count]);
    zVec  = _mm512_load_pd (&z[count]);

    r2Vec = _mm512_mul_pd  (xVec, xVec);
    r2Vec = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm512_fmadd_pd  (zVec, zVec, r2Vec);
    rVec  = _mm512_sqrt_pd (r2Vec);

    __m512d f0Vec;
    {
      __m512d B0_vec;
      NPME_Berfc_0_AVX_512 (B0_vec, _mm512_mul_pd  (_beta_Vec512, rVec));

      __m512d sin_vec, cos_vec;
      sin_vec = _mm512_sincos_pd (&cos_vec, _mm512_mul_pd  (_k0_Vec512, rVec));

      f0Vec   = _mm512_mul_pd  (_beta_Vec512, B0_vec);
      f0Vec   = _mm512_mul_pd  (cos_vec,      f0Vec);
    }
    _mm512_store_pd (&x_f0[count], f0Vec);

    count += 8;
  }
}

void NPME_Kfunc_Helmholtz_SR_Alt::CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Helmholtz_SR_Alt::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop              = N/8;
  const __m512d _k0_Vec512        = _mm512_set1_pd(_k0);
  const __m512d _k03_Vec512       = _mm512_set1_pd(_k03);
  const __m512d _k02_beta_Vec512  = _mm512_set1_pd(_k02_beta);
  const __m512d _beta_Vec512      = _mm512_set1_pd(_beta);
  const __m512d _beta3_Vec512     = _mm512_set1_pd(_beta3);

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm512_load_pd (&x_fX[count]);
    yVec  = _mm512_load_pd (&y_fY[count]);
    zVec  = _mm512_load_pd (&z_fZ[count]);

    r2Vec = _mm512_mul_pd  (xVec, xVec);
    r2Vec = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm512_fmadd_pd  (zVec, zVec, r2Vec);
    rVec  = _mm512_sqrt_pd (r2Vec);

    __m512d f0Vec, f1Vec, fXVec, fYVec, fZVec;
    {
      __m512d B0_vec, B1_vec;
      NPME_Berfc_1_AVX_512 (B0_vec, B1_vec, _mm512_mul_pd (_beta_Vec512, rVec));

      __m512d C0_vec, cos_vec;
      NPME_sinx_x_AVX_512 (C0_vec, cos_vec, _mm512_mul_pd  (_k0_Vec512, rVec));

      //f0    = _beta*B0*cos(v);
      f0Vec   = _mm512_mul_pd  (_beta_Vec512, B0_vec);
      f0Vec   = _mm512_mul_pd  (cos_vec,      f0Vec);

      //f1    = _beta3*B1*cos_v - _k02_beta*B0*C0
      __m512d B1_beta3_vec;
      B1_beta3_vec  = _mm512_mul_pd  (_beta3_Vec512, B1_vec);
      f1Vec         = _mm512_mul_pd  (_k02_beta_Vec512, B0_vec);
      f1Vec         = _mm512_mul_pd  (C0_vec,           f1Vec);
      f1Vec         = _mm512_fmsub_pd  (B1_beta3_vec, cos_vec, f1Vec);
    }


    fXVec = _mm512_mul_pd  (f1Vec, xVec);
    fYVec = _mm512_mul_pd  (f1Vec, yVec);
    fZVec = _mm512_mul_pd  (f1Vec, zVec);

    _mm512_store_pd (  &f0[count], f0Vec);
    _mm512_store_pd (&x_fX[count], fXVec);
    _mm512_store_pd (&y_fY[count], fYVec);
    _mm512_store_pd (&z_fZ[count], fZVec);

    count += 8;
  }
}
#endif

}//end namespace NPME_Library



