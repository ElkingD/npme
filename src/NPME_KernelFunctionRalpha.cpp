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
#include "NPME_KernelFunctionRalpha.h"
#include "NPME_FunctionDerivMatch.h"


namespace NPME_Library
{
//******************************************************************************
//******************************************************************************
//*******************************NPME_Kfunc_Ralpha******************************
//******************************************************************************
//******************************************************************************

bool NPME_Kfunc_Ralpha::SetParm (const double alpha)
{
  _alpha = alpha;

  return true;
}

void NPME_Kfunc_Ralpha::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Ralpha::Print\n";
  sprintf(str, "  alpha = %.3f\n", _alpha);
  os << str;
  os.flush();
}

void NPME_Kfunc_Ralpha::Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_f0[i]*x_f0[i] + y[i]*y[i] + z[i]*z[i];
    const double r  = sqrt(fabs(r2));
    x_f0[i]         = pow(r2, _alpha/2);
  }
}

void NPME_Kfunc_Ralpha::Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX[i]*x_fX[i] + y_fY[i]*y_fY[i] + z_fZ[i]*z_fZ[i];
    const double r  = sqrt(fabs(r2));
    f0[i]           = pow(r2, _alpha/2);
    const double f1 = _alpha*f0[i]/r2;

    x_fX[i]         = x_fX[i]*f1;
    y_fY[i]         = y_fY[i]*f1;
    z_fZ[i]         = z_fZ[i]*f1;
  }
}


#if NPME_USE_AVX
//AVX intrinsic functions of above
void NPME_Kfunc_Ralpha::CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop            = N/4;
  const __m256d _alphaVec256    = _mm256_set1_pd(_alpha);
  const __m256d _alpha_2_Vec256 = _mm256_set1_pd(_alpha/2);

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
    f0Vec = _mm256_pow_pd (r2Vec, _alpha_2_Vec256);

    _mm256_store_pd (&x_f0[count], f0Vec);

    count += 4;
  }
}

void NPME_Kfunc_Ralpha::CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop            = N/4;
  const __m256d _alphaVec256    = _mm256_set1_pd(_alpha);
  const __m256d _alpha_2_Vec256 = _mm256_set1_pd(_alpha/2);

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
    f0Vec = _mm256_pow_pd (r2Vec, _alpha_2_Vec256);

    f1Vec = _mm256_div_pd  (f0Vec, r2Vec);
    f1Vec = _mm256_mul_pd  (_alphaVec256, f1Vec);

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
void NPME_Kfunc_Ralpha::CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop            = N/8;
  const __m512d _alphaVec512    = _mm512_set1_pd(_alpha);
  const __m512d _alpha_2_Vec512 = _mm512_set1_pd(_alpha/2);

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

    __m512d f0Vec = _mm512_pow_pd (r2Vec, _alpha_2_Vec512);

    _mm512_store_pd (&x_f0[count], f0Vec);

    count += 8;
  }
}

void NPME_Kfunc_Ralpha::CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop            = N/8;
  const __m512d _alphaVec512    = _mm512_set1_pd(_alpha);
  const __m512d _alpha_2_Vec512 = _mm512_set1_pd(_alpha/2);

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

    f0Vec = _mm512_pow_pd (r2Vec, _alpha_2_Vec512);
    f1Vec = _mm512_div_pd  (f0Vec, r2Vec);
    f1Vec = _mm512_mul_pd  (_alphaVec512, f1Vec);


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



//******************************************************************************
//******************************************************************************
//*****************************NPME_Kfunc_Ralpha_LR_DM**************************
//******************************************************************************
//******************************************************************************


bool NPME_Kfunc_Ralpha_LR_DM::SetParm (const double alpha, const int Nder, 
    const double Rdir, bool PRINT, std::ostream& os)
{
  if (Nder > NPME_MaxDerivMatchOrder)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha_LR_DM::SetParm\n";
    char str[2000];
    sprintf(str, "  Nder = %d > %d = NPME_MaxDerivMatchOrder\n", Nder,
      NPME_MaxDerivMatchOrder);
    std::cout << str;
    return false;
  }

  _alpha = alpha;
  _Nder  = Nder;
  _Rdir  = Rdir;

  std::vector<double> f(_Nder+1);
  NPME_FunctionDerivMatch_RalphaRadialDeriv (&f[0], _Nder, _Rdir, _alpha);
  if (!NPME_FunctionDerivMatch_CalcEvenSeries (&_a[0], &_b[0], 
    &f[0], _Nder, _Rdir))
  {
    std::cout << "Error in NPME_Kfunc_Ralpha_LR_DM::SetParm\n";
    std::cout << "NPME_FunctionDerivMatch_CalcEvenSeries failed\n";
    return false;
  }


  if (PRINT)
  {
    char str[2000];
    os << "\n\nNPME_Kfunc_Ralpha_LR_DM::SetParm\n";
    sprintf(str, "   Rdir  = %10.6f\n", _Rdir);   os << str;
    sprintf(str, "   Nder  = %d\n", _Nder);       os << str;
    os << "\n";
    for (int i = 0; i <= _Nder; i++)
    {
      sprintf(str, "      a[%4d] = %25.15le\n", i, _a[i]);
      os << str;
    }
    os << "\n";
    for (int i = 0; i <= _Nder; i++)
    {
      sprintf(str, "      b[%4d] = %25.15le\n", i, _b[i]);
      os << str;
    }
  }

  return true;
}
void NPME_Kfunc_Ralpha_LR_DM::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Ralpha_LR_DM::Print\n";
  sprintf(str, "  alpha = %.3f\n", _alpha);   os << str;
  sprintf(str, "  Rdir  = %.3f\n", _Rdir);    os << str;
  sprintf(str, "  Nder  = %3d\n", _Nder);     os << str;
  os.flush();
}

void NPME_Kfunc_Ralpha_LR_DM::Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_f0[i]*x_f0[i] + y[i]*y[i] + z[i]*z[i];
    const double r  = sqrt(fabs(r2));
    if (r2 >_Rdir*_Rdir)
      x_f0[i] = pow(r2, _alpha/2);
    else
      x_f0[i] = NPME_FunctionDerivMatch_EvenSeriesReal (_Nder, &_a[0], r2);
  }
}

void NPME_Kfunc_Ralpha_LR_DM::Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX[i]*x_fX[i] + y_fY[i]*y_fY[i] + z_fZ[i]*z_fZ[i];
    const double r  = sqrt(fabs(r2));
    double f1;
    if (r2 >_Rdir*_Rdir)
    {
      f0[i] = pow(r2, _alpha/2);
      f1    = _alpha*f0[i]/r2;

    }
    else
    {
      f0[i] = NPME_FunctionDerivMatch_EvenSeriesReal (f1, _Nder, 
                &_a[0], &_b[0], r2);
    }

    x_fX[i]         = x_fX[i]*f1;
    y_fY[i]         = y_fY[i]*f1;
    z_fZ[i]         = z_fZ[i]*f1;
  }
}



#if NPME_USE_AVX
//AVX intrinsic functions of above
void NPME_Kfunc_Ralpha_LR_DM::CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha_LR_DM::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop            = N/4;
  const __m256d _alphaVec256    = _mm256_set1_pd(_alpha);
  const __m256d _alpha_2_Vec256 = _mm256_set1_pd(_alpha/2);
  const __m256d _Rdir2Vec256    = _mm256_set1_pd( _Rdir*_Rdir );

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
    __m256d f0_AVec;
    __m256d f0_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX (f0_AVec, r2Vec, _Nder, &_a[0]);
    f0_BVec = _mm256_pow_pd (r2Vec, _alpha_2_Vec256);

    //use (f0_AVec) if r < Rdir
    //use (f0_BVec) if r > Rdir
    {
      __m256d t0, dless, dmore;
      t0      = _mm256_cmp_pd (r2Vec, _Rdir2Vec256, 1);

      dless   = _mm256_and_pd    (t0, f0_AVec);
      dmore   = _mm256_andnot_pd (t0, f0_BVec);
      f0Vec   = _mm256_add_pd (dless, dmore);
    }

    _mm256_store_pd (&x_f0[count], f0Vec);

    count += 4;
  }
}

void NPME_Kfunc_Ralpha_LR_DM::CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha_LR_DM::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop            = N/4;
  const __m256d _alphaVec256    = _mm256_set1_pd(_alpha);
  const __m256d _alpha_2_Vec256 = _mm256_set1_pd(_alpha/2);
  const __m256d _Rdir2Vec256    = _mm256_set1_pd( _Rdir*_Rdir );

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
    __m256d f0_AVec, f1_AVec;
    __m256d f0_BVec, f1_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX (f0_AVec, f1_AVec, 
      r2Vec, _Nder, &_a[0], &_b[0]);

    f0_BVec = _mm256_pow_pd (r2Vec, _alpha_2_Vec256);
    f1_BVec = _mm256_div_pd  (f0_BVec, r2Vec);
    f1_BVec = _mm256_mul_pd  (_alphaVec256, f1_BVec);

    //use (f0_AVec) if r < Rdir
    //use (f0_BVec) if r > Rdir
    {
      __m256d t0, dless, dmore;
      t0      = _mm256_cmp_pd (r2Vec, _Rdir2Vec256, 1);

      dless   = _mm256_and_pd    (t0, f0_AVec);
      dmore   = _mm256_andnot_pd (t0, f0_BVec);
      f0Vec   = _mm256_add_pd (dless, dmore);

      dless   = _mm256_and_pd    (t0, f1_AVec);
      dmore   = _mm256_andnot_pd (t0, f1_BVec);
      f1Vec   = _mm256_add_pd (dless, dmore);
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
void NPME_Kfunc_Ralpha_LR_DM::CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha_LR_DM::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop            = N/8;
  const __m512d _alphaVec512    = _mm512_set1_pd(_alpha);
  const __m512d _alpha_2_Vec512 = _mm512_set1_pd(_alpha/2);
  const __m512d _Rdir2Vec512    = _mm512_set1_pd( _Rdir*_Rdir );

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
    __m512d f0_AVec;
    __m512d f0_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (f0_AVec, r2Vec, _Nder, &_a[0]);
    f0_BVec = _mm512_pow_pd (r2Vec, _alpha_2_Vec512);

    //use (f0_AVec) if r < Rdir
    //use (f0_BVec) if r > Rdir
    {
      __mmask8 maskVec = _mm512_cmp_pd_mask (r2Vec, _Rdir2Vec512, 1);
      f0Vec = _mm512_mask_mov_pd (f0_BVec, maskVec, f0_AVec);
    }

    _mm512_store_pd (&x_f0[count], f0Vec);

    count += 8;
  }
}

void NPME_Kfunc_Ralpha_LR_DM::CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha_LR_DM::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop            = N/8;
  const __m512d _alphaVec512    = _mm512_set1_pd(_alpha);
  const __m512d _alpha_2_Vec512 = _mm512_set1_pd(_alpha/2);
  const __m512d _Rdir2Vec512    = _mm512_set1_pd( _Rdir*_Rdir );

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

    __m512d f0_AVec, f1_AVec;
    __m512d f0_BVec, f1_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (f0_AVec, f1_AVec, 
      r2Vec, _Nder, &_a[0], &_b[0]);

    f0_BVec = _mm512_pow_pd (r2Vec, _alpha_2_Vec512);
    f1_BVec = _mm512_div_pd  (f0_BVec, r2Vec);
    f1_BVec = _mm512_mul_pd  (_alphaVec512, f1_BVec);

    //use (f0_AVec) if r < Rdir
    //use (f0_BVec) if r > Rdir
    {
      __mmask8 maskVec = _mm512_cmp_pd_mask (r2Vec, _Rdir2Vec512, 1);
      f0Vec = _mm512_mask_mov_pd (f0_BVec, maskVec, f0_AVec);
      f1Vec = _mm512_mask_mov_pd (f1_BVec, maskVec, f1_AVec);
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







//******************************************************************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
//*****************************NPME_Kfunc_Ralpha_SR_DM**************************
//******************************************************************************
//******************************************************************************

bool NPME_Kfunc_Ralpha_SR_DM::SetParm (const double alpha, const int Nder, 
    const double Rdir, bool PRINT, std::ostream& os)
{
  if (Nder > NPME_MaxDerivMatchOrder)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha_SR_DM::SetParm\n";
    char str[2000];
    sprintf(str, "  Nder = %d > %d = NPME_MaxDerivMatchOrder\n", Nder,
      NPME_MaxDerivMatchOrder);
    std::cout << str;
    return false;
  }

  _alpha = alpha;
  _Nder  = Nder;
  _Rdir  = Rdir;


  std::vector<double> f(_Nder+1);
  NPME_FunctionDerivMatch_RalphaRadialDeriv (&f[0], _Nder, _Rdir, _alpha);
  if (!NPME_FunctionDerivMatch_CalcEvenSeries (&_a[0], &_b[0], 
    &f[0], _Nder, _Rdir))
  {
    std::cout << "Error in NPME_Kfunc_Ralpha_SR_DM::SetParm\n";
    std::cout << "NPME_FunctionDerivMatch_CalcEvenSeries failed\n";
    return false;
  }

  if (PRINT)
  {
    char str[2000];
    os << "\n\nNPME_Kfunc_Ralpha_SR_DM::SetParm\n";
    sprintf(str, "   Rdir  = %10.6f\n", _Rdir);   os << str;
    sprintf(str, "   Nder  = %d\n", _Nder);       os << str;
    os << "\n";
    for (int i = 0; i <= _Nder; i++)
    {
      sprintf(str, "      a[%4d] = %25.15le\n", i, _a[i]);
      os << str;
    }
    os << "\n";
    for (int i = 0; i <= _Nder; i++)
    {
      sprintf(str, "      b[%4d] = %25.15le\n", i, _b[i]);
      os << str;
    }
  }

  return true;
}

void NPME_Kfunc_Ralpha_SR_DM::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Ralpha_SR_DM::Print\n";
  sprintf(str, "  alpha = %.3f\n", _alpha);   os << str;
  sprintf(str, "  Rdir  = %.3f\n", _Rdir);    os << str;
  sprintf(str, "  Nder  = %3d\n", _Nder);     os << str;
  os.flush();
}

void NPME_Kfunc_Ralpha_SR_DM::Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_f0[i]*x_f0[i] + y[i]*y[i] + z[i]*z[i];
    const double r  = sqrt(fabs(r2));
    if (r2 >_Rdir*_Rdir)
      x_f0[i] = 0;
    else
      x_f0[i] = pow(r2, _alpha/2) - 
                NPME_FunctionDerivMatch_EvenSeriesReal (_Nder, &_a[0], r2);
  }
}

void NPME_Kfunc_Ralpha_SR_DM::Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX[i]*x_fX[i] + y_fY[i]*y_fY[i] + z_fZ[i]*z_fZ[i];
    const double r  = sqrt(fabs(r2));
    double f1;
    if (r2 >_Rdir*_Rdir)
    {
      f0[i] = 0.0;
      f1    = 0.0;
    }
    else
    {
      double f0_exact = pow(r2, _alpha/2);
      double f1_exact = _alpha*f0_exact/r2;

      f0[i] = NPME_FunctionDerivMatch_EvenSeriesReal (f1, _Nder, 
                &_a[0], &_b[0], r2);

      f0[i] = f0_exact - f0[i];
      f1    = f1_exact - f1;
    }

    x_fX[i]         = x_fX[i]*f1;
    y_fY[i]         = y_fY[i]*f1;
    z_fZ[i]         = z_fZ[i]*f1;
  }
}




#if NPME_USE_AVX
//AVX intrinsic functions of above
void NPME_Kfunc_Ralpha_SR_DM::CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  const __m256d zeroVec = _mm256_set1_pd(0.0);

  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha_SR_DM::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop            = N/4;
  const __m256d _alphaVec256    = _mm256_set1_pd(_alpha);
  const __m256d _alpha_2_Vec256 = _mm256_set1_pd(_alpha/2);
  const __m256d _Rdir2Vec256    = _mm256_set1_pd( _Rdir*_Rdir );

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
    __m256d f0_AVec;
    __m256d f0_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX (f0_AVec, r2Vec, _Nder, &_a[0]);
    f0_BVec = _mm256_pow_pd (r2Vec, _alpha_2_Vec256);

    f0_BVec = _mm256_sub_pd (f0_BVec, f0_AVec);

    //use (f0_BVec) if r < Rdir
    //use (zeroVec) if r > Rdir
    {
      __m256d t0, dless, dmore;
      t0      = _mm256_cmp_pd (r2Vec, _Rdir2Vec256, 1);

      dless   = _mm256_and_pd    (t0, f0_BVec);
      dmore   = _mm256_andnot_pd (t0, zeroVec);
      f0Vec   = _mm256_add_pd (dless, dmore);
    }

    _mm256_store_pd (&x_f0[count], f0Vec);

    count += 4;
  }
}

void NPME_Kfunc_Ralpha_SR_DM::CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  const __m256d zeroVec = _mm256_set1_pd(0.0);

  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha_SR_DM::CalcAVX\n";
    std::cout << "N = " << N << " must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop            = N/4;
  const __m256d _alphaVec256    = _mm256_set1_pd(_alpha);
  const __m256d _alpha_2_Vec256 = _mm256_set1_pd(_alpha/2);
  const __m256d _Rdir2Vec256    = _mm256_set1_pd( _Rdir*_Rdir );

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
    __m256d f0_AVec, f1_AVec;
    __m256d f0_BVec, f1_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX (f0_AVec, f1_AVec, 
      r2Vec, _Nder, &_a[0], &_b[0]);

    f0_BVec = _mm256_pow_pd (r2Vec, _alpha_2_Vec256);
    f1_BVec = _mm256_div_pd  (f0_BVec, r2Vec);
    f1_BVec = _mm256_mul_pd  (_alphaVec256, f1_BVec);

    f0_BVec = _mm256_sub_pd (f0_BVec, f0_AVec);
    f1_BVec = _mm256_sub_pd (f1_BVec, f1_AVec);

    //use (f0_BVec) if r < Rdir
    //use (zeroVec) if r > Rdir
    {
      __m256d t0, dless, dmore;
      t0      = _mm256_cmp_pd (r2Vec, _Rdir2Vec256, 1);

      dless   = _mm256_and_pd    (t0, f0_BVec);
      dmore   = _mm256_andnot_pd (t0, zeroVec);
      f0Vec   = _mm256_add_pd (dless, dmore);

      dless   = _mm256_and_pd    (t0, f1_BVec);
      dmore   = _mm256_andnot_pd (t0, zeroVec);
      f1Vec   = _mm256_add_pd (dless, dmore);
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
void NPME_Kfunc_Ralpha_SR_DM::CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  const __m512d zeroVec = _mm512_set1_pd(0.0);
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha_SR_DM::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop            = N/8;
  const __m512d _alphaVec512    = _mm512_set1_pd(_alpha);
  const __m512d _alpha_2_Vec512 = _mm512_set1_pd(_alpha/2);
  const __m512d _Rdir2Vec512    = _mm512_set1_pd( _Rdir*_Rdir );

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
    __m512d f0_AVec;
    __m512d f0_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (f0_AVec, r2Vec, _Nder, &_a[0]);
    f0_BVec = _mm512_pow_pd (r2Vec, _alpha_2_Vec512);

    f0_BVec = _mm512_sub_pd (f0_BVec, f0_AVec);

    //use (f0_BVec) if r < Rdir
    //use (zeroVec) if r > Rdir
    {
      __mmask8 maskVec = _mm512_cmp_pd_mask (r2Vec, _Rdir2Vec512, 1);
      f0Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f0_BVec);
    }

    _mm512_store_pd (&x_f0[count], f0Vec);

    count += 8;
  }
}

void NPME_Kfunc_Ralpha_SR_DM::CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  const __m512d zeroVec = _mm512_set1_pd(0.0);
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Ralpha_SR_DM::CalcAVX_512\n";
    std::cout << "N = " << N << " must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop            = N/8;
  const __m512d _alphaVec512    = _mm512_set1_pd(_alpha);
  const __m512d _alpha_2_Vec512 = _mm512_set1_pd(_alpha/2);
  const __m512d _Rdir2Vec512    = _mm512_set1_pd( _Rdir*_Rdir );

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

    __m512d f0_AVec, f1_AVec;
    __m512d f0_BVec, f1_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (f0_AVec, f1_AVec, 
      r2Vec, _Nder, &_a[0], &_b[0]);

    f0_BVec = _mm512_pow_pd (r2Vec, _alpha_2_Vec512);
    f1_BVec = _mm512_div_pd  (f0_BVec, r2Vec);
    f1_BVec = _mm512_mul_pd  (_alphaVec512, f1_BVec);

    f0_BVec = _mm512_sub_pd (f0_BVec, f0_AVec);
    f1_BVec = _mm512_sub_pd (f1_BVec, f1_AVec);

    //use (f0_BVec) if r < Rdir
    //use (zeroVec) if r > Rdir
    {
      __mmask8 maskVec = _mm512_cmp_pd_mask (r2Vec, _Rdir2Vec512, 1);
      f0Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f0_BVec);
      f1Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f1_BVec);
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



