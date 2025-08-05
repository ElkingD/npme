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

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <complex.h>
#include <cstdio>

#include <iostream> 
#include <vector> 

#include <immintrin.h>


#include "Constant.h"
#include "KernelFunction.h"
#include "KernelFunctionLaplace.h"
#include "FunctionDerivMatch.h"
#include "MathFunctions.h"

namespace NPME_Library
{
//******************************************************************************
//******************************************************************************
//**************************Some Support Functions******************************
//******************************************************************************
//******************************************************************************

double NPME_EwaldSplitOrig_Rdir2Beta (const double Rdir, const double tol)
//find beta s.t. 
//erfc(Rdir*beta)/Rdir <= tol
//erfc(Rdir*beta) <= tol*Rdir
//x = Rdir*beta
//erfc(x)             ~ exp(-x2)/sqrt(NPME_Pi)/x
//exp(-x2)/x/sqrt(NPME_Pi) ~ tol*Rdir
//exp(-x2)/x          ~ tol*Rdir*sqrt(NPME_Pi)
//-x2 - log(x)        ~  log( tol*Rdir*sqrt(NPME_Pi) )
//x2 + log(x) + log( tol*Rdir*sqrt(NPME_Pi) ) = 0
{
  const int MaxIteration  = 1000;
  const double precision  = 1.0E-12;
  const double C          = log(sqrt(NPME_Pi)*tol*Rdir);

  double x = 0.1;
  for (int n = 0; n < MaxIteration; n++)
  {
    double F0 = x*x + log(x) + C;
    double F1 = 2*x + 1.0/x;

    if (fabs(F0) < precision)
      return x/Rdir;
    x -= F0/F1;
  }

  std::cout << "Error in NPME_EwaldSplitOrig_Rdir2Beta.\n";
  std::cout << "  did not reach convergence\n";
  std::cout.precision(2);
  std::cout << std::scientific;
  std::cout << "  Rdir = " << Rdir << std::endl;
  std::cout << "  tol  = " << tol  << std::endl;
  exit(0);
  return 0;
}

#if NPME_USE_AVX
void NPME_Kfunc_Laplace_AVX (__m256d& f0Vec, const __m256d& rVec)
{
  const __m256d oneVec = _mm256_set1_pd(1.0);
  f0Vec = _mm256_div_pd  (oneVec, rVec);
}

void NPME_Kfunc_Laplace_AVX (__m256d& f0Vec, __m256d& f1Vec, const __m256d& rVec, 
  const __m256d& r2Vec)
{
  const __m256d oneVec    = _mm256_set1_pd( 1.0);
  const __m256d negOneVec = _mm256_set1_pd(-1.0);
  f0Vec         = _mm256_div_pd  (oneVec, rVec);
  f1Vec         = _mm256_div_pd  (f0Vec, r2Vec);
  f1Vec         = _mm256_mul_pd  (f1Vec, negOneVec);
}
#endif

#if NPME_USE_AVX_512
void NPME_Kfunc_Laplace_AVX_512 (__m512d& f0Vec, const __m512d& rVec)
{
  const __m512d oneVec = _mm512_set1_pd(1.0);
  f0Vec = _mm512_div_pd  (oneVec, rVec);
}

void NPME_Kfunc_Laplace_AVX_512 (__m512d& f0Vec, __m512d& f1Vec, 
  const __m512d& rVec, const __m512d& r2Vec)
{
  const __m512d oneVec    = _mm512_set1_pd( 1.0);
  const __m512d negOneVec = _mm512_set1_pd(-1.0);
  f0Vec         = _mm512_div_pd  (oneVec, rVec);
  f1Vec         = _mm512_div_pd  (f0Vec, r2Vec);
  f1Vec         = _mm512_mul_pd  (f1Vec, negOneVec);
}
#endif

//******************************************************************************
//******************************************************************************
//*******************************NPME_Kfunc_Laplace*****************************
//******************************************************************************
//******************************************************************************

void NPME_Kfunc_Laplace::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Laplace::Print\n";
  os.flush();
}

void NPME_Kfunc_Laplace::Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_f0[i]*x_f0[i] + y[i]*y[i] + z[i]*z[i];
    const double r  = sqrt(fabs(r2));
    x_f0[i]         = 1.0/r;
  }
}

void NPME_Kfunc_Laplace::Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX[i]*x_fX[i] + y_fY[i]*y_fY[i] + z_fZ[i]*z_fZ[i];
    const double r  = sqrt(fabs(r2));
    f0[i]           = 1.0/r;
    const double f1 = -f0[i]/r2;

    x_fX[i]         = x_fX[i]*f1;
    y_fY[i]         = y_fY[i]*f1;
    z_fZ[i]         = z_fZ[i]*f1;
  }
}

#if NPME_USE_AVX
//AVX intrinsic functions of above
void NPME_Kfunc_Laplace::CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace::CalcAVX\n";
    std::cout << "N = " << N << "must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop = N/4;

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
    NPME_Kfunc_Laplace_AVX (f0Vec, rVec);

    _mm256_store_pd (&x_f0[count], f0Vec);

    count += 4;
  }
}

void NPME_Kfunc_Laplace::CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace::CalcAVX\n";
    std::cout << "N = " << N << "must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop = N/4;

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
    NPME_Kfunc_Laplace_AVX (f0Vec, f1Vec, rVec, r2Vec);

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
void NPME_Kfunc_Laplace::CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace::CalcAVX_512\n";
    std::cout << "N = " << N << "must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop = N/8;

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
    NPME_Kfunc_Laplace_AVX_512 (f0Vec, rVec);

    _mm512_store_pd (&x_f0[count], f0Vec);

    count += 8;
  }
}

void NPME_Kfunc_Laplace::CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace::CalcAVX_512\n";
    std::cout << "N = " << N << "must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop = N/8;

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
    NPME_Kfunc_Laplace_AVX_512 (f0Vec, f1Vec, rVec, r2Vec);

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
//****************************NPME_Kfunc_Laplace_LR_DM**************************
//******************************************************************************
//******************************************************************************



bool NPME_Kfunc_Laplace_LR_DM::SetParm (const int Nder, 
  const double Rdir, bool PRINT, std::ostream& os)
{
  if (Nder > NPME_MaxDerivMatchOrder)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_LR_DM::SetParm\n";
    char str[2000];
    sprintf(str, "  Nder = %d > %d = NPME_MaxDerivMatchOrder\n", Nder,
      NPME_MaxDerivMatchOrder);
    std::cout << str;
    return false;
  }

  _Nder = Nder;
  _Rdir = Rdir;

  std::vector<double> f(_Nder+1);
  NPME_FunctionDerivMatch_RalphaRadialDeriv (&f[0], _Nder, _Rdir, -1.0);
  if (!NPME_FunctionDerivMatch_CalcEvenSeries (&_a[0], &_b[0], 
    &f[0], _Nder, _Rdir))
  {
    std::cout << "Error in NPME_Kfunc_Laplace_LR_DM::SetParm\n";
    std::cout << "NPME_FunctionDerivMatch_CalcEvenSeries failed\n";
    return false;
  }

  if (PRINT)
  {
    char str[2000];
    os << "\n\nNPME_Kfunc_Laplace_LR_DM::SetParm\n";
    sprintf(str, "   Rdir  = %10.6f\n", _Rdir);   os << str;
    sprintf(str, "   Nder  = %d\n", _Nder);       os << str;
    os << "\n";
    for (int i = 0; i <= _Nder; i++)
    {
      sprintf(str, "      a[%4d] = %15.6le\n", i, _a[i]);
      os << str;
    }
    os << "\n";
    for (int i = 0; i <= _Nder; i++)
    {
      sprintf(str, "      b[%4d] = %15.6le\n", i, _b[i]);
      os << str;
    }
  }

  return true;
}

void NPME_Kfunc_Laplace_LR_DM::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Laplace_LR_DM::Print\n";
  sprintf(str, "  Rdir  = %.3f\n", _Rdir);    os << str;
  sprintf(str, "  Nder  = %3d\n", _Nder);     os << str;
  os.flush();
}

void NPME_Kfunc_Laplace_LR_DM::Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_f0[i]*x_f0[i] + y[i]*y[i] + z[i]*z[i];
    const double r  = sqrt(fabs(r2));
    x_f0[i]         = 1.0/r;

    if (r > _Rdir)
      x_f0[i] = 1.0/r;
    else
      x_f0[i] = NPME_FunctionDerivMatch_EvenSeriesReal (_Nder, &_a[0], r2);
  }
}

void NPME_Kfunc_Laplace_LR_DM::Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX[i]*x_fX[i] + y_fY[i]*y_fY[i] + z_fZ[i]*z_fZ[i];
    const double r  = sqrt(fabs(r2));

    double f1;
    if (r > _Rdir)
    {
      f0[i] = 1.0/r;
      f1    = -f0[i]/r2;
    }
    else
    {
      f0[i] = NPME_FunctionDerivMatch_EvenSeriesReal (f1, 
                _Nder, &_a[0], &_b[0], r2);
    }

    x_fX[i]         = x_fX[i]*f1;
    y_fY[i]         = y_fY[i]*f1;
    z_fZ[i]         = z_fZ[i]*f1;
  }
}



#if NPME_USE_AVX
void NPME_Kfunc_Laplace_LR_DM::CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_LR_DM::CalcAVX\n";
    std::cout << "N = " << N << "must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop          = N/4;
  const __m256d _RdirVec256   = _mm256_set1_pd( _Rdir );

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

    __m256d f0_Vec;
    __m256d f0_AVec;
    __m256d f0_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX (f0_AVec, r2Vec, _Nder, &_a[0]);
    NPME_Kfunc_Laplace_AVX (f0_BVec, rVec);

    //use (f0_AVec) if r < Rdir
    //use (f0_BVec) if r > Rdir
    {
      __m256d t0, dless, dmore;
      t0      = _mm256_cmp_pd (rVec, _RdirVec256, 1);

      dless   = _mm256_and_pd    (t0, f0_AVec);
      dmore   = _mm256_andnot_pd (t0, f0_BVec);
      f0_Vec  = _mm256_add_pd (dless, dmore);
    }


    _mm256_store_pd (&x_f0[count], f0_Vec);

    count += 4;
  }
}

void NPME_Kfunc_Laplace_LR_DM::CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
//input:  xVec[4], yVec[4], zVec[4]
//output: f0Vec[4] = components of kernel f0
//        fXVec[4] = components of fX = df0/dx
//        ...
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_LR_DM::CalcAVX\n";
    std::cout << "N = " << N << "must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop          = N/4;
  const __m256d _RdirVec256   = _mm256_set1_pd( _Rdir );

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



    __m256d f0_AVec, f1_AVec;
    __m256d f0_BVec, f1_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX (f0_AVec, f1_AVec, 
      r2Vec, _Nder, &_a[0], &_b[0]);
    NPME_Kfunc_Laplace_AVX (f0_BVec, f1_BVec, rVec, r2Vec);


    __m256d f0Vec, f1Vec;
    //use (f0_AVec) if r < Rdir
    //use (f0_BVec) if r > Rdir
    {
      __m256d t0, dless, dmore;
      t0      = _mm256_cmp_pd (rVec, _RdirVec256, 1);

      dless   = _mm256_and_pd    (t0, f0_AVec);
      dmore   = _mm256_andnot_pd (t0, f0_BVec);
      f0Vec   = _mm256_add_pd (dless, dmore);

      dless   = _mm256_and_pd    (t0, f1_AVec);
      dmore   = _mm256_andnot_pd (t0, f1_BVec);
      f1Vec   = _mm256_add_pd (dless, dmore);
    }

    __m256d fXVec, fYVec, fZVec;

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
void NPME_Kfunc_Laplace_LR_DM::CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_LR_DM::CalcAVX_512\n";
    std::cout << "N = " << N << "must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop        = N/8;
  const __m512d _RdirVec512 = _mm512_set1_pd( _Rdir );

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
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (f0_AVec, 
      r2Vec, _Nder, &_a[0]);
    NPME_Kfunc_Laplace_AVX_512 (f0_BVec, rVec);

    //use (f0_AVec) if r < Rdir
    //use (f0_BVec) if r > Rdir
    {
      __mmask8 maskVec = _mm512_cmp_pd_mask (rVec, _RdirVec512, 1);
      f0Vec = _mm512_mask_mov_pd (f0_BVec, maskVec, f0_AVec);
    }

    _mm512_store_pd (&x_f0[count], f0Vec);

    count += 8;
  }
}

void NPME_Kfunc_Laplace_LR_DM::CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_LR_DM::CalcAVX_512\n";
    std::cout << "N = " << N << "must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop        = N/8;
  const __m512d _RdirVec512 = _mm512_set1_pd( _Rdir );

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


    __m512d f0_AVec, f1_AVec;
    __m512d f0_BVec, f1_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (f0_AVec, f1_AVec, 
      r2Vec, _Nder, &_a[0], &_b[0]);
    NPME_Kfunc_Laplace_AVX_512 (f0_BVec, f1_BVec, rVec, r2Vec);

    __m512d f0Vec, f1Vec;
    //use (f0_AVec) if r < Rdir
    //use (f0_BVec) if r > Rdir
    {
      __mmask8 maskVec = _mm512_cmp_pd_mask (rVec, _RdirVec512, 1);
      f0Vec = _mm512_mask_mov_pd (f0_BVec, maskVec, f0_AVec);
      f1Vec = _mm512_mask_mov_pd (f1_BVec, maskVec, f1_AVec);
    }

    __m512d fXVec, fYVec, fZVec;

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
//****************************NPME_Kfunc_Laplace_SR_DM**************************
//******************************************************************************
//******************************************************************************



bool NPME_Kfunc_Laplace_SR_DM::SetParm (const int Nder, 
  const double Rdir, bool PRINT, std::ostream& os)
{
  if (Nder > NPME_MaxDerivMatchOrder)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_SR_DM::SetParm\n";
    char str[2000];
    sprintf(str, "  Nder = %d > %d = NPME_MaxDerivMatchOrder\n", Nder,
      NPME_MaxDerivMatchOrder);
    std::cout << str;
    return false;
  }

  _Nder = Nder;
  _Rdir = Rdir;


  std::vector<double> f(_Nder+1);
  NPME_FunctionDerivMatch_RalphaRadialDeriv (&f[0], _Nder, _Rdir, -1.0);
  if (!NPME_FunctionDerivMatch_CalcEvenSeries (&_a[0], &_b[0], 
    &f[0], _Nder, _Rdir))
  {
    std::cout << "Error in NPME_Kfunc_Laplace_SR_DM::SetParm\n";
    std::cout << "NPME_FunctionDerivMatch_CalcEvenSeries failed\n";
    return false;
  }

  if (PRINT)
  {
    char str[2000];
    os << "\n\nNPME_Kfunc_Laplace_SR_DM::SetParm\n";    
    sprintf(str, "   Rdir  = %10.6f\n", _Rdir);     os << str;
    sprintf(str, "   Nder  = %d\n", _Nder);         os << str;
    os << "\n";
    for (int i = 0; i <= _Nder; i++)
    {
      sprintf(str, "      a[%4d] = %15.6le\n", i, _a[i]);
      os << str;
    }
    os << "\n";
    for (int i = 0; i <= _Nder; i++)
    {
      sprintf(str, "      b[%4d] = %15.6le\n", i, _b[i]);
      os << str;
    }
  }

  return true;
}

void NPME_Kfunc_Laplace_SR_DM::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Laplace_SR_DM::Print\n";
  sprintf(str, "  Rdir  = %.3f\n", _Rdir);    os << str;
  sprintf(str, "  Nder  = %3d\n", _Nder);     os << str;
  os.flush();
}

void NPME_Kfunc_Laplace_SR_DM::Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_f0[i]*x_f0[i] + y[i]*y[i] + z[i]*z[i];
    const double r  = sqrt(fabs(r2));
    x_f0[i]         = 1.0/r;

    if (r > _Rdir)
      x_f0[i] = 0;
    else
      x_f0[i] = 1.0/r - 
                NPME_FunctionDerivMatch_EvenSeriesReal (_Nder, &_a[0], r2);
  }
}

void NPME_Kfunc_Laplace_SR_DM::Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX[i]*x_fX[i] + y_fY[i]*y_fY[i] + z_fZ[i]*z_fZ[i];
    const double r  = sqrt(fabs(r2));
    const double r3 = r*r2;

    double f1;
    if (r > _Rdir)
    {
      f0[i] = 0;
      f1    = 0;
    }
    else
    {
      f0[i] = 1.0/r - 
              NPME_FunctionDerivMatch_EvenSeriesReal (f1, 
                _Nder, &_a[0], &_b[0], r2);
      f1    = -1/r3 - f1;
    }

    x_fX[i]         = x_fX[i]*f1;
    y_fY[i]         = y_fY[i]*f1;
    z_fZ[i]         = z_fZ[i]*f1;
  }
}



#if NPME_USE_AVX
void NPME_Kfunc_Laplace_SR_DM::CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  const __m256d zeroVec = _mm256_set1_pd(0.0);
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_SR_DM::CalcAVX\n";
    std::cout << "N = " << N << "must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop        = N/4;
  const __m256d _RdirVec256 = _mm256_set1_pd( _Rdir );

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
    NPME_Kfunc_Laplace_AVX (f0_BVec, rVec);

    f0_BVec = _mm256_sub_pd (f0_BVec, f0_AVec);

    //use (f0_BVec) if r < Rdir
    //use (zeroVec) if r > Rdir
    {
      __m256d t0, dless, dmore;
      t0      = _mm256_cmp_pd (rVec, _RdirVec256, 1);

      dless   = _mm256_and_pd    (t0, f0_BVec);
      dmore   = _mm256_andnot_pd (t0, zeroVec);
      f0Vec   = _mm256_add_pd (dless, dmore);
    }

    _mm256_store_pd (&x_f0[count], f0Vec);

    count += 4;
  }
}

void NPME_Kfunc_Laplace_SR_DM::CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
//input:  xVec[4], yVec[4], zVec[4]
//output: f0Vec[4] = components of kernel f0
//        fXVec[4] = components of fX = df0/dx
//        ...
{
  const __m256d zeroVec = _mm256_set1_pd(0.0);
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_SR_DM::CalcAVX\n";
    std::cout << "N = " << N << "must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop        = N/4;
  const __m256d _RdirVec256 = _mm256_set1_pd( _Rdir );

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



    __m256d f0_AVec, f1_AVec;
    __m256d f0_BVec, f1_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX (f0_AVec, f1_AVec, 
      r2Vec, _Nder, &_a[0], &_b[0]);
    NPME_Kfunc_Laplace_AVX (f0_BVec, f1_BVec, rVec, r2Vec);

    f0_BVec = _mm256_sub_pd (f0_BVec, f0_AVec);
    f1_BVec = _mm256_sub_pd (f1_BVec, f1_AVec);

    __m256d f0Vec, f1Vec;
    //use (f0_BVec) if r < Rdir
    //use (zeroVec) if r > Rdir
    {
      __m256d t0, dless, dmore;
      t0      = _mm256_cmp_pd (rVec, _RdirVec256, 1);

      dless   = _mm256_and_pd    (t0, f0_BVec);
      dmore   = _mm256_andnot_pd (t0, zeroVec);
      f0Vec   = _mm256_add_pd (dless, dmore);

      dless   = _mm256_and_pd    (t0, f1_BVec);
      dmore   = _mm256_andnot_pd (t0, zeroVec);
      f1Vec   = _mm256_add_pd (dless, dmore);
    }

    __m256d fXVec, fYVec, fZVec;

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
void NPME_Kfunc_Laplace_SR_DM::CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  const __m512d zeroVec = _mm512_set1_pd(0.0);

  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_SR_DM::CalcAVX_512\n";
    std::cout << "N = " << N << "must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop        = N/8;
  const __m512d _RdirVec512 = _mm512_set1_pd( _Rdir );

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
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (f0_AVec, 
        r2Vec, _Nder, &_a[0]);
    NPME_Kfunc_Laplace_AVX_512 (f0_BVec, rVec);

    f0_BVec = _mm512_sub_pd (f0_BVec, f0_AVec);

    //use (f0_BVec) if r < Rdir
    //use (zeroVec) if r > Rdir
    {
      __mmask8 maskVec = _mm512_cmp_pd_mask (rVec, _RdirVec512, 1);
      f0Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f0_BVec);
    }

    _mm512_store_pd (&x_f0[count], f0Vec);

    count += 8;
  }
}

void NPME_Kfunc_Laplace_SR_DM::CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  const __m512d zeroVec = _mm512_set1_pd(0.0);

  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_SR_DM::CalcAVX_512\n";
    std::cout << "N = " << N << "must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop        = N/8;
  const __m512d _RdirVec512 = _mm512_set1_pd( _Rdir );

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

    __m512d f0_AVec, f1_AVec;
    __m512d f0_BVec, f1_BVec;
    NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (f0_AVec, f1_AVec, 
      r2Vec, _Nder, &_a[0], &_b[0]);
    NPME_Kfunc_Laplace_AVX_512 (f0_BVec, f1_BVec, rVec, r2Vec);

    f0_BVec = _mm512_sub_pd (f0_BVec, f0_AVec);
    f1_BVec = _mm512_sub_pd (f1_BVec, f1_AVec);

    __m512d f0Vec, f1Vec;
    //use (f0_AVec) if r < Rdir
    //use (f0_BVec) if r > Rdir
    {
      __mmask8 maskVec = _mm512_cmp_pd_mask (rVec, _RdirVec512, 1);
      f0Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f0_BVec);
      f1Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f1_BVec);
    }


    __m512d fXVec, fYVec, fZVec;

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
//***********************NPME_Kfunc_Laplace_LR_Original*************************
//******************************************************************************
//******************************************************************************

void NPME_Kfunc_Laplace_LR_Original::SetParm (const double beta)
{
  _beta     = beta;
  _beta3    = beta*beta*beta;
}

void NPME_Kfunc_Laplace_LR_Original::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Laplace_LR_Original::Print\n";
  sprintf(str, "  beta  = %.3f\n", _beta);    os << str;
  os.flush();
}

void NPME_Kfunc_Laplace_LR_Original::Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    double r2 = x_f0[i]*x_f0[i] + y[i]*y[i] + z[i]*z[i];
    double r  = sqrt(r2);
    x_f0[i]   = _beta*NPME_Berf_0 (r*_beta);
  }
}

void NPME_Kfunc_Laplace_LR_Original::Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX[i]*x_fX[i] + y_fY[i]*y_fY[i] + z_fZ[i]*z_fZ[i];
    const double r  = sqrt(r2);

    double B0, B1;
    B0 = NPME_Berf_1 (B1, _beta*r);

    f0[i]     = _beta*B0;
    double f1 = _beta3*B1;

    x_fX[i]         = x_fX[i]*f1;
    y_fY[i]         = y_fY[i]*f1;
    z_fZ[i]         = z_fZ[i]*f1;
  }
}


#if NPME_USE_AVX
//AVX intrinsic functions of above
void NPME_Kfunc_Laplace_LR_Original::CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_LR_Original::CalcAVX\n";
    std::cout << "N = " << N << "must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop              = N/4;
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


    __m256d B0_vec;
    NPME_Berf_0_AVX (B0_vec, _mm256_mul_pd  (_beta_Vec256, rVec));

    __m256d f0Vec = _mm256_mul_pd  (_beta_Vec256, B0_vec);

    _mm256_store_pd (&x_f0[count], f0Vec);

    count += 4;
  }
}

void NPME_Kfunc_Laplace_LR_Original::CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_LR_Original::CalcAVX\n";
    std::cout << "N = " << N << "must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop              = N/4;
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

    __m256d B0_vec, B1_vec;
    NPME_Berf_1_AVX (B0_vec, B1_vec, _mm256_mul_pd  (_beta_Vec256, rVec));

    //f0 = _beta*B0;
    f0Vec = _mm256_mul_pd  (_beta_Vec256, B0_vec);

    //f1 = _beta3*B1
    f1Vec = _mm256_mul_pd  (_beta3_Vec256, B1_vec);

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
void NPME_Kfunc_Laplace_LR_Original::CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_LR_Original::CalcAVX_512\n";
    std::cout << "N = " << N << "must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop              = N/8;
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
    __m512d B0_vec;
    NPME_Berf_0_AVX_512 (B0_vec, _mm512_mul_pd  (_beta_Vec512, rVec));

    f0Vec = _mm512_mul_pd  (_beta_Vec512, B0_vec);

    _mm512_store_pd (&x_f0[count], f0Vec);

    count += 8;
  }
}

void NPME_Kfunc_Laplace_LR_Original::CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_LR_Original::CalcAVX_512\n";
    std::cout << "N = " << N << "must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop              = N/8;
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
    __m512d B0_vec, B1_vec;
    NPME_Berf_1_AVX_512 (B0_vec, B1_vec, _mm512_mul_pd  (_beta_Vec512, rVec));

    //f0 = _beta*B0;
    f0Vec = _mm512_mul_pd  (_beta_Vec512, B0_vec);

    //f1 = _beta3*B1
    f1Vec = _mm512_mul_pd  (_beta3_Vec512, B1_vec);


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
//***********************NPME_Kfunc_Laplace_SR_Original*************************
//******************************************************************************
//******************************************************************************

void NPME_Kfunc_Laplace_SR_Original::SetParm (const double beta)
{
  _beta     = beta;
  _beta3    = beta*beta*beta;
}

void NPME_Kfunc_Laplace_SR_Original::Print (std::ostream& os) const
{
  char str[2000];
  os << "\nNPME_Kfunc_Laplace_SR_Original::Print\n";
  sprintf(str, "  beta  = %.3f\n", _beta);    os << str;
  os.flush();
}

void NPME_Kfunc_Laplace_SR_Original::Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  for (size_t i = 0; i < N; i++)
  {
    double r2 = x_f0[i]*x_f0[i] + y[i]*y[i] + z[i]*z[i];
    double r  = sqrt(r2);
    x_f0[i]   = _beta*NPME_Berfc_0 (r*_beta);
  }
}

void NPME_Kfunc_Laplace_SR_Original::Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX[i]*x_fX[i] + y_fY[i]*y_fY[i] + z_fZ[i]*z_fZ[i];
    const double r  = sqrt(r2);

    double B0, B1;
    B0 = NPME_Berfc_1 (B1, _beta*r);

    f0[i]     = _beta*B0;
    double f1 = _beta3*B1;

    x_fX[i]         = x_fX[i]*f1;
    y_fY[i]         = y_fY[i]*f1;
    z_fZ[i]         = z_fZ[i]*f1;
  }
}


#if NPME_USE_AVX
//AVX intrinsic functions of above
void NPME_Kfunc_Laplace_SR_Original::CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_SR_Original::CalcAVX\n";
    std::cout << "N = " << N << "must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop              = N/4;
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


    __m256d B0_vec;
    NPME_Berfc_0_AVX (B0_vec, _mm256_mul_pd  (_beta_Vec256, rVec));

    __m256d f0Vec = _mm256_mul_pd  (_beta_Vec256, B0_vec);

    _mm256_store_pd (&x_f0[count], f0Vec);

    count += 4;
  }
}

void NPME_Kfunc_Laplace_SR_Original::CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_SR_Original::CalcAVX\n";
    std::cout << "N = " << N << "must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop              = N/4;
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

    __m256d B0_vec, B1_vec;
    NPME_Berfc_1_AVX (B0_vec, B1_vec, _mm256_mul_pd  (_beta_Vec256, rVec));

    //f0 = _beta*B0;
    f0Vec = _mm256_mul_pd  (_beta_Vec256, B0_vec);

    //f1 = _beta3*B1
    f1Vec = _mm256_mul_pd  (_beta3_Vec256, B1_vec);

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
void NPME_Kfunc_Laplace_SR_Original::CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_SR_Original::CalcAVX_512\n";
    std::cout << "N = " << N << "must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop              = N/8;
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
    __m512d B0_vec;
    NPME_Berfc_0_AVX_512 (B0_vec, _mm512_mul_pd  (_beta_Vec512, rVec));

    f0Vec = _mm512_mul_pd  (_beta_Vec512, B0_vec);

    _mm512_store_pd (&x_f0[count], f0Vec);

    count += 8;
  }
}

void NPME_Kfunc_Laplace_SR_Original::CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Kfunc_Laplace_SR_Original::CalcAVX_512\n";
    std::cout << "N = " << N << "must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop              = N/8;
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
    __m512d B0_vec, B1_vec;
    NPME_Berfc_1_AVX_512 (B0_vec, B1_vec, _mm512_mul_pd  (_beta_Vec512, rVec));

    //f0 = _beta*B0;
    f0Vec = _mm512_mul_pd  (_beta_Vec512, B0_vec);

    //f1 = _beta3*B1
    f1Vec = _mm512_mul_pd  (_beta3_Vec512, B1_vec);


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



