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

#ifndef NPME_FUNCTION_DERIV_MATCH_H
#define NPME_FUNCTION_DERIV_MATCH_H

#include <stdlib.h>
#include <math.h>


#include <immintrin.h>

namespace NPME_Library
{



//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
//************************ Derivative Matching Functions***********************
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

void NPME_FunctionDerivMatch_HelmholtzRadialDeriv (_Complex double *f, 
  const int N, const double r, const _Complex double k0);
void NPME_FunctionDerivMatch_HelmholtzRadialDeriv_OLD (_Complex double *f, 
  const int N, const double r, const _Complex double k0);
//calculates f[N+1] where
//f[0] = cexp(I*k0*r)/r
//f[1] = (1/r d/dr)   f[0]
//f[n] = (1/r d/dr)^n f[0]

void NPME_FunctionDerivMatch_RalphaRadialDeriv (double *f, const int N, 
  const double r, const double alpha);
//calculates f[N+1] where
//f[0] = r^alpha
//f[1] = (1/r d/dr)   f[0]
//f[n] = (1/r d/dr)^n f[0]




bool NPME_FunctionDerivMatch_CalcEvenSeries (double *a, 
  double *b, const double *fRad, const int Nder, const double Rdir);
bool NPME_FunctionDerivMatch_CalcEvenSeries (_Complex double *a, 
  _Complex double *b, const _Complex double *fRad, 
  const int Nder, const double Rdir);
//input:  fRad[nDer+1] = (1/r d/dr)^n f0(r) at r = Rdir
//        n = 0, 1, .. Nder
//output: a[nDer+1]
//        b[nDer+1]



bool NPME_FunctionDerivMatch_CalcEvenSeries_Solve (double *a, 
  double *b, double *fRad, const int Nder, const double Rdir);
bool NPME_FunctionDerivMatch_CalcEvenSeries_Solve (_Complex double *a, 
  _Complex double *b, _Complex double *fRad, const int Nder, const double Rdir);
//input:  fRad[nDer+1] = (1/r d/dr)^n f0(r) at r = Rdir
//        n = 0, 1, .. Nder
//output: a[nDer+1]
//        b[nDer+1]



double NPME_FunctionDerivMatch_EvenSeriesReal (const int N, 
  const double *a, const double r2);
_Complex double NPME_FunctionDerivMatch_EvenSeriesComplex (const int N, 
  const _Complex double *a, const double r2);
//a[N+1], calculates 
//Sum{a[n]*r^2n } n = 0, .. N

double NPME_FunctionDerivMatch_EvenSeriesReal (double& f1, 
  const int N, const double *a, const double *b, 
  const double r2);
_Complex double NPME_FunctionDerivMatch_EvenSeriesComplex (_Complex double& f1, 
  const int N, const _Complex double *a, const _Complex double *b, 
  const double r2);
//a[N+1], calculates 
//f0 = Sum{a[n]*r^2n } n = 0, .. N
//f1 = 1/r d/dr f0(r)
//   = Sum{b[n]*r^2n } n = 0, .. N-1


#if NPME_USE_AVX

inline void NPME_FunctionDerivMatch_EvenSeriesReal_AVX (
  __m256d& f0Vec, const __m256d& r2Vec, const int N, const double *a)
//a[N+1]
//returns Sum {a[n]*x^n} n = 0, 1, .. N
{
  __m256d a1Vec, a2Vec;
  a1Vec = _mm256_set1_pd(a[N-1]);
  a2Vec = _mm256_set1_pd(a[N]);

  //f0Vec  = a2Vec*r2Vec + a1Vec
  #if NPME_USE_AVX_FMA
  {
    f0Vec = _mm256_fmadd_pd (a2Vec, r2Vec, a1Vec);
  }
  #else
  {
    f0Vec = _mm256_add_pd (_mm256_mul_pd (a2Vec, r2Vec), a1Vec);
  }
  #endif

  for (int k = N - 2; k >= 0; k--)
  {
    a1Vec = _mm256_set1_pd(a[k]);

    //f0Vec  = a2Vec*r2Vec + a1Vec
    #if NPME_USE_AVX_FMA
    {
      f0Vec = _mm256_fmadd_pd (f0Vec, r2Vec, a1Vec);
    }
    #else
    {
      f0Vec = _mm256_add_pd (_mm256_mul_pd (f0Vec, r2Vec), a1Vec);
    }
    #endif
  }
}

inline void NPME_FunctionDerivMatch_EvenSeriesReal_AVX (
  __m256d& f0Vec, __m256d& f1Vec, const __m256d& r2Vec, 
  const int N, const double *a, const double *b)
//a[N+1], calculates 
//f0 = Sum{a[n]*r^2n } n = 0, .. N
//f1 = 1/r d/dr f0(r)
//   = Sum{b[n]*r^2n } n = 0, .. N-1
{
  __m256d a1Vec, a2Vec;

  //f0rVec and f0iVec
  a1Vec = _mm256_set1_pd(a[N-1]);
  a2Vec = _mm256_set1_pd(a[N]);
  #if NPME_USE_AVX_FMA
  {
    f0Vec = _mm256_fmadd_pd (a2Vec, r2Vec, a1Vec);
  }
  #else
  {
    f0Vec = _mm256_add_pd (_mm256_mul_pd (a2Vec, r2Vec), a1Vec);
  }
  #endif
  for (int k = N - 2; k >= 0; k--)
  {
    a1Vec = _mm256_set1_pd(a[k]);

    //f0Vec  = a2Vec*r2Vec + a1Vec
    #if NPME_USE_AVX_FMA
    {
      f0Vec = _mm256_fmadd_pd (f0Vec, r2Vec, a1Vec);
    }
    #else
    {
      f0Vec = _mm256_add_pd (_mm256_mul_pd (f0Vec, r2Vec), a1Vec);
    }
    #endif
  }

  //f1rVec and f1iVec
  a1Vec = _mm256_set1_pd(b[N-2]);
  a2Vec = _mm256_set1_pd(b[N-1]);
  #if NPME_USE_AVX_FMA
  {
    f1Vec = _mm256_fmadd_pd (a2Vec, r2Vec, a1Vec);
  }
  #else
  {
    f1Vec = _mm256_add_pd (_mm256_mul_pd (a2Vec, r2Vec), a1Vec);
  }
  #endif
  for (int k = N - 3; k >= 0; k--)
  {
    a1Vec = _mm256_set1_pd(b[k]);

    #if NPME_USE_AVX_FMA
    {
      f1Vec = _mm256_fmadd_pd (f1Vec, r2Vec, a1Vec);
    }
    #else
    {
      f1Vec = _mm256_add_pd (_mm256_mul_pd (f1Vec, r2Vec), a1Vec);
    }
    #endif
  }

}


inline void NPME_FunctionDerivMatch_EvenSeriesComplex_AVX (
  __m256d& f0rVec, __m256d& f0iVec, 
  const __m256d& r2Vec, const int N, const _Complex double *a)
//a[N+1]
//returns Sum {a[n]*x^n} n = 0, 1, .. N
{
  __m256d a1RVec, a1IVec, a2RVec, a2IVec;

  a1RVec = _mm256_set1_pd(creal(a[N-1]));
  a1IVec = _mm256_set1_pd(cimag(a[N-1]));
  a2RVec = _mm256_set1_pd(creal(a[N]));
  a2IVec = _mm256_set1_pd(cimag(a[N]));


  //f0Vec  = a2Vec*r2Vec + a1Vec
  //f0rVec = a2RVec*r2Vec + a1RVec
  //f0iVec = a2IVec*r2Vec + a1IVec

  #if NPME_USE_AVX_FMA
  {
    f0rVec = _mm256_fmadd_pd (a2RVec, r2Vec, a1RVec);
    f0iVec = _mm256_fmadd_pd (a2IVec, r2Vec, a1IVec);
  }
  #else
  {
    f0rVec = _mm256_add_pd (_mm256_mul_pd (a2RVec, r2Vec), a1RVec);
    f0iVec = _mm256_add_pd (_mm256_mul_pd (a2IVec, r2Vec), a1IVec);
  }
  #endif

  for (int k = N - 2; k >= 0; k--)
  {
    a1RVec = _mm256_set1_pd(creal(a[k]));
    a1IVec = _mm256_set1_pd(cimag(a[k]));

    //f0Vec  = a2Vec*r2Vec + a1Vec
    //f0rVec = a2RVec*r2Vec + a1RVec
    //f0iVec = a2IVec*r2Vec + a1IVec

    #if NPME_USE_AVX_FMA
    {
      f0rVec = _mm256_fmadd_pd (f0rVec, r2Vec, a1RVec);
      f0iVec = _mm256_fmadd_pd (f0iVec, r2Vec, a1IVec);
    }
    #else
    {
      f0rVec = _mm256_add_pd (_mm256_mul_pd (f0rVec, r2Vec), a1RVec);
      f0iVec = _mm256_add_pd (_mm256_mul_pd (f0iVec, r2Vec), a1IVec);
    }
    #endif
  }
}


inline void NPME_FunctionDerivMatch_EvenSeriesComplex_AVX (
  __m256d& f0rVec, __m256d& f0iVec, 
  __m256d& f1rVec, __m256d& f1iVec, const __m256d& r2Vec, 
  const int N, const _Complex double *a, const _Complex double *b)
//a[N+1], calculates 
//f0 = Sum{a[n]*r^2n } n = 0, .. N
//f1 = 1/r d/dr f0(r)
//   = Sum{b[n]*r^2n } n = 0, .. N-1
{
  __m256d a1RVec, a1IVec, a2RVec, a2IVec;

  //f0rVec and f0iVec
  a1RVec = _mm256_set1_pd(creal(a[N-1]));
  a1IVec = _mm256_set1_pd(cimag(a[N-1]));
  a2RVec = _mm256_set1_pd(creal(a[N]));
  a2IVec = _mm256_set1_pd(cimag(a[N]));
  #if NPME_USE_AVX_FMA
  {
    f0rVec = _mm256_fmadd_pd (a2RVec, r2Vec, a1RVec);
    f0iVec = _mm256_fmadd_pd (a2IVec, r2Vec, a1IVec);
  }
  #else
  {
    f0rVec = _mm256_add_pd (_mm256_mul_pd (a2RVec, r2Vec), a1RVec);
    f0iVec = _mm256_add_pd (_mm256_mul_pd (a2IVec, r2Vec), a1IVec);
  }
  #endif
  for (int k = N - 2; k >= 0; k--)
  {
    a1RVec = _mm256_set1_pd(creal(a[k]));
    a1IVec = _mm256_set1_pd(cimag(a[k]));

    //f0Vec  = a2Vec*r2Vec + a1Vec
    //f0rVec = a2RVec*r2Vec + a1RVec
    //f0iVec = a2IVec*r2Vec + a1IVec

    #if NPME_USE_AVX_FMA
    {
      f0rVec = _mm256_fmadd_pd (f0rVec, r2Vec, a1RVec);
      f0iVec = _mm256_fmadd_pd (f0iVec, r2Vec, a1IVec);
    }
    #else
    {
      f0rVec = _mm256_add_pd (_mm256_mul_pd (f0rVec, r2Vec), a1RVec);
      f0iVec = _mm256_add_pd (_mm256_mul_pd (f0iVec, r2Vec), a1IVec);
    }
    #endif
  }

  //f1rVec and f1iVec
  a1RVec = _mm256_set1_pd(creal(b[N-2]));
  a1IVec = _mm256_set1_pd(cimag(b[N-2]));
  a2RVec = _mm256_set1_pd(creal(b[N-1]));
  a2IVec = _mm256_set1_pd(cimag(b[N-1]));
  #if NPME_USE_AVX_FMA
  {
    f1rVec = _mm256_fmadd_pd (a2RVec, r2Vec, a1RVec);
    f1iVec = _mm256_fmadd_pd (a2IVec, r2Vec, a1IVec);
  }
  #else
  {
    f1rVec = _mm256_add_pd (_mm256_mul_pd (a2RVec, r2Vec), a1RVec);
    f1iVec = _mm256_add_pd (_mm256_mul_pd (a2IVec, r2Vec), a1IVec);
  }
  #endif
  for (int k = N - 3; k >= 0; k--)
  {
    a1RVec = _mm256_set1_pd(creal(b[k]));
    a1IVec = _mm256_set1_pd(cimag(b[k]));

    #if NPME_USE_AVX_FMA
    {
      f1rVec = _mm256_fmadd_pd (f1rVec, r2Vec, a1RVec);
      f1iVec = _mm256_fmadd_pd (f1iVec, r2Vec, a1IVec);
    }
    #else
    {
      f1rVec = _mm256_add_pd (_mm256_mul_pd (f1rVec, r2Vec), a1RVec);
      f1iVec = _mm256_add_pd (_mm256_mul_pd (f1iVec, r2Vec), a1IVec);
    }
    #endif
  }

}
#endif




#if NPME_USE_AVX_512

inline void NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (
  __m512d& f0Vec, const __m512d& r2Vec, const int N, const double *a)
//a[N+1]
//returns Sum {a[n]*x^n} n = 0, 1, .. N
{
  __m512d a1Vec, a2Vec;
  a1Vec = _mm512_set1_pd(a[N-1]);
  a2Vec = _mm512_set1_pd(a[N]);

  //f0Vec  = a2Vec*r2Vec + a1Vec
  f0Vec = _mm512_fmadd_pd (a2Vec, r2Vec, a1Vec);

  for (int k = N - 2; k >= 0; k--)
  {
    a1Vec = _mm512_set1_pd(a[k]);
    //f0Vec  = a2Vec*r2Vec + a1Vec
    f0Vec = _mm512_fmadd_pd (f0Vec, r2Vec, a1Vec);
  }
}
inline void NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (
  __m512d& f0Vec, __m512d& f1Vec, const __m512d& r2Vec, 
  const int N, const double *a, const double *b)
//a[N+1], calculates 
//f0 = Sum{a[n]*r^2n } n = 0, .. N
//f1 = 1/r d/dr f0(r)
//   = Sum{b[n]*r^2n } n = 0, .. N-1
{
  __m512d a1Vec, a2Vec;

  //f0rVec and f0iVec
  a1Vec = _mm512_set1_pd(a[N-1]);
  a2Vec = _mm512_set1_pd(a[N]);
  f0Vec = _mm512_fmadd_pd (a2Vec, r2Vec, a1Vec);

  for (int k = N - 2; k >= 0; k--)
  {
    a1Vec = _mm512_set1_pd(a[k]);
    //f0Vec  = a2Vec*r2Vec + a1Vec
    f0Vec = _mm512_fmadd_pd (f0Vec, r2Vec, a1Vec);
  }

  //f1rVec and f1iVec
  a1Vec = _mm512_set1_pd(b[N-2]);
  a2Vec = _mm512_set1_pd(b[N-1]);
  f1Vec = _mm512_fmadd_pd (a2Vec, r2Vec, a1Vec);

  for (int k = N - 3; k >= 0; k--)
  {
    a1Vec = _mm512_set1_pd(b[k]);
    f1Vec = _mm512_fmadd_pd (f1Vec, r2Vec, a1Vec);
  }
}

inline void NPME_FunctionDerivMatch_EvenSeriesComplex_AVX_512 (
  __m512d& f0rVec, __m512d& f0iVec, 
  const __m512d& r2Vec, const int N, const _Complex double *a)
//a[N+1]
//returns Sum {a[n]*x^n} n = 0, 1, .. N
{
  __m512d a1RVec, a1IVec, a2RVec, a2IVec;

  a1RVec = _mm512_set1_pd(creal(a[N-1]));
  a1IVec = _mm512_set1_pd(cimag(a[N-1]));
  a2RVec = _mm512_set1_pd(creal(a[N]));
  a2IVec = _mm512_set1_pd(cimag(a[N]));

  f0rVec = _mm512_fmadd_pd (a2RVec, r2Vec, a1RVec);
  f0iVec = _mm512_fmadd_pd (a2IVec, r2Vec, a1IVec);

  for (int k = N - 2; k >= 0; k--)
  {
    a1RVec = _mm512_set1_pd(creal(a[k]));
    a1IVec = _mm512_set1_pd(cimag(a[k]));
    f0rVec = _mm512_fmadd_pd (f0rVec, r2Vec, a1RVec);
    f0iVec = _mm512_fmadd_pd (f0iVec, r2Vec, a1IVec);
  }
}


inline void NPME_FunctionDerivMatch_EvenSeriesComplex_AVX_512 (
  __m512d& f0rVec, __m512d& f0iVec, 
  __m512d& f1rVec, __m512d& f1iVec, const __m512d& r2Vec, 
  const int N, const _Complex double *a, const _Complex double *b)
//a[N+1], calculates 
//f0 = Sum{a[n]*r^2n } n = 0, .. N
//f1 = 1/r d/dr f0(r)
//   = Sum{b[n]*r^2n } n = 0, .. N-1
{
  __m512d a1RVec, a1IVec, a2RVec, a2IVec;

  //f0rVec and f0iVec
  a1RVec = _mm512_set1_pd(creal(a[N-1]));
  a1IVec = _mm512_set1_pd(cimag(a[N-1]));
  a2RVec = _mm512_set1_pd(creal(a[N]));
  a2IVec = _mm512_set1_pd(cimag(a[N]));

  f0rVec = _mm512_fmadd_pd (a2RVec, r2Vec, a1RVec);
  f0iVec = _mm512_fmadd_pd (a2IVec, r2Vec, a1IVec);

  for (int k = N - 2; k >= 0; k--)
  {
    a1RVec = _mm512_set1_pd(creal(a[k]));
    a1IVec = _mm512_set1_pd(cimag(a[k]));
    f0rVec = _mm512_fmadd_pd (f0rVec, r2Vec, a1RVec);
    f0iVec = _mm512_fmadd_pd (f0iVec, r2Vec, a1IVec);
  }

  //f1rVec and f1iVec
  a1RVec = _mm512_set1_pd(creal(b[N-2]));
  a1IVec = _mm512_set1_pd(cimag(b[N-2]));
  a2RVec = _mm512_set1_pd(creal(b[N-1]));
  a2IVec = _mm512_set1_pd(cimag(b[N-1]));

  f1rVec = _mm512_fmadd_pd (a2RVec, r2Vec, a1RVec);
  f1iVec = _mm512_fmadd_pd (a2IVec, r2Vec, a1IVec);

  for (int k = N - 3; k >= 0; k--)
  {
    a1RVec = _mm512_set1_pd(creal(b[k]));
    a1IVec = _mm512_set1_pd(cimag(b[k]));

    f1rVec = _mm512_fmadd_pd (f1rVec, r2Vec, a1RVec);
    f1iVec = _mm512_fmadd_pd (f1iVec, r2Vec, a1IVec);
  }

}
#endif

}//end namespace NPME_Library


#endif // NPME_FUNCTION_DERIV_MATCH_H


