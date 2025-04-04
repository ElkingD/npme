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

#ifndef NPME_MATH_FUNCTIONS_H
#define NPME_MATH_FUNCTIONS_H

#include <iostream> 
#include <immintrin.h>

#include "NPME_VectorIntrinsic.h"

namespace NPME_Library
{
//***************Scalar Math Functions Used in Ewald Splitting*****************
double NPME_Berf_0 (const double x);
double NPME_Berf_1 (double& B1, const double x);
//B0 = erf(x)/x
//B1 = 1/x dB0/dx

double NPME_Berfc_0 (const double x);
double NPME_Berfc_1 (double& B1, const double x);
//B0 = erfc(x)/x
//B1 = 1/x d/dx B0



double NPME_sinx_x (const double x);
//returns sin(x)/x

double NPME_sinx_x (double& f1, const double x);
//returns sin(x)/x
//f1 = 1/x d/dx sin(x)/x

void NPME_SphereHankel (_Complex double *h,
  const int n, const _Complex double z);
//calculates spherical Hankel functions (first or second order) 
//by upward recursion
//input:  z = x + I*y
//        n = max order
//output: h[n+1] = {h0(z), h1(z), .. hn(z)}
//        h = h1 if y >= 0
//        h = h2 if y <  0

double NPME_IntPow (double x, int n);
//calculates x^n (n = integer)

double NPME_Factorial (int n);

double NPME_Factorial2 (int n, int m);
//n!/m!



#if NPME_USE_AVX
inline void NPME_Berf_0_AVX (__m256d& B0_vec, const __m256d& xVec)
{
  const __m256d C0_0_Vec = _mm256_set1_pd( 1.128379167095512574);
  const __m256d C0_2_Vec = _mm256_set1_pd(-3.761263890318375246E-1);
  const __m256d C0_4_Vec = _mm256_set1_pd( 1.128379167095512574E-1);
  const __m256d C0_6_Vec = _mm256_set1_pd(-2.686617064513125176E-2);
  const __m256d C0_8_Vec = _mm256_set1_pd( 5.223977625442187842E-3);
  const __m256d xMaxVec  = _mm256_set1_pd(0.1);

  const __m256d x2Vec    = _mm256_mul_pd (xVec, xVec);


  __m256d B0_vecA, B0_vecB;
  #if NPME_USE_AVX_FMA
  {
    B0_vecA = _mm256_fmadd_pd (C0_8_Vec, x2Vec, C0_6_Vec);
    B0_vecA = _mm256_fmadd_pd (B0_vecA,   x2Vec, C0_4_Vec);
    B0_vecA = _mm256_fmadd_pd (B0_vecA,   x2Vec, C0_2_Vec);
    B0_vecA = _mm256_fmadd_pd (B0_vecA,   x2Vec, C0_0_Vec);
  }
  #else
  {
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (C0_8_Vec, x2Vec), C0_6_Vec);
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (B0_vecA,   x2Vec), C0_4_Vec);
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (B0_vecA,   x2Vec), C0_2_Vec);
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (B0_vecA,   x2Vec), C0_0_Vec);
  }
  #endif

  B0_vecB = _mm256_erf_pd (xVec);
  B0_vecB = _mm256_div_pd (B0_vecB, xVec);

  //use (B0_vecA) if x < xMax
  //use (B0_vecB) if x > xMax
  {
    __m256d t0, dless, dmore;
    t0      = _mm256_cmp_pd (xVec, xMaxVec, 1);

    dless   = _mm256_and_pd    (t0, B0_vecA);
    dmore   = _mm256_andnot_pd (t0, B0_vecB);
    B0_vec  = _mm256_add_pd (dless, dmore);
  }
}


inline void NPME_Berf_1_AVX (__m256d& B0_vec, __m256d& B1_vec, 
  const __m256d& xVec)
{
  const __m256d xMaxVec  = _mm256_set1_pd(0.1);
  const __m256d nOneVec  = _mm256_set1_pd(-1.0);

  const __m256d C0_0_Vec = _mm256_set1_pd( 1.128379167095512574);
  const __m256d C0_2_Vec = _mm256_set1_pd(-3.761263890318375246E-1);
  const __m256d C0_4_Vec = _mm256_set1_pd( 1.128379167095512574E-1);
  const __m256d C0_6_Vec = _mm256_set1_pd(-2.686617064513125176E-2);
  const __m256d C0_8_Vec = _mm256_set1_pd( 5.223977625442187842E-3);

  const __m256d C1_0_Vec = _mm256_set1_pd(-7.522527780636750492E-1);
  const __m256d C1_2_Vec = _mm256_set1_pd( 4.513516668382050296E-1);
  const __m256d C1_4_Vec = _mm256_set1_pd(-1.611970238707875106E-1);
  const __m256d C1_6_Vec = _mm256_set1_pd( 4.179182100353750274E-2);
  const __m256d C1_8_Vec = _mm256_set1_pd(-8.548327023450852833E-3);

  const __m256d x2Vec    = _mm256_mul_pd (xVec, xVec);


  __m256d B0_vecA, B0_vecB, B1_vecA, B1_vecB;
  #if NPME_USE_AVX_FMA
  {
    B0_vecA = _mm256_fmadd_pd (C0_8_Vec,  x2Vec, C0_6_Vec);
    B0_vecA = _mm256_fmadd_pd (B0_vecA,   x2Vec, C0_4_Vec);
    B0_vecA = _mm256_fmadd_pd (B0_vecA,   x2Vec, C0_2_Vec);
    B0_vecA = _mm256_fmadd_pd (B0_vecA,   x2Vec, C0_0_Vec);

    B1_vecA = _mm256_fmadd_pd (C1_8_Vec,  x2Vec, C1_6_Vec);
    B1_vecA = _mm256_fmadd_pd (B1_vecA,   x2Vec, C1_4_Vec);
    B1_vecA = _mm256_fmadd_pd (B1_vecA,   x2Vec, C1_2_Vec);
    B1_vecA = _mm256_fmadd_pd (B1_vecA,   x2Vec, C1_0_Vec);
  }
  #else
  {
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (C0_8_Vec,  x2Vec), C0_6_Vec);
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (B0_vecA,   x2Vec), C0_4_Vec);
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (B0_vecA,   x2Vec), C0_2_Vec);
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (B0_vecA,   x2Vec), C0_0_Vec);

    B1_vecA = _mm256_add_pd (_mm256_mul_pd (C1_8_Vec,  x2Vec), C1_6_Vec);
    B1_vecA = _mm256_add_pd (_mm256_mul_pd (B1_vecA,   x2Vec), C1_4_Vec);
    B1_vecA = _mm256_add_pd (_mm256_mul_pd (B1_vecA,   x2Vec), C1_2_Vec);
    B1_vecA = _mm256_add_pd (_mm256_mul_pd (B1_vecA,   x2Vec), C1_0_Vec);
  }
  #endif

  B0_vecB = _mm256_erf_pd (xVec);
  B0_vecB = _mm256_div_pd (B0_vecB, xVec);


  __m256d expVec = _mm256_exp_pd ( _mm256_mul_pd(nOneVec, x2Vec) );
  #if NPME_USE_AVX_FMA
  {
    B1_vecB = _mm256_fmsub_pd (C0_0_Vec,  expVec, B0_vecB);
  }
  #else
  {
    B1_vecB = _mm256_sub_pd (_mm256_mul_pd (C0_0_Vec,  expVec), B0_vecB);
  }
  #endif
  B1_vecB = _mm256_div_pd (B1_vecB, x2Vec);


  //use (B0_vecA) if x < xMax
  //use (B0_vecB) if x > xMax
  {
    __m256d t0, dless, dmore;
    t0      = _mm256_cmp_pd (xVec, xMaxVec, 1);

    dless   = _mm256_and_pd    (t0, B0_vecA);
    dmore   = _mm256_andnot_pd (t0, B0_vecB);
    B0_vec  = _mm256_add_pd (dless, dmore);

    dless   = _mm256_and_pd    (t0, B1_vecA);
    dmore   = _mm256_andnot_pd (t0, B1_vecB);
    B1_vec  = _mm256_add_pd (dless, dmore);
  }
}


inline void NPME_Berfc_0_AVX (__m256d& B0_vec, const __m256d& xVec)
{
  B0_vec = _mm256_erfc_pd (xVec);
  B0_vec = _mm256_div_pd (B0_vec, xVec);
}


inline void NPME_Berfc_1_AVX (__m256d& B0_vec, __m256d& B1_vec, 
  const __m256d& xVec)
{
  const __m256d C0_0_Vec = _mm256_set1_pd(-1.128379167095512574);//-2/sqrt(Pi)
  const __m256d nOneVec  = _mm256_set1_pd(-1.0);
  const __m256d x2Vec    = _mm256_mul_pd (xVec, xVec);

  B0_vec = _mm256_erfc_pd (xVec);
  B0_vec = _mm256_div_pd (B0_vec, xVec);

  //B1 = (-B0 - C0_0*exp(-x2))/x2;
  __m256d expVec = _mm256_exp_pd ( _mm256_mul_pd(nOneVec, x2Vec) );

  #if NPME_USE_AVX_FMA
  {
    B1_vec = _mm256_fmsub_pd (C0_0_Vec, expVec, B0_vec);
  }
  #else
  {
    B1_vec = _mm256_sub_pd (_mm256_mul_pd (C0_0_Vec, expVec), B0_vec);
  }
  #endif
  B1_vec = _mm256_div_pd (B1_vec, x2Vec);
}







inline void NPME_sinx_x_AVX (__m256d& B0_vec, __m256d& cos_vec, 
  const __m256d& xVec)
//B0_vec[4] = sin(xVec[4])/xVec[4]
{
  const __m256d C0_0_Vec = _mm256_set1_pd( 1.000000000000000000);
  const __m256d C0_2_Vec = _mm256_set1_pd(-1.666666666666666667E-1);
  const __m256d C0_4_Vec = _mm256_set1_pd( 8.333333333333333333E-3);
  const __m256d C0_6_Vec = _mm256_set1_pd(-1.984126984126984127E-4);
  const __m256d C0_8_Vec = _mm256_set1_pd( 2.755731922398589065E-6);
  const __m256d xMaxVec  = _mm256_set1_pd(1.0E-1);

  const __m256d x2Vec    = _mm256_mul_pd (xVec, xVec);


  __m256d B0_vecA, B0_vecB;
  #if NPME_USE_AVX_FMA
  {
    B0_vecA = _mm256_fmadd_pd (C0_8_Vec, x2Vec, C0_6_Vec);
    B0_vecA = _mm256_fmadd_pd (B0_vecA,   x2Vec, C0_4_Vec);
    B0_vecA = _mm256_fmadd_pd (B0_vecA,   x2Vec, C0_2_Vec);
    B0_vecA = _mm256_fmadd_pd (B0_vecA,   x2Vec, C0_0_Vec);
  }
  #else
  {
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (C0_8_Vec, x2Vec), C0_6_Vec);
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (B0_vecA,   x2Vec), C0_4_Vec);
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (B0_vecA,   x2Vec), C0_2_Vec);
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (B0_vecA,   x2Vec), C0_0_Vec);
  }
  #endif

  __m256d sin_vec;
  sin_vec = _mm256_sincos_pd (&cos_vec, xVec);
  B0_vecB = _mm256_div_pd (sin_vec, xVec);


  //use (B0_vecA) if x < xMax
  //use (B0_vecB) if x > xMax
  {
    __m256d t0, dless, dmore;
    t0      = _mm256_cmp_pd (xVec, xMaxVec, 1);

    dless   = _mm256_and_pd    (t0, B0_vecA);
    dmore   = _mm256_andnot_pd (t0, B0_vecB);
    B0_vec  = _mm256_add_pd (dless, dmore);
  }
}

inline void NPME_sinx_x_AVX (__m256d& B0_vec, __m256d& B1_vec, 
  __m256d& cos_vec, const __m256d& xVec)
//B0_vec[4] = sin(xVec[4])/xVec[4]
//B1_vec[4] = 1/x d/dx B0_vec[4]
{
  const __m256d C0_0_Vec = _mm256_set1_pd( 1.000000000000000000);
  const __m256d C0_2_Vec = _mm256_set1_pd(-1.666666666666666667E-1);
  const __m256d C0_4_Vec = _mm256_set1_pd( 8.333333333333333333E-3);
  const __m256d C0_6_Vec = _mm256_set1_pd(-1.984126984126984127E-4);
  const __m256d C0_8_Vec = _mm256_set1_pd( 2.755731922398589065E-6);

  const __m256d C1_0_Vec = _mm256_set1_pd(-3.333333333333333333E-1);
  const __m256d C1_2_Vec = _mm256_set1_pd( 3.333333333333333333E-2);
  const __m256d C1_4_Vec = _mm256_set1_pd(-1.190476190476190476E-3);
  const __m256d C1_6_Vec = _mm256_set1_pd( 2.204585537918871252E-5);



  const __m256d xMaxVec  = _mm256_set1_pd(1.0E-1);

  const __m256d x2Vec    = _mm256_mul_pd (xVec, xVec);


  __m256d B0_vecA, B0_vecB, B1_vecA, B1_vecB;
  #if NPME_USE_AVX_FMA
  {
    B0_vecA = _mm256_fmadd_pd (C0_8_Vec, x2Vec, C0_6_Vec);
    B0_vecA = _mm256_fmadd_pd (B0_vecA,   x2Vec, C0_4_Vec);
    B0_vecA = _mm256_fmadd_pd (B0_vecA,   x2Vec, C0_2_Vec);
    B0_vecA = _mm256_fmadd_pd (B0_vecA,   x2Vec, C0_0_Vec);

    B1_vecA = _mm256_fmadd_pd (C1_6_Vec,  x2Vec, C1_4_Vec);
    B1_vecA = _mm256_fmadd_pd (B1_vecA,   x2Vec, C1_2_Vec);
    B1_vecA = _mm256_fmadd_pd (B1_vecA,   x2Vec, C1_0_Vec);
  }
  #else
  {
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (C0_8_Vec, x2Vec), C0_6_Vec);
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (B0_vecA,   x2Vec), C0_4_Vec);
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (B0_vecA,   x2Vec), C0_2_Vec);
    B0_vecA = _mm256_add_pd (_mm256_mul_pd (B0_vecA,   x2Vec), C0_0_Vec);

    B1_vecA = _mm256_add_pd (_mm256_mul_pd (C1_6_Vec,  x2Vec), C1_4_Vec);
    B1_vecA = _mm256_add_pd (_mm256_mul_pd (B1_vecA,   x2Vec), C1_2_Vec);
    B1_vecA = _mm256_add_pd (_mm256_mul_pd (B1_vecA,   x2Vec), C1_0_Vec);
  }
  #endif

  __m256d sin_vec;
  sin_vec = _mm256_sincos_pd (&cos_vec, xVec);
  B0_vecB = _mm256_div_pd (sin_vec, xVec);

  B1_vecB = _mm256_sub_pd (cos_vec,  B0_vecB);
  B1_vecB = _mm256_div_pd (B1_vecB, x2Vec);


  //use (B0_vecA) if x < xMax
  //use (B0_vecB) if x > xMax
  {
    __m256d t0, dless, dmore;
    t0      = _mm256_cmp_pd (xVec, xMaxVec, 1);

    dless   = _mm256_and_pd    (t0, B0_vecA);
    dmore   = _mm256_andnot_pd (t0, B0_vecB);
    B0_vec  = _mm256_add_pd (dless, dmore);

    dless   = _mm256_and_pd    (t0, B1_vecA);
    dmore   = _mm256_andnot_pd (t0, B1_vecB);
    B1_vec  = _mm256_add_pd (dless, dmore);
  }
}



#endif

#if NPME_USE_AVX_512
inline void NPME_Berf_0_AVX_512 (__m512d& B0_vec, const __m512d& xVec)
{
  const __m512d C0_0_Vec = _mm512_set1_pd( 1.128379167095512574);
  const __m512d C0_2_Vec = _mm512_set1_pd(-3.761263890318375246E-1);
  const __m512d C0_4_Vec = _mm512_set1_pd( 1.128379167095512574E-1);
  const __m512d C0_6_Vec = _mm512_set1_pd(-2.686617064513125176E-2);
  const __m512d C0_8_Vec = _mm512_set1_pd( 5.223977625442187842E-3);
  const __m512d xMaxVec  = _mm512_set1_pd(0.1);

  const __m512d x2Vec    = _mm512_mul_pd (xVec, xVec);


  __m512d B0_vecA, B0_vecB;
  B0_vecA = _mm512_fmadd_pd (C0_8_Vec, x2Vec, C0_6_Vec);
  B0_vecA = _mm512_fmadd_pd (B0_vecA,   x2Vec, C0_4_Vec);
  B0_vecA = _mm512_fmadd_pd (B0_vecA,   x2Vec, C0_2_Vec);
  B0_vecA = _mm512_fmadd_pd (B0_vecA,   x2Vec, C0_0_Vec);


  B0_vecB = _mm512_erf_pd (xVec);
  B0_vecB = _mm512_div_pd (B0_vecB, xVec);

  //use B0_vecA if r < Rdir
  //use B0_vecB if r > Rdir
  {
    __mmask8 maskVec = _mm512_cmp_pd_mask (xVec, xMaxVec, 1);
    B0_vec = _mm512_mask_mov_pd (B0_vecB, maskVec, B0_vecA);
  }
}


inline void NPME_Berf_1_AVX_512 (__m512d& B0_vec, __m512d& B1_vec, 
  const __m512d& xVec)
{
  const __m512d xMaxVec  = _mm512_set1_pd(0.1);
  const __m512d nOneVec  = _mm512_set1_pd(-1.0);

  const __m512d C0_0_Vec = _mm512_set1_pd( 1.128379167095512574);
  const __m512d C0_2_Vec = _mm512_set1_pd(-3.761263890318375246E-1);
  const __m512d C0_4_Vec = _mm512_set1_pd( 1.128379167095512574E-1);
  const __m512d C0_6_Vec = _mm512_set1_pd(-2.686617064513125176E-2);
  const __m512d C0_8_Vec = _mm512_set1_pd( 5.223977625442187842E-3);

  const __m512d C1_0_Vec = _mm512_set1_pd(-7.522527780636750492E-1);
  const __m512d C1_2_Vec = _mm512_set1_pd( 4.513516668382050296E-1);
  const __m512d C1_4_Vec = _mm512_set1_pd(-1.611970238707875106E-1);
  const __m512d C1_6_Vec = _mm512_set1_pd( 4.179182100353750274E-2);
  const __m512d C1_8_Vec = _mm512_set1_pd(-8.548327023450852833E-3);

  const __m512d x2Vec    = _mm512_mul_pd (xVec, xVec);


  __m512d B0_vecA, B0_vecB, B1_vecA, B1_vecB;
  B0_vecA = _mm512_fmadd_pd (C0_8_Vec,  x2Vec, C0_6_Vec);
  B0_vecA = _mm512_fmadd_pd (B0_vecA,   x2Vec, C0_4_Vec);
  B0_vecA = _mm512_fmadd_pd (B0_vecA,   x2Vec, C0_2_Vec);
  B0_vecA = _mm512_fmadd_pd (B0_vecA,   x2Vec, C0_0_Vec);

  B1_vecA = _mm512_fmadd_pd (C1_8_Vec,  x2Vec, C1_6_Vec);
  B1_vecA = _mm512_fmadd_pd (B1_vecA,   x2Vec, C1_4_Vec);
  B1_vecA = _mm512_fmadd_pd (B1_vecA,   x2Vec, C1_2_Vec);
  B1_vecA = _mm512_fmadd_pd (B1_vecA,   x2Vec, C1_0_Vec);


  B0_vecB = _mm512_erf_pd (xVec);
  B0_vecB = _mm512_div_pd (B0_vecB, xVec);


  __m512d expVec = _mm512_exp_pd ( _mm512_mul_pd(nOneVec, x2Vec) );
  B1_vecB = _mm512_fmsub_pd (C0_0_Vec,  expVec, B0_vecB);

  B1_vecB = _mm512_div_pd (B1_vecB, x2Vec);


  //use B0_vecA if r < Rdir
  //use B0_vecB if r > Rdir
  {
    __mmask8 maskVec = _mm512_cmp_pd_mask (xVec, xMaxVec, 1);
    B0_vec = _mm512_mask_mov_pd (B0_vecB, maskVec, B0_vecA);
    B1_vec = _mm512_mask_mov_pd (B1_vecB, maskVec, B1_vecA);
  }
}


inline void NPME_Berfc_0_AVX_512 (__m512d& B0_vec, const __m512d& xVec)
{
  B0_vec = _mm512_erfc_pd (xVec);
  B0_vec = _mm512_div_pd (B0_vec, xVec);
}

inline void NPME_Berfc_1_AVX_512 (__m512d& B0_vec, __m512d& B1_vec, 
  const __m512d& xVec)
{
  const __m512d C0_0_Vec = _mm512_set1_pd(-1.128379167095512574);//-2/sqrt(Pi)
  const __m512d nOneVec  = _mm512_set1_pd(-1.0);
  const __m512d x2Vec    = _mm512_mul_pd (xVec, xVec);

  B0_vec = _mm512_erfc_pd (xVec);
  B0_vec = _mm512_div_pd (B0_vec, xVec);

  //B1 = (-B0 - C0_0*exp(-x2))/x2;
  __m512d expVec  = _mm512_exp_pd ( _mm512_mul_pd(nOneVec, x2Vec) );
  B1_vec          = _mm512_fmsub_pd (C0_0_Vec, expVec, B0_vec);
  B1_vec          = _mm512_div_pd (B1_vec, x2Vec);
}


inline void NPME_sinx_x_AVX_512 (__m512d& B0_vec, __m512d& cos_vec, 
  const __m512d& xVec)
//B0_vec[8] = sin(xVec[8])/xVec[8]
{
  const __m512d C0_0_Vec = _mm512_set1_pd( 1.000000000000000000);
  const __m512d C0_2_Vec = _mm512_set1_pd(-1.666666666666666667E-1);
  const __m512d C0_4_Vec = _mm512_set1_pd( 8.333333333333333333E-3);
  const __m512d C0_6_Vec = _mm512_set1_pd(-1.984126984126984127E-4);
  const __m512d C0_8_Vec = _mm512_set1_pd( 2.755731922398589065E-6);
  const __m512d xMaxVec  = _mm512_set1_pd(1.0E-1);

  const __m512d x2Vec    = _mm512_mul_pd (xVec, xVec);


  __m512d B0_vecA, B0_vecB;
  B0_vecA = _mm512_fmadd_pd (C0_8_Vec, x2Vec, C0_6_Vec);
  B0_vecA = _mm512_fmadd_pd (B0_vecA,   x2Vec, C0_4_Vec);
  B0_vecA = _mm512_fmadd_pd (B0_vecA,   x2Vec, C0_2_Vec);
  B0_vecA = _mm512_fmadd_pd (B0_vecA,   x2Vec, C0_0_Vec);


  __m512d sin_vec;
  sin_vec = _mm512_sincos_pd (&cos_vec, xVec);
  B0_vecB = _mm512_div_pd (sin_vec, xVec);


  //use B0_vecA if r < Rdir
  //use B0_vecB if r > Rdir
  {
    __mmask8 maskVec = _mm512_cmp_pd_mask (xVec, xMaxVec, 1);
    B0_vec = _mm512_mask_mov_pd (B0_vecB, maskVec, B0_vecA);
  }
}

inline void NPME_sinx_x_AVX_512 (__m512d& B0_vec, __m512d& B1_vec, 
  __m512d& cos_vec, const __m512d& xVec)
//B0_vec[4] = sin(xVec[4])/xVec[4]
//B1_vec[4] = 1/x d/dx B0_vec[4]
{
  const __m512d C0_0_Vec = _mm512_set1_pd( 1.000000000000000000);
  const __m512d C0_2_Vec = _mm512_set1_pd(-1.666666666666666667E-1);
  const __m512d C0_4_Vec = _mm512_set1_pd( 8.333333333333333333E-3);
  const __m512d C0_6_Vec = _mm512_set1_pd(-1.984126984126984127E-4);
  const __m512d C0_8_Vec = _mm512_set1_pd( 2.755731922398589065E-6);

  const __m512d C1_0_Vec = _mm512_set1_pd(-3.333333333333333333E-1);
  const __m512d C1_2_Vec = _mm512_set1_pd( 3.333333333333333333E-2);
  const __m512d C1_4_Vec = _mm512_set1_pd(-1.190476190476190476E-3);
  const __m512d C1_6_Vec = _mm512_set1_pd( 2.204585537918871252E-5);



  const __m512d xMaxVec  = _mm512_set1_pd(1.0E-1);

  const __m512d x2Vec    = _mm512_mul_pd (xVec, xVec);


  __m512d B0_vecA, B0_vecB, B1_vecA, B1_vecB;
  B0_vecA = _mm512_fmadd_pd (C0_8_Vec, x2Vec, C0_6_Vec);
  B0_vecA = _mm512_fmadd_pd (B0_vecA,   x2Vec, C0_4_Vec);
  B0_vecA = _mm512_fmadd_pd (B0_vecA,   x2Vec, C0_2_Vec);
  B0_vecA = _mm512_fmadd_pd (B0_vecA,   x2Vec, C0_0_Vec);

  B1_vecA = _mm512_fmadd_pd (C1_6_Vec,  x2Vec, C1_4_Vec);
  B1_vecA = _mm512_fmadd_pd (B1_vecA,   x2Vec, C1_2_Vec);
  B1_vecA = _mm512_fmadd_pd (B1_vecA,   x2Vec, C1_0_Vec);


  __m512d sin_vec;
  sin_vec = _mm512_sincos_pd (&cos_vec, xVec);
  B0_vecB = _mm512_div_pd (sin_vec, xVec);

  B1_vecB = _mm512_sub_pd (cos_vec,  B0_vecB);
  B1_vecB = _mm512_div_pd (B1_vecB, x2Vec);


  //use B0_vecA if r < Rdir
  //use B0_vecB if r > Rdir
  {
    __mmask8 maskVec = _mm512_cmp_pd_mask (xVec, xMaxVec, 1);
    B0_vec = _mm512_mask_mov_pd (B0_vecB, maskVec, B0_vecA);
    B1_vec = _mm512_mask_mov_pd (B1_vecB, maskVec, B1_vecA);
  }
}


#endif

}//end namespace NPME_Library


#endif // NPME_MATH_FUNCTIONS_H



