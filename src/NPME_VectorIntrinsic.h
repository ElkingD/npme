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

#ifndef NPME_VECTOR_INTRINSIC_H
#define NPME_VECTOR_INTRINSIC_H

#include <cstdio>
#include <iostream>

#include <immintrin.h>

#include "NPME_Constant.h"

namespace NPME_Library
{

//******************************************************************************
//******************************************************************************
//***************************Transpose Functions********************************
//******************************************************************************
//******************************************************************************

//_mm256_4x4transpose_pd_CPU_Time = 3.133709e-09
//_mm256_3x4transpose_pd_CPU_Time = 2.194339e-09
//_mm256_4x3transpose_pd_CPU_Time = 2.196644e-09
//_mm256_6x4transpose_pd_CPU_Time = 4.698701e-09
//_mm256_4x6transpose_pd_CPU_Time = 4.701920e-09

//_mm512_8x8transpose_pd_CPU_Time = 9.705492e-09
//_mm512_8x3transpose_pd_CPU_Time = 3.396265e-09
//_mm512_3x8transpose_pd_CPU_Time = 3.028712e-09

#if NPME_USE_AVX
inline void NPME_mm256_4x4transpose_pd (
  const __m256d& v0, const __m256d& v1, const __m256d& v2, const __m256d& v3,
        __m256d& t0,       __m256d& t1,       __m256d& t2,       __m256d& t3)
//t0,t1,t2,t3 is the transpose of v0,v1,v2,v3
{
  __m256d s0 = _mm256_shuffle_pd(v0, v1, 0b0000);
  __m256d s1 = _mm256_shuffle_pd(v0, v1, 0b1111);
  __m256d s2 = _mm256_shuffle_pd(v2, v3, 0b0000);
  __m256d s3 = _mm256_shuffle_pd(v2, v3, 0b1111);

  t0 = _mm256_permute2f128_pd(s0, s2, 0b00100000);
  t1 = _mm256_permute2f128_pd(s1, s3, 0b00100000);
  t2 = _mm256_permute2f128_pd(s0, s2, 0b00110001);
  t3 = _mm256_permute2f128_pd(s1, s3, 0b00110001);
}
inline void NPME_mm256_2x4transpose_pd (
  const __m256d& a,  const __m256d& b, 
        __m256d& t0,       __m256d& t1)
//input:  a  = (a0, a1, a2, a3)  b = (b0, b1, b2, b3)
//output: t0 = (a0, b0, a1, b1) t1 = (a2, b2, a3, b3)
{
  __m256d s0 = _mm256_shuffle_pd(a, b, 0b0000);
  __m256d s1 = _mm256_shuffle_pd(a, b, 0b1111);
  t0 = _mm256_permute2f128_pd(s0, s1, 0b00100000);
  t1 = _mm256_permute2f128_pd(s0, s1, 0b00110001);
}
inline void NPME_mm256_4x2transpose_pd (
  const __m256d& t0, const __m256d& t1,
        __m256d& a,        __m256d& b)
//input:  t0 = (a0, b0, a1, b1) t1 = (a2, b2, a3, b3)
//output: a  = (a0, a1, a2, a3)  b = (b0, b1, b2, b3)
{
  __m256d s0  = _mm256_permute2f128_pd(t0, t1, 0b00100000);
  __m256d s1  = _mm256_permute2f128_pd(t0, t1, 0b00110001);
  a           = _mm256_shuffle_pd(s0, s1, 0b0000);
  b           = _mm256_shuffle_pd(s0, s1, 0b1111);
}

inline void NPME_mm256_3x4transpose_pd (
  const __m256d& x,  const __m256d& y,  const __m256d& z, 
        __m256d& t0,       __m256d& t1,       __m256d& t2)
//input:  x  = (x0, x1, x2, x3)  y = (y0, y1, y2, y3)  z = (z0, z1, z2, z3)
//output: t0 = (x0, y0, z0, x1) t1 = (y1, z1, x2, y2) t2 = (z2, x3, y3, z3)
{
  __m256d s0 = _mm256_shuffle_pd(x, y, 0b0000);
  __m256d s1 = _mm256_shuffle_pd(y, z, 0b1111);
  __m256d s2 = _mm256_shuffle_pd(z, x, 0b1010);

  t0 = _mm256_permute2f128_pd(s0, s2, 0b00100000);
  t1 = _mm256_permute2f128_pd(s0, s1, 0b00010010);
  t2 = _mm256_permute2f128_pd(s1, s2, 0b00010011);
}


inline void NPME_mm256_4x3transpose_pd (
  const __m256d& t0, const __m256d& t1, const __m256d& t2, 
        __m256d&  x,       __m256d&  y,       __m256d&  z)
//input:  t0 = (x0, y0, z0, x1) t1 = (y1, z1, x2, y2) t2 = (z2, x3, y3, z3)
//output: x  = (x0, x1, x2, x3)  y = (y0, y1, y2, y3)  z = (z0, z1, z2, z3)
{
  __m256d s0 = _mm256_permute2f128_pd(t0, t1, 0b00110000);
  __m256d s1 = _mm256_permute2f128_pd(t1, t2, 0b00110000);
  __m256d s2 = _mm256_permute2f128_pd(t2, t0, 0b00000011);

  x = _mm256_shuffle_pd(s0, s2, 0b1010);
  y = _mm256_shuffle_pd(s0, s1, 0b0101);
  z = _mm256_shuffle_pd(s2, s1, 0b1010);
}

inline void NPME_mm256_6x4transpose_pd (
  const __m256d& RX, const __m256d& RY, const __m256d& RZ, 
  const __m256d& IX, const __m256d& IY, const __m256d& IZ, 
        __m256d& t0,       __m256d& t1,       __m256d& t2,
        __m256d& t3,       __m256d& t4,       __m256d& t5)
//input:  RX = (RX0, RX1, RX2, RX3)  RY = (RY0, RY1, RY2, RY3)  RZ = (RZ0, RZ1, RZ2, RZ3)
//        IX = (IX0, IX1, IX2, IX3)  IY = (IY0, IY1, IY2, IY3)  IZ = (IZ0, IZ1, IZ2, IZ3)
//output: t0 = (RX0, IX0, RY0, IY0)  t1 = (RZ0, IZ0, RX1, IX1)  t2 = (RY1, IY1, RZ1, IZ1)
//        t0 = (RX2, IX2, RY2, IY2)  t1 = (RZ2, IZ2, RX3, IX3)  t2 = (RY3, IY3, RZ3, IZ3)
{
  __m256d s0 = _mm256_shuffle_pd(RX, IX, 0b0000);
  __m256d s1 = _mm256_shuffle_pd(RX, IX, 0b1111);
  __m256d s2 = _mm256_shuffle_pd(RY, IY, 0b0000);
  __m256d s3 = _mm256_shuffle_pd(RY, IY, 0b1111);
  __m256d s4 = _mm256_shuffle_pd(RZ, IZ, 0b0000);
  __m256d s5 = _mm256_shuffle_pd(RZ, IZ, 0b1111);


  t0 = _mm256_permute2f128_pd(s0, s2, 0b00100000);
  t1 = _mm256_permute2f128_pd(s1, s4, 0b00000010);
  t2 = _mm256_permute2f128_pd(s3, s5, 0b00100000);
  t3 = _mm256_permute2f128_pd(s0, s2, 0b00110001);
  t4 = _mm256_permute2f128_pd(s1, s4, 0b00010011);
  t5 = _mm256_permute2f128_pd(s3, s5, 0b00110001);
}

inline void NPME_mm256_4x6transpose_pd (
  const __m256d& t0, const __m256d& t1, const __m256d& t2,
  const __m256d& t3, const __m256d& t4, const __m256d& t5,
        __m256d& RX,       __m256d& RY,       __m256d& RZ, 
        __m256d& IX,       __m256d& IY,       __m256d& IZ)
//input:  t0 = (RX0, IX0, RY0, IY0)  t1 = (RZ0, IZ0, RX1, IX1)  t2 = (RY1, IY1, RZ1, IZ1)
//        t0 = (RX2, IX2, RY2, IY2)  t1 = (RZ2, IZ2, RX3, IX3)  t2 = (RY3, IY3, RZ3, IZ3)
//output: RX = (RX0, RX1, RX2, RX3)  RY = (RY0, RY1, RY2, RY3)  RZ = (RZ0, RZ1, RZ2, RZ3)
//        IX = (IX0, IX1, IX2, IX3)  IY = (IY0, IY1, IY2, IY3)  IZ = (IZ0, IZ1, IZ2, IZ3)
{
  __m256d s0 = _mm256_permute2f128_pd(t0, t3, 0b00100000);
  __m256d s1 = _mm256_permute2f128_pd(t1, t4, 0b00110001);
  __m256d s2 = _mm256_permute2f128_pd(t0, t3, 0b00110001);
  __m256d s3 = _mm256_permute2f128_pd(t2, t5, 0b00100000);
  __m256d s4 = _mm256_permute2f128_pd(t1, t4, 0b00100000);
  __m256d s5 = _mm256_permute2f128_pd(t2, t5, 0b00110001);


  RX = _mm256_shuffle_pd(s0, s1, 0b0000);
  IX = _mm256_shuffle_pd(s0, s1, 0b1111);
  RY = _mm256_shuffle_pd(s2, s3, 0b0000);
  IY = _mm256_shuffle_pd(s2, s3, 0b1111);
  RZ = _mm256_shuffle_pd(s4, s5, 0b0000);
  IZ = _mm256_shuffle_pd(s4, s5, 0b1111);
}



inline void NPME_mm256_4x3_loadCoordTranspose (const double *coord,
  __m256d& xVec, __m256d& yVec, __m256d& zVec)
//input:  coord[12] = {r0[3], r1[3], r2[3], r3[3]};
//output: xVec = {x0, x1, x2, x3};
//        yVec = {y0, y1, y2, y3};
//        zVec = {z0, z1, z2, z3};
{
  __m256d t0 = _mm256_loadu_pd (&coord[0]);
  __m256d t1 = _mm256_loadu_pd (&coord[4]);
  __m256d t2 = _mm256_loadu_pd (&coord[8]);
  NPME_mm256_4x3transpose_pd (t0, t1, t2, xVec, yVec, zVec);
}

inline void NPME_mm256_4x4HorizontalSum_pd (__m256d& sumVec, 
  const __m256d& aVec, const __m256d& bVec, 
  const __m256d& cVec, const __m256d& dVec)
//sum[0] = aVec[0]+aVec[1]+aVec[2]+aVec[3]
//sum[1] = bVec[0]+bVec[1]+bVec[2]+bVec[3]
//sum[2] = cVec[0]+cVec[1]+cVec[2]+cVec[3]
//sum[3] = dVec[0]+dVec[1]+dVec[2]+dVec[3]
{
  __m256d t0, t1, t2, t3;
  NPME_mm256_4x4transpose_pd (aVec, bVec, cVec, dVec, t0, t1, t2, t3);

  sumVec = _mm256_add_pd (_mm256_add_pd (t0, t1), 
                          _mm256_add_pd (t2, t3) );
}

inline void NPME_mm256_4x4HorizontalSum_pd (double sum[4], 
  const __m256d& aVec, const __m256d& bVec, 
  const __m256d& cVec, const __m256d& dVec)
//sum[0] = aVec[0]+aVec[1]+aVec[2]+aVec[3]
//sum[1] = bVec[0]+bVec[1]+bVec[2]+bVec[3]
//sum[2] = cVec[0]+cVec[1]+cVec[2]+cVec[3]
//sum[3] = dVec[0]+dVec[1]+dVec[2]+dVec[3]
{
  __m256d sumVec;
  NPME_mm256_4x4HorizontalSum_pd (sumVec, aVec, bVec, cVec, dVec);

  double *ptr = (double *) &sumVec;
  sum[0] = ptr[0];
  sum[1] = ptr[1];
  sum[2] = ptr[2];
  sum[3] = ptr[3];
}


#endif


#if NPME_USE_AVX_512
inline void NPME_mm512_8x8transpose_pd (
  const __m512d& v0, const __m512d& v1, const __m512d& v2, const __m512d& v3,
  const __m512d& v4, const __m512d& v5, const __m512d& v6, const __m512d& v7,
        __m512d& t0,       __m512d& t1,       __m512d& t2,       __m512d& t3,
        __m512d& t4,       __m512d& t5,       __m512d& t6,       __m512d& t7)
//t0,t1,t2,t3,t4,t5,t6,t7 is the transpose of v0,v1,v2,v3,v4,v5,v6,v7
{
  //s0 = (a0,b0,a2,b2,a4,b4,a6,b6)
  //s1 = (a1,b1,a3,b3,a5,b5,a7,b7)
  //s2 = (c0,d0,c2,d2,c4,d4,c6,d6)
  //s3 = (c1,d1,c3,d3,c5,d5,c7,d7)
  __m512d s0   = _mm512_permutex2var_pd (v0, _mm512_setr_epi64 (0, 8, 2, 10, 4, 12, 6, 14), v1);
  __m512d s2   = _mm512_permutex2var_pd (v2, _mm512_setr_epi64 (0, 8, 2, 10, 4, 12, 6, 14), v3);
  __m512d s4   = _mm512_permutex2var_pd (v4, _mm512_setr_epi64 (0, 8, 2, 10, 4, 12, 6, 14), v5);
  __m512d s6   = _mm512_permutex2var_pd (v6, _mm512_setr_epi64 (0, 8, 2, 10, 4, 12, 6, 14), v7);

  //s4 = (e0,f0,e2,f2,e4,f4,e6,f6)
  //s5 = (e1,f1,e3,f3,e5,f5,e7,f7)
  //s6 = (g0,h0,g2,h2,g4,h4,g6,h6)
  //s7 = (g1,h1,g3,h3,g5,h5,g7,h7)
  __m512d s1   = _mm512_permutex2var_pd (v0, _mm512_setr_epi64 (1, 9, 3, 11, 5, 13, 7, 15), v1);
  __m512d s3   = _mm512_permutex2var_pd (v2, _mm512_setr_epi64 (1, 9, 3, 11, 5, 13, 7, 15), v3);
  __m512d s5   = _mm512_permutex2var_pd (v4, _mm512_setr_epi64 (1, 9, 3, 11, 5, 13, 7, 15), v5);
  __m512d s7   = _mm512_permutex2var_pd (v6, _mm512_setr_epi64 (1, 9, 3, 11, 5, 13, 7, 15), v7);

  //r0 = (a0,b0,a2,b2,c0,d0,c2,d2)
  //r1 = (a4,b4,a6,b6,c4,d4,c6,d6)
  //r2 = (a1,b1,a3,b3,c1,d1,c3,d3)
  //r3 = (a5,b5,a7,b7,c5,d5,c7,d7)
  __m512d r0   = _mm512_permutex2var_pd (s0, _mm512_setr_epi64 (0, 1, 2, 3,  8,  9, 10, 11), s2);
  __m512d r2   = _mm512_permutex2var_pd (s1, _mm512_setr_epi64 (0, 1, 2, 3,  8,  9, 10, 11), s3);
  __m512d r4   = _mm512_permutex2var_pd (s4, _mm512_setr_epi64 (0, 1, 2, 3,  8,  9, 10, 11), s6);
  __m512d r6   = _mm512_permutex2var_pd (s5, _mm512_setr_epi64 (0, 1, 2, 3,  8,  9, 10, 11), s7);

  //r4 = (e0,f0,e2,f2,g0,h0,g2,h2)
  //r5 = (e4,f4,e6,f6,g4,h4,g6,h6)
  //r6 = (e1,f1,e3,f3,g1,h1,g3,h3)
  //r7 = (e5,f5,e7,f7,g5,h5,g7,h7)
  __m512d r1   = _mm512_permutex2var_pd (s0, _mm512_setr_epi64 (4, 5, 6, 7, 12, 13, 14, 15), s2);
  __m512d r3   = _mm512_permutex2var_pd (s1, _mm512_setr_epi64 (4, 5, 6, 7, 12, 13, 14, 15), s3);
  __m512d r5   = _mm512_permutex2var_pd (s4, _mm512_setr_epi64 (4, 5, 6, 7, 12, 13, 14, 15), s6);
  __m512d r7   = _mm512_permutex2var_pd (s5, _mm512_setr_epi64 (4, 5, 6, 7, 12, 13, 14, 15), s7);

  t0    = _mm512_permutex2var_pd (r0, _mm512_setr_epi64 (0, 1, 4, 5,  8,  9, 12, 13), r4);
  t1    = _mm512_permutex2var_pd (r2, _mm512_setr_epi64 (0, 1, 4, 5,  8,  9, 12, 13), r6);
  t4    = _mm512_permutex2var_pd (r1, _mm512_setr_epi64 (0, 1, 4, 5,  8,  9, 12, 13), r5);
  t5    = _mm512_permutex2var_pd (r3, _mm512_setr_epi64 (0, 1, 4, 5,  8,  9, 12, 13), r7);

  t2    = _mm512_permutex2var_pd (r0, _mm512_setr_epi64 (2, 3, 6, 7, 10, 11, 14, 15), r4);
  t3    = _mm512_permutex2var_pd (r2, _mm512_setr_epi64 (2, 3, 6, 7, 10, 11, 14, 15), r6);
  t6    = _mm512_permutex2var_pd (r1, _mm512_setr_epi64 (2, 3, 6, 7, 10, 11, 14, 15), r5);
  t7    = _mm512_permutex2var_pd (r3, _mm512_setr_epi64 (2, 3, 6, 7, 10, 11, 14, 15), r7);
}

inline void NPME_mm512_8x2transpose_pd (
  const __m512d& t0, const __m512d& t1, 
        __m512d&  x,       __m512d&  y)
//input:  t0 = (x0,y0,x1,y1,x2,y2,x3,y3) 
//        t1 = (x4,y4,x5,y5,x6,y6,x7,y7) 
//output: x  = (x0,x1,x2,x3,x4,x5,x6,x7)
//        y  = (y0,y1,y2,y3,y4,y5,y6,y7)
{
  x = _mm512_permutex2var_pd (t0, _mm512_setr_epi64 (0, 2, 4, 6, 8, 10, 12, 14), t1);
  y = _mm512_permutex2var_pd (t0, _mm512_setr_epi64 (1, 3, 5, 7, 9, 11, 13, 15), t1);
}
inline void NPME_mm512_2x8transpose_pd (
  const __m512d&  x, const __m512d&  y,
        __m512d& t0,       __m512d& t1)
//input:  x  = (x0,x1,x2,x3,x4,x5,x6,x7)
//        y  = (y0,y1,y2,y3,y4,y5,y6,y7)
//output: t0 = (x0,y0,x1,y1,x2,y2,x3,y3) 
//        t1 = (x4,y4,x5,y5,x6,y6,x7,y7) 
{
  t0 = _mm512_permutex2var_pd (x, _mm512_setr_epi64 (0,  8, 1,  9, 2, 10, 3, 11), y);
  t1 = _mm512_permutex2var_pd (x, _mm512_setr_epi64 (4, 12, 5, 13, 6, 14, 7, 15), y);
}


inline void NPME_mm512_8x3transpose_pd (
  const __m512d& t0, const __m512d& t1, const __m512d& t2, 
        __m512d&  x,       __m512d&  y,       __m512d&  z)
//input:  t0 = (x0,y0,z0,x1,y1,z1,x2,y2) 
//        t1 = (z2,x3,y3,z3,x4,y4,z4,x5)
//        t2 = (y5,z5,x6,y6,z6,x7,y7,z7)
//output: x  = (x0,x1,x2,x3,x4,x5,x6,x7)
//        y  = (y0,y1,y2,y3,y4,y5,y6,y7)
//        z  = (z0,z1,z2,z3,z4,z5,z6,z7)
{
  __m512d s;

//s   = (x0,x3,x0,x1,x4,x0,x2,x5)
  s   = _mm512_permutex2var_pd (t0, _mm512_setr_epi64 (0, 9, 0, 3, 12, 0, 6, 15), t1);
  x   = _mm512_permutex2var_pd (s,  _mm512_setr_epi64 (0, 3, 6, 1, 4, 7, 10, 13), t2);

//s   = (x0,y0,y3,x0,y1,y4,x0,y2)
  s   = _mm512_permutex2var_pd (t0, _mm512_setr_epi64 (0, 1, 10, 0, 4, 13, 0, 7), t1);
  y   = _mm512_permutex2var_pd (s,  _mm512_setr_epi64 (1, 4, 7, 2, 5, 8, 11, 14), t2);

//s   = (z2,x0,z0,z3,x0,z1,z4,x0)
  s   = _mm512_permutex2var_pd (t0, _mm512_setr_epi64 (8, 0, 2, 11, 0, 5, 14, 0), t1);
  z   = _mm512_permutex2var_pd (s,  _mm512_setr_epi64 (2, 5, 0, 3, 6, 9, 12, 15), t2);
}

inline void NPME_mm512_3x8transpose_pd (
  const __m512d&  x, const __m512d&  y, const __m512d&  z, 
        __m512d& t0,       __m512d& t1,       __m512d& t2)
//input:  x  = (x0,x1,x2,x3,x4,x5,x6,x7)
//        y  = (y0,y1,y2,y3,y4,y5,y6,y7)
//        z  = (z0,z1,z2,z3,z4,z5,z6,z7)
//output: t0 = (x0,y0,z0,x1,y1,z1,x2,y2) 
//        t1 = (z2,x3,y3,z3,x4,y4,z4,x5)
//        t2 = (y5,z5,x6,y6,z6,x7,y7,z7)

{
  __m512d s;

//s = (x0,y0,x1,y1,x2,y2,x0,x0)
  s   = _mm512_permutex2var_pd (x, _mm512_setr_epi64 (0, 8, 1, 9, 2, 10, 0, 0), y);
  t0  = _mm512_permutex2var_pd (s, _mm512_setr_epi64 (0, 1, 8, 2, 3,  9, 4, 5), z);

//s = (x3,y3,x4,y4,x5,x0,x0,x0)
  s   = _mm512_permutex2var_pd (x, _mm512_setr_epi64 (3, 11, 4, 12, 5, 0, 0,  0), y);
  t1  = _mm512_permutex2var_pd (s, _mm512_setr_epi64 (10, 0, 1, 11, 2, 3, 12, 4), z);

//s = (y5,x6,y6,x7,y7,x0,x0,x0)
  s   = _mm512_permutex2var_pd (x, _mm512_setr_epi64 (13, 6, 14, 7, 15, 0, 0, 0), y);
  t2  = _mm512_permutex2var_pd (s, _mm512_setr_epi64 (0, 13,  1, 2, 14, 3, 4, 15), z);
}

inline void NPME_mm512_4x8transpose_pd (
  const __m512d&  a, const __m512d&  b, const __m512d&  c, const __m512d&  d,
        __m512d& t0,       __m512d& t1,       __m512d& t2,       __m512d& t3)
//input:  a  = (a0,a1,a2,a3,a4,a5,a6,a7)
//        b  = (b0,b1,b2,b3,b4,b5,b6,b7)
//        c  = (c0,c1,c2,c3,c4,c5,c6,c7)
//        d  = (d0,d1,d2,d3,d4,d5,d6,d7)
//output: t0 = (a0,b0,c0,d0,a1,b1,c1,d1)
//        t1 = (a2,b2,c2,d2,a3,b3,c3,d3)
//        t2 = (a4,b4,c4,d4,a5,b5,c5,d5)
//        t3 = (a6,b6,c6,d6,a7,b7,c7,d7)
{
  __m512d s0 = _mm512_permutex2var_pd (a, _mm512_setr_epi64 (0,  8, 1,  9, 2, 10, 3, 11), b);
  __m512d s1 = _mm512_permutex2var_pd (a, _mm512_setr_epi64 (4, 12, 5, 13, 6, 14, 7, 15), b);
  __m512d s2 = _mm512_permutex2var_pd (c, _mm512_setr_epi64 (0,  8, 1,  9, 2, 10, 3, 11), d);
  __m512d s3 = _mm512_permutex2var_pd (c, _mm512_setr_epi64 (4, 12, 5, 13, 6, 14, 7, 15), d);

  t0 = _mm512_permutex2var_pd (s0, _mm512_setr_epi64 (0,  1,  8,  9, 2,  3, 10, 11), s2);
  t1 = _mm512_permutex2var_pd (s0, _mm512_setr_epi64 (4,  5, 12, 13, 6,  7, 14, 15), s2);
  t2 = _mm512_permutex2var_pd (s1, _mm512_setr_epi64 (0,  1,  8,  9, 2,  3, 10, 11), s3);
  t3 = _mm512_permutex2var_pd (s1, _mm512_setr_epi64 (4,  5, 12, 13, 6,  7, 14, 15), s3);
}

inline void NPME_mm512_8x4transpose_pd (
  const __m512d& t0, const __m512d& t1, const __m512d& t2, const __m512d& t3,
        __m512d&  a,       __m512d&  b,       __m512d&  c,       __m512d&  d)
//input:  t0 = (a0,b0,c0,d0,a1,b1,c1,d1)
//        t1 = (a2,b2,c2,d2,a3,b3,c3,d3)
//        t2 = (a4,b4,c4,d4,a5,b5,c5,d5)
//        t3 = (a6,b6,c6,d6,a7,b7,c7,d7)
//output: a  = (a0,a1,a2,a3,a4,a5,a6,a7)
//        b  = (b0,b1,b2,b3,b4,b5,b6,b7)
//        c  = (c0,c1,c2,c3,c4,c5,c6,c7)
//        d  = (d0,d1,d2,d3,d4,d5,d6,d7)

{
  __m512d s0 = _mm512_permutex2var_pd (t0, _mm512_setr_epi64 (0, 1, 4, 5,  8,  9, 12, 13), t1);
  __m512d s2 = _mm512_permutex2var_pd (t0, _mm512_setr_epi64 (2, 3, 6, 7, 10, 11, 14, 15), t1);
  __m512d s1 = _mm512_permutex2var_pd (t2, _mm512_setr_epi64 (0, 1, 4, 5,  8,  9, 12, 13), t3);
  __m512d s3 = _mm512_permutex2var_pd (t2, _mm512_setr_epi64 (2, 3, 6, 7, 10, 11, 14, 15), t3);

  a = _mm512_permutex2var_pd (s0, _mm512_setr_epi64 (0, 2, 4,  6,  8, 10, 12, 14), s1);
  b = _mm512_permutex2var_pd (s0, _mm512_setr_epi64 (1, 3, 5,  7,  9, 11, 13, 15), s1);
  c = _mm512_permutex2var_pd (s2, _mm512_setr_epi64 (0, 2, 4,  6,  8, 10, 12, 14), s3);
  d = _mm512_permutex2var_pd (s2, _mm512_setr_epi64 (1, 3, 5,  7,  9, 11, 13, 15), s3);
}

inline void NPME_mm512_6x8transpose_pd (
  const __m512d& RX, const __m512d& RY, const __m512d& RZ, 
  const __m512d& IX, const __m512d& IY, const __m512d& IZ, 
        __m512d& t0,       __m512d& t1,       __m512d& t2,
        __m512d& t3,       __m512d& t4,       __m512d& t5)
//input:  RX = (RX0,RX1,RX2,RX3,RX4,RX5,RX6,RX7)
//        RY = (RY0,RY1,RY2,RY3,RY4,RY5,RY6,RY7)
//        RZ = (RZ0,RZ1,RZ2,RZ3,RZ4,RZ5,RZ6,RZ7)
//        IX = (IX0,IX1,IX2,IX3,IX4,IX5,IX6,IX7)
//        IY = (IY0,IY1,IY2,IY3,IY4,IY5,IY6,IY7)
//        IZ = (IZ0,IZ1,IZ2,IZ3,IZ4,IZ5,IZ6,IZ7)
//output: t0 = (RX0,IX0,RY0,IY0,RZ0,IZ0,RX1,IX1)
//        t1 = (RY1,IY1,RZ1,IZ1,RX2,IX2,RY2,IY2)
//        t2 = (RZ2,IZ2,RX3,IX3,RY3,IY3,RZ3,IZ3)
//        t3 = (RX4,IX4,RY4,IY4,RZ4,IZ4,RX5,IX5)
//        t4 = (RY5,IY5,RZ5,IZ5,RX6,IX6,RY6,IY6)
//        t5 = (RZ6,IZ6,RX7,IX7,RY7,IY7,RZ7,IZ7)
{
  //X0 = (RX0,IX0,RX1,IX1,RX2,IX2,RX3,IX3)
  //X1 = (RX4,IX4,RX5,IX5,RX6,IX6,RX7,IX7)
  //Y0 = (RY0,IY0,RY1,IY1,RY2,IY2,RY3,IY3)
  //Y1 = (RY4,IY4,RY5,IY5,RY6,IY6,RY7,IY7)
  //Z0 = (RZ0,IZ0,RZ1,IZ1,RZ2,IZ2,RZ3,IZ3)
  //Z1 = (RZ4,IZ4,RZ5,IZ5,RZ6,IZ6,RZ7,IZ7)
  __m512d X0  = _mm512_permutex2var_pd (RX, _mm512_setr_epi64 (0, 8, 1, 9, 2, 10, 3, 11), IX);
  __m512d X1  = _mm512_permutex2var_pd (RX, _mm512_setr_epi64 (4, 12, 5, 13, 6, 14, 7, 15), IX);
  __m512d Y0  = _mm512_permutex2var_pd (RY, _mm512_setr_epi64 (0, 8, 1, 9, 2, 10, 3, 11), IY);
  __m512d Y1  = _mm512_permutex2var_pd (RY, _mm512_setr_epi64 (4, 12, 5, 13, 6, 14, 7, 15), IY);
  __m512d Z0  = _mm512_permutex2var_pd (RZ, _mm512_setr_epi64 (0, 8, 1, 9, 2, 10, 3, 11), IZ);
  __m512d Z1  = _mm512_permutex2var_pd (RZ, _mm512_setr_epi64 (4, 12, 5, 13, 6, 14, 7, 15), IZ);

  //s0 = (RX0,IX0,RX1,IX1,RY0,IY0,RY1,IY1)
  //s1 = (RY1,IY1,RX2,IX2,RY2,IY2,RX0,RX0)
  //s2 = (RX3,IX3,RY3,IY3,RX0,RX0,RX0,RX0)
  //s3 = (RX4,IX4,RX5,IX5,RY4,IY4,RY5,IY5)
  //s4 = (RY5,IY5,RX6,IX6,RY6,IY6,RX4,RX4)
  //s5 = (RX7,IX7,RY7,IY7,RX4,RX4,RX4,RX4)
  __m512d s0  = _mm512_permutex2var_pd (X0, _mm512_setr_epi64 (0, 1, 2, 3, 8, 9, 10, 11), Y0);
  __m512d s1  = _mm512_permutex2var_pd (X0, _mm512_setr_epi64 (10, 11, 4, 5, 12, 13, 0, 0), Y0);
  __m512d s2  = _mm512_permutex2var_pd (X0, _mm512_setr_epi64 (6, 7, 14, 15, 0, 0, 0, 0), Y0);
  __m512d s3  = _mm512_permutex2var_pd (X1, _mm512_setr_epi64 (0, 1, 2, 3, 8, 9, 10, 11), Y1);
  __m512d s4  = _mm512_permutex2var_pd (X1, _mm512_setr_epi64 (10, 11, 4, 5, 12, 13, 0, 0), Y1);
  __m512d s5  = _mm512_permutex2var_pd (X1, _mm512_setr_epi64 (6, 7, 14, 15, 0, 0, 0, 0), Y1);

  t0  = _mm512_permutex2var_pd (s0, _mm512_setr_epi64 (0, 1, 4, 5, 8, 9, 2, 3), Z0);
  t1  = _mm512_permutex2var_pd (s1, _mm512_setr_epi64 (0, 1, 10, 11, 2, 3, 4, 5), Z0);
  t2  = _mm512_permutex2var_pd (s2, _mm512_setr_epi64 (12, 13, 0, 1, 2, 3, 14, 15), Z0);
  t3  = _mm512_permutex2var_pd (s3, _mm512_setr_epi64 (0, 1, 4, 5, 8, 9, 2, 3), Z1);
  t4  = _mm512_permutex2var_pd (s4, _mm512_setr_epi64 (0, 1, 10, 11, 2, 3, 4, 5), Z1);
  t5  = _mm512_permutex2var_pd (s5, _mm512_setr_epi64 (12, 13, 0, 1, 2, 3, 14, 15), Z1);
}

inline void NPME_mm512_8x6transpose_pd (
  const __m512d& t0, const __m512d& t1, const __m512d& t2,
  const __m512d& t3, const __m512d& t4, const __m512d& t5,
        __m512d& RX,       __m512d& RY,       __m512d& RZ, 
        __m512d& IX,       __m512d& IY,       __m512d& IZ)
//input:  t0 = (RX0,IX0,RY0,IY0,RZ0,IZ0,RX1,IX1)
//        t1 = (RY1,IY1,RZ1,IZ1,RX2,IX2,RY2,IY2)
//        t2 = (RZ2,IZ2,RX3,IX3,RY3,IY3,RZ3,IZ3)
//        t3 = (RX4,IX4,RY4,IY4,RZ4,IZ4,RX5,IX5)
//        t4 = (RY5,IY5,RZ5,IZ5,RX6,IX6,RY6,IY6)
//        t5 = (RZ6,IZ6,RX7,IX7,RY7,IY7,RZ7,IZ7)
//output: RX = (RX0,RX1,RX2,RX3,RX4,RX5,RX6,RX7)
//        RY = (RY0,RY1,RY2,RY3,RY4,RY5,RY6,RY7)
//        RZ = (RZ0,RZ1,RZ2,RZ3,RZ4,RZ5,RZ6,RZ7)
//        IX = (IX0,IX1,IX2,IX3,IX4,IX5,IX6,IX7)
//        IY = (IY0,IY1,IY2,IY3,IY4,IY5,IY6,IY7)
//        IZ = (IZ0,IZ1,IZ2,IZ3,IZ4,IZ5,IZ6,IZ7)
{
  __m512d R0, R1, s0, s1, s2, s3;

  //s0 = (RX0,IX0,RX1,IX1,RY0,IY0,RY1,IY1)
  //s1 = (RX2,IX2,RX3,IX3,RY2,IY2,RY3,IY3)
  //s2 = (RX4,IX4,RX5,IX5,RY4,IY4,RY5,IY5)
  //s3 = (RX6,IX6,RX7,IX7,RY6,IY6,RY7,IY7)
  s0  = _mm512_permutex2var_pd (t0, _mm512_setr_epi64 (0, 1, 6, 7, 2, 3, 8, 9), t1);
  s1  = _mm512_permutex2var_pd (t1, _mm512_setr_epi64 (4, 5, 10, 11, 6, 7, 12, 13), t2);
  s2  = _mm512_permutex2var_pd (t3, _mm512_setr_epi64 (0, 1, 6, 7, 2, 3, 8, 9), t4);
  s3  = _mm512_permutex2var_pd (t4, _mm512_setr_epi64 (4, 5, 10, 11, 6, 7, 12, 13), t5);

  //R0 = (RX0,IX0,RX1,IX1,RX2,IX2,RX3,IX3)
  //R1 = (RX4,IX4,RX5,IX5,RX6,IX6,RX7,IX7)
  R0  = _mm512_permutex2var_pd (s0, _mm512_setr_epi64 (0, 1, 2, 3, 8, 9, 10, 11), s1);
  R1  = _mm512_permutex2var_pd (s2, _mm512_setr_epi64 (0, 1, 2, 3, 8, 9, 10, 11), s3);
  RX  = _mm512_permutex2var_pd (R0, _mm512_setr_epi64 (0, 2, 4, 6, 8, 10, 12, 14), R1);
  IX  = _mm512_permutex2var_pd (R0, _mm512_setr_epi64 (1, 3, 5, 7, 9, 11, 13, 15), R1);


  //R0 = (RY0,IY0,RY1,IY1,RY2,IY2,RY3,IY3)
  //R1 = (RY4,IY4,RY5,IY5,RY6,IY6,RY7,IY7)
  R0  = _mm512_permutex2var_pd (s0, _mm512_setr_epi64 (4, 5, 6, 7, 12, 13, 14, 15), s1);
  R1  = _mm512_permutex2var_pd (s2, _mm512_setr_epi64 (4, 5, 6, 7, 12, 13, 14, 15), s3);
  RY  = _mm512_permutex2var_pd (R0, _mm512_setr_epi64 (0, 2, 4, 6, 8, 10, 12, 14), R1);
  IY  = _mm512_permutex2var_pd (R0, _mm512_setr_epi64 (1, 3, 5, 7, 9, 11, 13, 15), R1);


  //s0 = (RZ0,IZ0,RZ1,IZ1,RX0,RX0,RX0,RX0)
  //s1 = (RZ4,IZ4,RZ5,IZ5,RX4,RX4,RX4,RX4)
  s0  = _mm512_permutex2var_pd (t0, _mm512_setr_epi64 (4, 5, 10, 11, 0, 0, 0, 0), t1);
  s1  = _mm512_permutex2var_pd (t3, _mm512_setr_epi64 (4, 5, 10, 11, 0, 0, 0, 0), t4);
  //R0 = (RZ0,IZ0,RZ1,IZ1,RZ2,IZ2,RZ3,IZ3)
  //R1 = (RZ4,IZ4,RZ5,IZ5,RZ6,IZ6,RZ7,IZ7)
  R0  = _mm512_permutex2var_pd (s0, _mm512_setr_epi64 (0, 1, 2, 3, 8, 9, 14, 15), t2);
  R1  = _mm512_permutex2var_pd (s1, _mm512_setr_epi64 (0, 1, 2, 3, 8, 9, 14, 15), t5);
  RZ  = _mm512_permutex2var_pd (R0, _mm512_setr_epi64 (0, 2, 4, 6, 8, 10, 12, 14), R1);
  IZ  = _mm512_permutex2var_pd (R0, _mm512_setr_epi64 (1, 3, 5, 7, 9, 11, 13, 15), R1);
}

inline void NPME_mm512_4x8HorizontalSum_pd (double sum[4], 
  const __m512d& aVec, const __m512d& bVec, 
  const __m512d& cVec, const __m512d& dVec)
//sum[0] = aVec[0]+aVec[1]+aVec[2]+aVec[3]+aVec[4]+aVec[5]+aVec[6]+aVec[7]
//sum[1] = bVec[0]+bVec[1]+bVec[2]+bVec[3]+bVec[4]+bVec[5]+bVec[6]+bVec[7]
//sum[2] = cVec[0]+cVec[1]+cVec[2]+cVec[3]+cVec[4]+cVec[5]+cVec[6]+cVec[7]
//sum[3] = dVec[0]+dVec[1]+dVec[2]+dVec[3]+dVec[4]+dVec[5]+dVec[6]+dVec[7]
{
  __m512d t0, t1, t2, t3;
  NPME_mm512_4x8transpose_pd (aVec, bVec, cVec, dVec, t0, t1, t2, t3);

  __m512d sumVec = _mm512_add_pd (_mm512_add_pd (t0, t1), 
                                  _mm512_add_pd (t2, t3) );

  double *ptr = (double *) &sumVec;
  sum[0] = ptr[0] + ptr[4];
  sum[1] = ptr[1] + ptr[5];
  sum[2] = ptr[2] + ptr[6];
  sum[3] = ptr[3] + ptr[7];
}

#endif



//******************************************************************************
//******************************************************************************
//******************************Print Functions*********************************
//******************************************************************************
//******************************************************************************



#if NPME_USE_AVX
inline void NPME_mm256_PrintVec_pd (std::ostream& os,
  const char *desc, const __m256d& a)
//prints:
//"desc" = a[0] a[1] a[2] a[3]
{
  char str[500];
  const double *aPtr = (double *) &a;
  sprintf(str, "%s = %12.6f %12.6f %12.6f %12.6f\n", 
    desc, aPtr[0], aPtr[1], aPtr[2], aPtr[3]);
  os << str;
}


#endif

#if NPME_USE_AVX_512

inline void NPME_mm512_PrintVec_pd (std::ostream& os, 
  const char *desc, const __m512d& a)
//prints:
//"desc" = a[0] a[1] a[2] a[3] a[4] a[5] a[6] a[7]
{
  char str[500];
  const double *aPtr = (double *) &a;
  sprintf(str, "%s = %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n", 
    desc, aPtr[0], aPtr[1], aPtr[2], aPtr[3], 
          aPtr[4], aPtr[5], aPtr[6], aPtr[7]);
  os << str;
}

#endif

//******************************************************************************
//******************************************************************************
//****************************Branch Functions**********************************
//******************************************************************************
//******************************************************************************

//#define _CMP_EQ_OQ    0x00 // Equal (ordered, non-signaling)  
//#define _CMP_LT_OS    0x01 // Less-than (ordered, signaling)  
//#define _CMP_LE_OS    0x02 // Less-than-or-equal (ordered, signaling)  
//#define _CMP_UNORD_Q  0x03 // Unordered (non-signaling)  
//#define _CMP_NEQ_UQ   0x04 // Not-equal (unordered, non-signaling)  
//#define _CMP_NLT_US   0x05 // Not-less-than (unordered, signaling)  
//#define _CMP_NLE_US   0x06 // Not-less-than-or-equal (unordered, signaling)  
//#define _CMP_ORD_Q    0x07 // Ordered (nonsignaling)   
//#define _CMP_EQ_UQ    0x08 // Equal (unordered, non-signaling)  
//#define _CMP_NGE_US   0x09 // Not-greater-than-or-equal (unord, signaling)  
//#define _CMP_NGT_US   0x0a // Not-greater-than (unordered, signaling)  
//#define _CMP_FALSE_OQ 0x0b // False (ordered, non-signaling)  
//#define _CMP_NEQ_OQ   0x0c // Not-equal (ordered, non-signaling)  
//#define _CMP_GE_OS    0x0d // Greater-than-or-equal (ordered, signaling)  
//#define _CMP_GT_OS    0x0e // Greater-than (ordered, signaling)  
//#define _CMP_TRUE_UQ  0x0f // True (unordered, non-signaling)  
//#define _CMP_EQ_OS    0x10 // Equal (ordered, signaling)  
//#define _CMP_LT_OQ    0x11 // Less-than (ordered, non-signaling)  
//#define _CMP_LE_OQ    0x12 // Less-than-or-equal (ordered, non-signaling)  
//#define _CMP_UNORD_S  0x13 // Unordered (signaling)  
//#define _CMP_NEQ_US   0x14 // Not-equal (unordered, signaling)  
//#define _CMP_NLT_UQ   0x15 // Not-less-than (unordered, non-signaling)  
//#define _CMP_NLE_UQ   0x16 // Not-less-than-or-equal (unord, non-signaling)  
//#define _CMP_ORD_S    0x17 // Ordered (signaling)  
//#define _CMP_EQ_US    0x18 // Equal (unordered, signaling)  
//#define _CMP_NGE_UQ   0x19 // Not-greater-than-or-equal (unord, non-sign)  
//#define _CMP_NGT_UQ   0x1a // Not-greater-than (unordered, non-signaling)  
//#define _CMP_FALSE_OS 0x1b // False (ordered, signaling)  
//#define _CMP_NEQ_OS   0x1c // Not-equal (ordered, signaling)  
//#define _CMP_GE_OQ    0x1d // Greater-than-or-equal (ordered, non-signaling)  
//#define _CMP_GT_OQ    0x1e // Greater-than (ordered, non-signaling)  
//#define _CMP_TRUE_US  0x1f // True (unordered, signaling)  


#if NPME_USE_AVX
inline void NPME_mm256_lessthan_pd (
  const __m256d& a,  const __m256d& b, const __m256d& c,
        __m256d& d)
//for (int i = 0; i < 4; i++)
//  if (a[i] < b[i]) d[i] = c[i]
//  else             d[i] = 0.0;
{
  __m256d t0  = _mm256_cmp_pd (a, b, 1);
  d           = _mm256_and_pd (t0, c);
}

inline void NPME_mm256_lessthan_pd (
  const __m256d& a,  const __m256d& b, const __m256d& c1, const __m256d& c2,
        __m256d& dless, __m256d& dmore)
//for (int i = 0; i < 4; i++)
//  if (a[i] < b[i]) {dless[i] = c1[i]; dmore[i] = 0.0; }
//  else             {dless[i] = 0.0;   dmore[i] = c2[i]; }
{
  __m256d t0  = _mm256_cmp_pd (a, b, 1);
  dless       = _mm256_and_pd    (t0, c1);
  dmore       = _mm256_andnot_pd (t0, c2);
}

inline void NPME_mm256_setlessthan_pd (
  const __m256d& a,  const __m256d& b, const __m256d& c1, const __m256d& c2,
        __m256d& d)
//for (int i = 0; i < 4; i++)
//  if (a[i] < b[i]) d[i] = c1[i];
//  else             d[i] = c2[i];
{
  __m256d t0    = _mm256_cmp_pd (a, b, 1);
  __m256d dless = _mm256_and_pd    (t0, c1);
  __m256d dmore = _mm256_andnot_pd (t0, c2);

  d = _mm256_add_pd (dless, dmore);
}

#endif

#if NPME_USE_AVX_512
inline void NPME_mm512_lessthan_pd (
  const __m512d& a,  const __m512d& b, 
  const __m512d& c1, const __m512d& c2,
  __m512d& d)
//for (int i = 0; i < 8; i++)
//  if (a[i] < b[i]) d[i] = c1[i]
//  else             d[i] = c2[i]
{
  __mmask8 maskVec = _mm512_cmp_pd_mask (a, b, 1);
  d = _mm512_mask_mov_pd (c2, maskVec, c1);
}

#endif


}//namespace NPME_Library
#endif // NPME_VECTOR_INTRINSIC_H




