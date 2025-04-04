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

#ifndef NPME_POTENTIAL_SUPPORT_FUNCTIONS_H
#define NPME_POTENTIAL_SUPPORT_FUNCTIONS_H

#include <immintrin.h>

#include "NPME_VectorIntrinsic.h"


namespace NPME_Library
{
#if NPME_USE_AVX
inline void NPME_TransposeCoord_AVX (const size_t nCharge, const double *coord, 
  double *xAlignTmp, double *yAlignTmp, double *zAlignTmp)
//input:  coord[3*nCharge] not aligned
//        nCharge must be a multiple of 4
//output: xAlignTmp[nCharge] = aligned at 32 bit boundary
//        yAlignTmp[nCharge] = aligned at 32 bit boundary
//        zAlignTmp[nCharge] = aligned at 32 bit boundary
{
  if (nCharge%4 != 0)
  {
    std::cout << "Error in NPME_TransposeCoord_AVX.\n";
    std::cout << "nCharge = " << nCharge << " is not a multiple of 4\n";
    exit(0);
  }

  const size_t m = nCharge/4;

  size_t indexCrd   = 0;
  size_t indexAlign = 0;
  for (size_t i = 0; i < m; i++)
  {
    const double *X  = &coord[indexCrd];
    indexCrd += 12;

    const __m256d t0 = _mm256_loadu_pd (&X[0]);
    const __m256d t1 = _mm256_loadu_pd (&X[4]);
    const __m256d t2 = _mm256_loadu_pd (&X[8]);

    __m256d x, y, z;
    NPME_mm256_4x3transpose_pd (t0, t1, t2, x, y, z);
    _mm256_store_pd (&xAlignTmp[indexAlign], x);
    _mm256_store_pd (&yAlignTmp[indexAlign], y);
    _mm256_store_pd (&zAlignTmp[indexAlign], z);
    indexAlign += 4;
  }
}
inline void NPME_TransposeV1_AVX (const size_t nCharge, double *V1, 
  const double *V0_align, const double *VX_align, 
  const double *VY_align, const double *VZ_align)
//input:  nCharge must be a multiple of 4
//        V0_align[nCharge] = aligned at 32 bit boundary
//        VX_align[nCharge] = aligned at 32 bit boundary
//        VY_align[nCharge] = aligned at 32 bit boundary
//        VZ_align[nCharge] = aligned at 32 bit boundary
//output: V1[nCharge][4] = need not be aligned
{
  if (nCharge%4 != 0)
  {
    std::cout << "Error in NPME_TransposeV1_AVX.\n";
    std::cout << "nCharge = " << nCharge << " is not a multiple of 4\n";
    exit(0);
  }

  const size_t m = nCharge/4;

  size_t index4x  = 0;
  size_t index16x = 0;
  for (size_t i = 0; i < m; i++)
  {
    const __m256d V0_vec = _mm256_load_pd (&V0_align[index4x]);
    const __m256d VX_vec = _mm256_load_pd (&VX_align[index4x]);
    const __m256d VY_vec = _mm256_load_pd (&VY_align[index4x]);
    const __m256d VZ_vec = _mm256_load_pd (&VZ_align[index4x]);
    index4x += 4;

    __m256d t0, t1, t2, t3;
    NPME_mm256_4x4transpose_pd (V0_vec, VX_vec, VY_vec, VZ_vec,
      t0, t1, t2, t3);
    _mm256_storeu_pd (&V1[index16x   ], t0);
    _mm256_storeu_pd (&V1[index16x+ 4], t1);
    _mm256_storeu_pd (&V1[index16x+ 8], t2);
    _mm256_storeu_pd (&V1[index16x+12], t3);
    index16x += 16;
  }
}



inline void NPME_TransposeAddUpdateV1_AVX (const size_t nCharge, double *V1, 
  const double *V0_align, const double *VX_align, 
  const double *VY_align, const double *VZ_align)
//similar to NPME_TransposeV1_AVX, but updates V1 by adding 
//instead of overwriting V1
//input:  nCharge must be a multiple of 4
//        V0_align[nCharge] = aligned at 32 bit boundary
//        VX_align[nCharge] = aligned at 32 bit boundary
//        VY_align[nCharge] = aligned at 32 bit boundary
//        VZ_align[nCharge] = aligned at 32 bit boundary
//output: V1[nCharge][4] = need not be aligned
{
  if (nCharge%4 != 0)
  {
    std::cout << "Error in NPME_TransposeAddUpdateV1_AVX.\n";
    std::cout << "nCharge = " << nCharge << " is not a multiple of 4\n";
    exit(0);
  }

  const size_t m = nCharge/4;

  size_t index4x  = 0;
  size_t index16x = 0;
  for (size_t i = 0; i < m; i++)
  {
    __m256d A_vec = _mm256_load_pd (&V0_align[index4x]);
    __m256d B_vec = _mm256_load_pd (&VX_align[index4x]);
    __m256d C_vec = _mm256_load_pd (&VY_align[index4x]);
    __m256d D_vec = _mm256_load_pd (&VZ_align[index4x]);
    index4x += 4;

    __m256d t0_vec, t1_vec, t2_vec, t3_vec;
    NPME_mm256_4x4transpose_pd (A_vec, B_vec, C_vec, D_vec,
      t0_vec, t1_vec, t2_vec, t3_vec);

    A_vec = _mm256_loadu_pd (&V1[index16x   ]);
    B_vec = _mm256_loadu_pd (&V1[index16x+4 ]);
    C_vec = _mm256_loadu_pd (&V1[index16x+8 ]);
    D_vec = _mm256_loadu_pd (&V1[index16x+12]);

    t0_vec = _mm256_add_pd (t0_vec, A_vec);
    t1_vec = _mm256_add_pd (t1_vec, B_vec);
    t2_vec = _mm256_add_pd (t2_vec, C_vec);
    t3_vec = _mm256_add_pd (t3_vec, D_vec);

    _mm256_storeu_pd (&V1[index16x   ], t0_vec);
    _mm256_storeu_pd (&V1[index16x+ 4], t1_vec);
    _mm256_storeu_pd (&V1[index16x+ 8], t2_vec);
    _mm256_storeu_pd (&V1[index16x+12], t3_vec);
    index16x += 16;
  }
}

inline void NPME_TransposeV1_AVX (
  const size_t nCharge, _Complex double *V1, 
  const double *V0_r_align, const double *V0_i_align,
  const double *VX_r_align, const double *VX_i_align,
  const double *VY_r_align, const double *VY_i_align,
  const double *VZ_r_align, const double *VZ_i_align)
//input:  nCharge must be a multiple of 4
//        V0_align[nCharge] = aligned at 32 bit boundary
//        VX_align[nCharge] = aligned at 32 bit boundary
//        VY_align[nCharge] = aligned at 32 bit boundary
//        VZ_align[nCharge] = aligned at 32 bit boundary
//output: V1[nCharge][4] = need not be aligned
{
  if (nCharge%4 != 0)
  {
    std::cout << "Error in NPME_TransposeV1_AVX.\n";
    std::cout << "nCharge = " << nCharge << " is not a multiple of 4\n";
    exit(0);
  }

  const size_t m = nCharge/4;

  double *V1r = (double *) V1;

  size_t index4x  = 0;
  size_t index32x = 0;
  for (size_t i = 0; i < m; i++)
  {
    const __m256d V0_r_vec = _mm256_load_pd (&V0_r_align[index4x]);
    const __m256d VX_r_vec = _mm256_load_pd (&VX_r_align[index4x]);
    const __m256d VY_r_vec = _mm256_load_pd (&VY_r_align[index4x]);
    const __m256d VZ_r_vec = _mm256_load_pd (&VZ_r_align[index4x]);

    const __m256d V0_i_vec = _mm256_load_pd (&V0_i_align[index4x]);
    const __m256d VX_i_vec = _mm256_load_pd (&VX_i_align[index4x]);
    const __m256d VY_i_vec = _mm256_load_pd (&VY_i_align[index4x]);
    const __m256d VZ_i_vec = _mm256_load_pd (&VZ_i_align[index4x]);

    index4x += 4;

    __m256d t0, t1, t2, t3;
    NPME_mm256_4x4transpose_pd (
      V0_r_vec, V0_i_vec, VX_r_vec, VX_i_vec,
      t0, t1, t2, t3);

    __m256d s0, s1, s2, s3;
    NPME_mm256_4x4transpose_pd (
      VY_r_vec, VY_i_vec, VZ_r_vec, VZ_i_vec,
      s0, s1, s2, s3);

    _mm256_storeu_pd (&V1r[index32x   ], t0);
    _mm256_storeu_pd (&V1r[index32x+ 4], s0);

    _mm256_storeu_pd (&V1r[index32x+ 8], t1);
    _mm256_storeu_pd (&V1r[index32x+12], s1);

    _mm256_storeu_pd (&V1r[index32x+16], t2);
    _mm256_storeu_pd (&V1r[index32x+20], s2);

    _mm256_storeu_pd (&V1r[index32x+24], t3);
    _mm256_storeu_pd (&V1r[index32x+28], s3);


    index32x += 32;
  }
}


inline void NPME_TransposeAddUpdateV1_AVX (
  const size_t nCharge, _Complex double *V1, 
  const double *V0_r_align, const double *V0_i_align,
  const double *VX_r_align, const double *VX_i_align,
  const double *VY_r_align, const double *VY_i_align,
  const double *VZ_r_align, const double *VZ_i_align)
//similar to NPME_TransposeV1_AVX, but updates V1 by adding 
//instead of overwriting V1
//input:  nCharge must be a multiple of 4
//        V0_align[nCharge] = aligned at 32 bit boundary
//        VX_align[nCharge] = aligned at 32 bit boundary
//        VY_align[nCharge] = aligned at 32 bit boundary
//        VZ_align[nCharge] = aligned at 32 bit boundary
//output: V1[nCharge][4] = need not be aligned
{
  if (nCharge%4 != 0)
  {
    std::cout << "Error in NPME_TransposeAddUpdateV1_AVX.\n";
    std::cout << "nCharge = " << nCharge << " is not a multiple of 4\n";
    exit(0);
  }

  const size_t m = nCharge/4;

  double *V1r = (double *) V1;

  size_t index4x  = 0;
  size_t index32x = 0;
  for (size_t i = 0; i < m; i++)
  {
    __m256d t0, t1, t2, t3;
    __m256d s0, s1, s2, s3;
    {
      const __m256d V0_r_vec = _mm256_load_pd (&V0_r_align[index4x]);
      const __m256d VX_r_vec = _mm256_load_pd (&VX_r_align[index4x]);
      const __m256d VY_r_vec = _mm256_load_pd (&VY_r_align[index4x]);
      const __m256d VZ_r_vec = _mm256_load_pd (&VZ_r_align[index4x]);

      const __m256d V0_i_vec = _mm256_load_pd (&V0_i_align[index4x]);
      const __m256d VX_i_vec = _mm256_load_pd (&VX_i_align[index4x]);
      const __m256d VY_i_vec = _mm256_load_pd (&VY_i_align[index4x]);
      const __m256d VZ_i_vec = _mm256_load_pd (&VZ_i_align[index4x]);

      index4x += 4;

      NPME_mm256_4x4transpose_pd (
        V0_r_vec, V0_i_vec, VX_r_vec, VX_i_vec,
        t0, t1, t2, t3);
      NPME_mm256_4x4transpose_pd (
        VY_r_vec, VY_i_vec, VZ_r_vec, VZ_i_vec,
        s0, s1, s2, s3);
    }

    __m256d Avec;

    Avec = _mm256_load_pd (&V1r[index32x   ]); t0 = _mm256_add_pd (t0, Avec);
    Avec = _mm256_load_pd (&V1r[index32x+ 4]); s0 = _mm256_add_pd (s0, Avec);

    Avec = _mm256_load_pd (&V1r[index32x+ 8]); t1 = _mm256_add_pd (t1, Avec);
    Avec = _mm256_load_pd (&V1r[index32x+12]); s1 = _mm256_add_pd (s1, Avec);

    Avec = _mm256_load_pd (&V1r[index32x+16]); t2 = _mm256_add_pd (t2, Avec);
    Avec = _mm256_load_pd (&V1r[index32x+20]); s2 = _mm256_add_pd (s2, Avec);

    Avec = _mm256_load_pd (&V1r[index32x+24]); t3 = _mm256_add_pd (t3, Avec);
    Avec = _mm256_load_pd (&V1r[index32x+28]); s3 = _mm256_add_pd (s3, Avec);

    _mm256_storeu_pd (&V1r[index32x   ], t0);
    _mm256_storeu_pd (&V1r[index32x+ 4], s0);

    _mm256_storeu_pd (&V1r[index32x+ 8], t1);
    _mm256_storeu_pd (&V1r[index32x+12], s1);

    _mm256_storeu_pd (&V1r[index32x+16], t2);
    _mm256_storeu_pd (&V1r[index32x+20], s2);

    _mm256_storeu_pd (&V1r[index32x+24], t3);
    _mm256_storeu_pd (&V1r[index32x+28], s3);


    index32x += 32;
  }
}


inline void NPME_Real2Complex_AVX (const size_t N, _Complex double *z, 
  const double *xAlignTmp, const double *yAlignTmp)
//input:  xAlignTmp[N] = aligned at 32 bit boundary
//        yAlignTmp[N] = aligned at 32 bit boundary
//        N must be a multiple of 4
//output: z[N] need not be aligned
//        where z[N] = x[N] + I*y[N]
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Real2Complex_AVX.\n";
    std::cout << "N = " << N << " is not a multiple of 4\n";
    exit(0);
  }

  const size_t m  = N/4;
  double *zr      = (double *) z;
  //zr[2*N] = zr[8*m]

  size_t indexCrd   = 0;
  size_t indexAlign = 0;
  for (size_t i = 0; i < m; i++)
  {
    const __m256d x = _mm256_load_pd (&xAlignTmp[indexAlign]);
    const __m256d y = _mm256_load_pd (&yAlignTmp[indexAlign]);
    indexAlign += 4;

    __m256d t0, t1;
    NPME_mm256_2x4transpose_pd (x, y, t0, t1);

    double *X  = &zr[indexCrd];
    _mm256_storeu_pd (&X[0], t0);
    _mm256_storeu_pd (&X[4], t1);
    indexCrd += 8;
  }
}
inline void NPME_Complex2Real_AVX (const size_t N, const _Complex double *z, 
  double *xAlignTmp, double *yAlignTmp)
//input:  z[N] need not be aligned
//        N must be a multiple of 4
//output: xAlignTmp[N] = aligned at 32 bit boundary
//        yAlignTmp[N] = aligned at 32 bit boundary
//        where z[N] = x[N] + I*y[N]
{
  if (N%4 != 0)
  {
    std::cout << "Error in NPME_Complex2Real_AVX.\n";
    std::cout << "N = " << N << " is not a multiple of 4\n";
    exit(0);
  }

  const size_t m    = N/4;
  const double *zr  = (const double *) z;
  //zr[2*N] = zr[8*m]

  size_t indexCrd   = 0;
  size_t indexAlign = 0;
  for (size_t i = 0; i < m; i++)
  {
    const double *X  = &zr[indexCrd];
    const __m256d t0 = _mm256_loadu_pd (&X[0]);
    const __m256d t1 = _mm256_loadu_pd (&X[4]);
    indexCrd += 8;

    __m256d x, y;
    NPME_mm256_4x2transpose_pd (t0, t1, x, y);
    _mm256_store_pd (&xAlignTmp[indexAlign], x);
    _mm256_store_pd (&yAlignTmp[indexAlign], y);
    indexAlign += 4;
  }
}

inline void NPME_TransformQrealCoord_4x_AVX (const size_t nCharge, 
  const double *charge, const double *coord, double *qReCrd_4x)
//input:  charge[nCharge], coord[3*nCharge] not aligned
//output: qReCrd_4x[4*M] = aligned at 32 bit boundary
//        qReCrd_4x[] = {q0, q1, q2, q3, 
//                       x0, x1, x2, x3, 
//                       y0, y1, y2, y3, 
//                       z0, z1, z2, z3, 
//                       q4, q5, q6, q7, ...
//if (nCharge%4 != 0), q is padded with zeros and x,y,z are padded with
//NPME_Pot_Xpad = large number to prevent |r1-r2| > 0
{
  const size_t remain = nCharge%4;
  const size_t nLoop  = (nCharge-remain)/4;


  size_t index4i  = 0;
  size_t index12i = 0;
  size_t index16i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    const double *X  = &coord[index12i];

    __m256d qVec, xVec, yVec, zVec;
    const __m256d t0  = _mm256_loadu_pd (&X[0]);
    const __m256d t1  = _mm256_loadu_pd (&X[4]);
    const __m256d t2  = _mm256_loadu_pd (&X[8]);
    qVec              = _mm256_loadu_pd (&charge[index4i]);

    NPME_mm256_4x3transpose_pd (t0, t1, t2, xVec, yVec, zVec);
    _mm256_store_pd (&qReCrd_4x[index16i   ], qVec);
    _mm256_store_pd (&qReCrd_4x[index16i+ 4], xVec);
    _mm256_store_pd (&qReCrd_4x[index16i+ 8], yVec);
    _mm256_store_pd (&qReCrd_4x[index16i+12], zVec);

    index4i  += 4;
    index12i += 12;
    index16i += 16;
  }

  if (remain > 0)
  {
    const double X = NPME_Pot_Xpad;
    double q4[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
    double x4[4]  __attribute__((aligned(64))) = {X, X, X, X};
    double y4[4]  __attribute__((aligned(64))) = {X, X, X, X};
    double z4[4]  __attribute__((aligned(64))) = {X, X, X, X};

    for (size_t i = 0; i < remain; i++)
    {
      size_t index = 4*nLoop+i;
      q4[i] = charge[index];
      x4[i] = coord[3*index  ];
      y4[i] = coord[3*index+1];
      z4[i] = coord[3*index+2];
    }
    __m256d qVec, xVec, yVec, zVec;
    qVec  = _mm256_load_pd (q4);
    xVec  = _mm256_load_pd (x4);
    yVec  = _mm256_load_pd (y4);
    zVec  = _mm256_load_pd (z4);

    _mm256_store_pd (&qReCrd_4x[index16i   ], qVec);
    _mm256_store_pd (&qReCrd_4x[index16i+ 4], xVec);
    _mm256_store_pd (&qReCrd_4x[index16i+ 8], yVec);
    _mm256_store_pd (&qReCrd_4x[index16i+12], zVec);
  }
}
 
inline void NPME_TransformQcomplexCoord_4x_AVX (const size_t nCharge, 
  const _Complex double *charge, const double *coord, double *qCoCrd_4x)
//input:  charge[nCharge][2], coord[nCharge][3] not aligned
//output: qCoCrd_4x[5*M] = aligned at 32 bit boundary
//        qCoCrd_4x[] = {q0r, q1r, q2r, q3r, 
//                       q0i, q1i, q2i, q3i, 
//                       x0,  x1,  x2,  x3, 
//                       y0,  y1,  y2,  y3, 
//                       z0,  z1,  z2,  z3, 
//                       q4i, q5i, q6i, q7i, ...
//if (nCharge%4 != 0), q is padded with zeros and x,y,z are padded with
//NPME_Pot_Xpad = large number to prevent |r1-r2| > 0
{
  const size_t remain       = nCharge%4;
  const size_t nLoop        = (nCharge-remain)/4;
  const double *chargeReal  = (const double *) charge;

  size_t index8i  = 0;
  size_t index12i = 0;
  size_t index20i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    //complex charges
    __m256d t0, t1, t2, s0, s1, s2;
    t0 = _mm256_loadu_pd (&chargeReal[index8i  ]);
    t1 = _mm256_loadu_pd (&chargeReal[index8i+4]);
    NPME_mm256_4x2transpose_pd (t0, t1, s0, s1);
    _mm256_store_pd (&qCoCrd_4x[index20i   ], s0);
    _mm256_store_pd (&qCoCrd_4x[index20i+ 4], s1);

    //coord
    const double *X  = &coord[index12i];
    t0  = _mm256_loadu_pd (&X[0]);
    t1  = _mm256_loadu_pd (&X[4]);
    t2  = _mm256_loadu_pd (&X[8]);
    NPME_mm256_4x3transpose_pd (t0, t1, t2, s0, s1, s2);
    _mm256_store_pd (&qCoCrd_4x[index20i+ 8], s0);
    _mm256_store_pd (&qCoCrd_4x[index20i+12], s1);
    _mm256_store_pd (&qCoCrd_4x[index20i+16], s2);

    index8i  += 8;
    index12i += 12;
    index20i += 20;
  }

  if (remain > 0)
  {
    const double X = NPME_Pot_Xpad;
    double q4R[4] __attribute__((aligned(64))) = {0, 0, 0, 0};
    double q4I[4] __attribute__((aligned(64))) = {0, 0, 0, 0};
    double x4[4]  __attribute__((aligned(64))) = {X, X, X, X};
    double y4[4]  __attribute__((aligned(64))) = {X, X, X, X};
    double z4[4]  __attribute__((aligned(64))) = {X, X, X, X};

    for (size_t i = 0; i < remain; i++)
    {
      size_t index = 4*nLoop+i;
      q4R[i] = creal(charge[index]);
      q4I[i] = cimag(charge[index]);
      x4 [i] = coord[3*index  ];
      y4 [i] = coord[3*index+1];
      z4 [i] = coord[3*index+2];
    }
    __m256d t0;
    t0 = _mm256_load_pd (q4R);  _mm256_store_pd (&qCoCrd_4x[index20i   ], t0);
    t0 = _mm256_load_pd (q4I);  _mm256_store_pd (&qCoCrd_4x[index20i+ 4], t0);
    t0 = _mm256_load_pd ( x4);  _mm256_store_pd (&qCoCrd_4x[index20i+ 8], t0);
    t0 = _mm256_load_pd ( y4);  _mm256_store_pd (&qCoCrd_4x[index20i+12], t0);
    t0 = _mm256_load_pd ( z4);  _mm256_store_pd (&qCoCrd_4x[index20i+16], t0);
  }
}
inline void NPME_TransformQreal_4x_AVX (const size_t nCharge, 
  const double *charge, double *charge_4x)
//input:  charge[nCharge] not aligned
//output: charge[4*M] = aligned at 32 bit boundary
//        charge[]    = {q0, q1, q2, q3, 
//                       q4, q5, q6, q7, ...
//if (nCharge%4 != 0), q is padded with zeros 
{
  const size_t remain = nCharge%4;
  const size_t nLoop  = (nCharge-remain)/4;


  size_t index4i  = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d qVec = _mm256_loadu_pd (&charge[index4i]);
    _mm256_store_pd (&charge_4x[index4i], qVec);
    index4i += 4;
  }

  if (remain > 0)
  {

    double q4[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
    for (size_t i = 0; i < remain; i++)
    {
      size_t index = 4*nLoop+i;
      q4[i] = charge[index];
    }
    __m256d qVec;
    qVec  = _mm256_load_pd (q4);
    _mm256_store_pd (&charge_4x[index4i], qVec);
  }
}
inline void NPME_TransformQcomplex_4x_AVX (const size_t nCharge, 
  const _Complex double *charge, double *charge_4x)
//input:  charge[nCharge][2] not aligned
//output: charge_4x[4*M] = aligned at 32 bit boundary
//        charge_4x[]    = {q0r, q1r, q2r, q3r, 
//                          q0i, q1i, q2i, q3i, 
//                          q4r, q5r, q6r, q7r, 
//if (nCharge%4 != 0), q is padded with zeros 
{
  const size_t remain       = nCharge%4;
  const size_t nLoop        = (nCharge-remain)/4;
  const double *chargeReal  = (const double *) charge;

  size_t index8i  = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    //complex charges
    __m256d t0, t1, s0, s1;
    t0 = _mm256_loadu_pd (&chargeReal[index8i  ]);
    t1 = _mm256_loadu_pd (&chargeReal[index8i+4]);
    NPME_mm256_4x2transpose_pd (t0, t1, s0, s1);
    _mm256_store_pd (&charge_4x[index8i  ], s0);
    _mm256_store_pd (&charge_4x[index8i+4], s1);

    index8i  += 8;
  }

  if (remain > 0)
  {
    double q4R[4] __attribute__((aligned(64))) = {0, 0, 0, 0};
    double q4I[4] __attribute__((aligned(64))) = {0, 0, 0, 0};

    for (size_t i = 0; i < remain; i++)
    {
      size_t index = 4*nLoop+i;
      q4R[i] = creal(charge[index]);
      q4I[i] = cimag(charge[index]);
    }
    __m256d t0;
    t0 = _mm256_load_pd (q4R);  _mm256_store_pd (&charge_4x[index8i  ], t0);
    t0 = _mm256_load_pd (q4I);  _mm256_store_pd (&charge_4x[index8i+4], t0);
  }
}

inline void NPME_TransformCoord_4x_AVX (const size_t nCharge, 
  const double *coord, double *coord_4x)
//input:  coord[3*nCharge] not aligned
//output: coord_4x[3*M] = aligned at 32 bit boundary
//        coord_4x[]  = {x0, x1, x2, x3, 
//                       y0, y1, y2, y3, 
//                       z0, z1, z2, z3, 
//                       x4, x5, x6, x7, ...
//if (nCharge%4 != 0), x,y,z are padded with
//NPME_Pot_Xpad = large number to prevent |r1-r2| > 0
{
  const size_t remain = nCharge%4;
  const size_t nLoop  = (nCharge-remain)/4;

  size_t index12i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    const double *X  = &coord[index12i];

    __m256d xVec, yVec, zVec;
    const __m256d t0  = _mm256_loadu_pd (&X[0]);
    const __m256d t1  = _mm256_loadu_pd (&X[4]);
    const __m256d t2  = _mm256_loadu_pd (&X[8]);

    NPME_mm256_4x3transpose_pd (t0, t1, t2, xVec, yVec, zVec);
    _mm256_store_pd (&coord_4x[index12i  ], xVec);
    _mm256_store_pd (&coord_4x[index12i+4], yVec);
    _mm256_store_pd (&coord_4x[index12i+8], zVec);

    index12i += 12;
  }

  if (remain > 0)
  {
    const double X = NPME_Pot_Xpad;
    double x4[4]  __attribute__((aligned(64))) = {X, X, X, X};
    double y4[4]  __attribute__((aligned(64))) = {X, X, X, X};
    double z4[4]  __attribute__((aligned(64))) = {X, X, X, X};

    for (size_t i = 0; i < remain; i++)
    {
      size_t index = 4*nLoop+i;
      x4[i] = coord[3*index  ];
      y4[i] = coord[3*index+1];
      z4[i] = coord[3*index+2];
    }
    __m256d xVec, yVec, zVec;
    xVec  = _mm256_load_pd (x4);
    yVec  = _mm256_load_pd (y4);
    zVec  = _mm256_load_pd (z4);

    _mm256_store_pd (&coord_4x[index12i   ], xVec);
    _mm256_store_pd (&coord_4x[index12i+ 4], yVec);
    _mm256_store_pd (&coord_4x[index12i+ 8], zVec);
  }
}

inline void NPME_TransformRealV1_4x_2_V1_AVX (const size_t nCharge, 
  const double *V1_4x, double *V1)
//input:  V1_4x[4*M] = = aligned at 32 bit boundary
//        V1_4x[] = {V0[0], V0[1], V0[2], V0[3],
//                   VX[0], VX[1], VX[2], VX[3],
//                   VY[0], VY[1], VY[2], VY[3],
//                   VZ[0], VZ[1], VZ[2], VZ[3],
//                   V0[4], V0[5], V0[6], V0[7],..
//output: V1[4*nCharge] = not aligned
//                      = {V0[0], VX[0], VY[0], VZ[0],
//                         V0[1], VX[1], VY[1], VZ[1], ..
{
  const size_t remain = nCharge%4;
  const size_t nLoop  = (nCharge-remain)/4;

  size_t index16i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    const __m256d V0  = _mm256_load_pd (&V1_4x[index16i   ]);
    const __m256d VX  = _mm256_load_pd (&V1_4x[index16i+ 4]);
    const __m256d VY  = _mm256_load_pd (&V1_4x[index16i+ 8]);
    const __m256d VZ  = _mm256_load_pd (&V1_4x[index16i+12]);

    __m256d t0, t1, t2, t3;
    NPME_mm256_4x4transpose_pd (V0, VX, VY, VZ, t0, t1, t2, t3);

    _mm256_storeu_pd (&V1[index16i   ], t0);
    _mm256_storeu_pd (&V1[index16i+ 4], t1);
    _mm256_storeu_pd (&V1[index16i+ 8], t2);
    _mm256_storeu_pd (&V1[index16i+12], t3);

    index16i += 16;
  }

  if (remain > 0)
  {
    double *V1_loc        = &V1[16*nLoop];
    const double *V0_loc  = &V1_4x[index16i   ];
    const double *VX_loc  = &V1_4x[index16i+ 4];
    const double *VY_loc  = &V1_4x[index16i+ 8];
    const double *VZ_loc  = &V1_4x[index16i+12];

    size_t index4i = 0;
    for (size_t i = 0; i < remain; i++)
    {
      V1_loc[index4i  ] = V0_loc[i];
      V1_loc[index4i+1] = VX_loc[i];
      V1_loc[index4i+2] = VY_loc[i];
      V1_loc[index4i+3] = VZ_loc[i];
      index4i += 4;
    }
  }
}



inline void NPME_TransformUpdateRealV1_4x_2_V1_AVX (const size_t nCharge, 
  const double *V1_4x, double *V1)
//input:  V1_4x[4*M] = = aligned at 32 bit boundary
//        V1_4x[] = {V0[0], V0[1], V0[2], V0[3],
//                   VX[0], VX[1], VX[2], VX[3],
//                   VY[0], VY[1], VY[2], VY[3],
//                   VZ[0], VZ[1], VZ[2], VZ[3],
//                   V0[4], V0[5], V0[6], V0[7],..
//output: V1[4*nCharge] = not aligned
//                      = {V0[0], VX[0], VY[0], VZ[0],
//                         V0[1], VX[1], VY[1], VZ[1], ..
{
  const size_t remain = nCharge%4;
  const size_t nLoop  = (nCharge-remain)/4;

  size_t index16i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    const __m256d V0  = _mm256_load_pd (&V1_4x[index16i   ]);
    const __m256d VX  = _mm256_load_pd (&V1_4x[index16i+ 4]);
    const __m256d VY  = _mm256_load_pd (&V1_4x[index16i+ 8]);
    const __m256d VZ  = _mm256_load_pd (&V1_4x[index16i+12]);

    __m256d t0, t1, t2, t3;
    NPME_mm256_4x4transpose_pd (V0, VX, VY, VZ, t0, t1, t2, t3);

    __m256d s0  = _mm256_loadu_pd (&V1[index16i   ]);
    __m256d s1  = _mm256_loadu_pd (&V1[index16i+ 4]);
    __m256d s2  = _mm256_loadu_pd (&V1[index16i+ 8]);
    __m256d s3  = _mm256_loadu_pd (&V1[index16i+12]);

    t0 = _mm256_add_pd (s0, t0);
    t1 = _mm256_add_pd (s1, t1);
    t2 = _mm256_add_pd (s2, t2);
    t3 = _mm256_add_pd (s3, t3);

    _mm256_storeu_pd (&V1[index16i   ], t0);
    _mm256_storeu_pd (&V1[index16i+ 4], t1);
    _mm256_storeu_pd (&V1[index16i+ 8], t2);
    _mm256_storeu_pd (&V1[index16i+12], t3);

    index16i += 16;
  }

  if (remain > 0)
  {
    double *V1_loc        = &V1[16*nLoop];
    const double *V0_loc  = &V1_4x[index16i   ];
    const double *VX_loc  = &V1_4x[index16i+ 4];
    const double *VY_loc  = &V1_4x[index16i+ 8];
    const double *VZ_loc  = &V1_4x[index16i+12];

    size_t index4i = 0;
    for (size_t i = 0; i < remain; i++)
    {
      V1_loc[index4i  ] += V0_loc[i];
      V1_loc[index4i+1] += VX_loc[i];
      V1_loc[index4i+2] += VY_loc[i];
      V1_loc[index4i+3] += VZ_loc[i];
      index4i += 4;
    }
  }
}


inline void NPME_TransformComplexV1_4x_2_V1_AVX (const size_t nCharge, 
  const double *V1_4x, _Complex double *V1)
//input:  V1_4x[4*M] = aligned at 32 bit boundary
//        V1_4x[] = {V0r[0], V0r[1], V0r[2], V0r[3],
//                   V0i[0], V0i[1], V0i[2], V0i[3],
//                   VXr[0], VXr[1], VXr[2], VXr[3],
//                   VXi[0], VXi[1], VXi[2], VXi[3],
//                   VYr[0], VYr[1], VYr[2], VYr[3],
//                   VYi[0], VYi[1], VYi[2], VYi[3],
//                   VZr[0], VZr[1], VZr[2], VZr[3],
//                   VZi[0], VZi[1], VZi[2], VZi[3],
//                   V0r[4], V0r[5], V0r[6], V0r[7],
//output: V1[4*nCharge]= not aligned
//  = {V0r[0], V0i[0], VXr[0], VXi[0], VYr[0], VYi[0], VZr[0], VZi[0],
//     V0r[1], V0i[1], VXr[1], VXi[1], VYr[1], VYi[1], VZr[1], VZi[1],
//     ...
{
  const size_t remain = nCharge%4;
  const size_t nLoop  = (nCharge-remain)/4;

  double *V1r = (double *) V1;

  size_t index32i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d a0, a1, a2, a3, b0, b1, b2, b3;
    __m256d c0, c1, c2, c3, d0, d1, d2, d3;

    //V0r, V0i, VXr, VXi
    a0  = _mm256_load_pd (&V1_4x[index32i   ]);
    a1  = _mm256_load_pd (&V1_4x[index32i+ 4]);
    a2  = _mm256_load_pd (&V1_4x[index32i+ 8]);
    a3  = _mm256_load_pd (&V1_4x[index32i+12]);

    NPME_mm256_4x4transpose_pd (a0, a1, a2, a3, b0, b1, b2, b3);

    //VYr, VYi, VZr, VZi
    c0  = _mm256_load_pd (&V1_4x[index32i+16]);
    c1  = _mm256_load_pd (&V1_4x[index32i+20]);
    c2  = _mm256_load_pd (&V1_4x[index32i+24]);
    c3  = _mm256_load_pd (&V1_4x[index32i+28]);

    NPME_mm256_4x4transpose_pd (c0, c1, c2, c3, d0, d1, d2, d3);


    _mm256_storeu_pd (&V1r[index32i   ], b0);
    _mm256_storeu_pd (&V1r[index32i+ 4], d0);
    _mm256_storeu_pd (&V1r[index32i+ 8], b1);
    _mm256_storeu_pd (&V1r[index32i+12], d1);
    _mm256_storeu_pd (&V1r[index32i+16], b2);
    _mm256_storeu_pd (&V1r[index32i+20], d2);
    _mm256_storeu_pd (&V1r[index32i+24], b3);
    _mm256_storeu_pd (&V1r[index32i+28], d3);

    index32i += 32;
  }

  if (remain > 0)
  {
    _Complex double *V1_loc = &V1[16*nLoop];
    const double *V0r_loc   = &V1_4x[index32i   ];
    const double *V0i_loc   = &V1_4x[index32i+ 4];
    const double *VXr_loc   = &V1_4x[index32i+ 8];
    const double *VXi_loc   = &V1_4x[index32i+12];
    const double *VYr_loc   = &V1_4x[index32i+16];
    const double *VYi_loc   = &V1_4x[index32i+20];
    const double *VZr_loc   = &V1_4x[index32i+24];
    const double *VZi_loc   = &V1_4x[index32i+28];

    size_t index4i = 0;
    for (size_t i = 0; i < remain; i++)
    {
      V1_loc[index4i  ] = V0r_loc[i] + I*V0i_loc[i];
      V1_loc[index4i+1] = VXr_loc[i] + I*VXi_loc[i];
      V1_loc[index4i+2] = VYr_loc[i] + I*VYi_loc[i];
      V1_loc[index4i+3] = VZr_loc[i] + I*VZi_loc[i];
      index4i += 4;
    }
  }
}

inline void NPME_TransformUpdateComplexV1_4x_2_V1_AVX (const size_t nCharge, 
  const double *V1_4x, _Complex double *V1)
//input:  V1_4x[4*M] = aligned at 32 bit boundary
//        V1_4x[] = {V0r[0], V0r[1], V0r[2], V0r[3],
//                   V0i[0], V0i[1], V0i[2], V0i[3],
//                   VXr[0], VXr[1], VXr[2], VXr[3],
//                   VXi[0], VXi[1], VXi[2], VXi[3],
//                   VYr[0], VYr[1], VYr[2], VYr[3],
//                   VYi[0], VYi[1], VYi[2], VYi[3],
//                   VZr[0], VZr[1], VZr[2], VZr[3],
//                   VZi[0], VZi[1], VZi[2], VZi[3],
//                   V0r[4], V0r[5], V0r[6], V0r[7],
//output: V1[4*nCharge]= not aligned
//  = {V0r[0], V0i[0], VXr[0], VXi[0], VYr[0], VYi[0], VZr[0], VZi[0],
//     V0r[1], V0i[1], VXr[1], VXi[1], VYr[1], VYi[1], VZr[1], VZi[1],
//     ...
{
  const size_t remain = nCharge%4;
  const size_t nLoop  = (nCharge-remain)/4;

  double *V1r = (double *) V1;

  size_t index32i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d a0, a1, a2, a3, b0, b1, b2, b3;
    __m256d d0, d1, d2, d3;

    //V0r, V0i, VXr, VXi
    a0  = _mm256_load_pd (&V1_4x[index32i   ]);
    a1  = _mm256_load_pd (&V1_4x[index32i+ 4]);
    a2  = _mm256_load_pd (&V1_4x[index32i+ 8]);
    a3  = _mm256_load_pd (&V1_4x[index32i+12]);

    NPME_mm256_4x4transpose_pd (a0, a1, a2, a3, b0, b1, b2, b3);

    //VYr, VYi, VZr, VZi
    a0  = _mm256_load_pd (&V1_4x[index32i+16]);
    a1  = _mm256_load_pd (&V1_4x[index32i+20]);
    a2  = _mm256_load_pd (&V1_4x[index32i+24]);
    a3  = _mm256_load_pd (&V1_4x[index32i+28]);

    NPME_mm256_4x4transpose_pd (a0, a1, a2, a3, d0, d1, d2, d3);

    __m256d t0;
    t0 = _mm256_loadu_pd (&V1r[index32i  ]);
    b0 = _mm256_add_pd (t0, b0);
    _mm256_storeu_pd (&V1r[index32i   ], b0);

    t0 = _mm256_loadu_pd (&V1r[index32i+ 4]);
    d0 = _mm256_add_pd (t0, d0);
    _mm256_storeu_pd (&V1r[index32i+ 4], d0);

    t0 = _mm256_loadu_pd (&V1r[index32i+ 8]);
    b1 = _mm256_add_pd (t0, b1);
    _mm256_storeu_pd (&V1r[index32i+ 8], b1);

    t0 = _mm256_loadu_pd (&V1r[index32i+12]);
    d1 = _mm256_add_pd (t0, d1);
    _mm256_storeu_pd (&V1r[index32i+12], d1);

    t0 = _mm256_loadu_pd (&V1r[index32i+16]);
    b2 = _mm256_add_pd (t0, b2);
    _mm256_storeu_pd (&V1r[index32i+16], b2);

    t0 = _mm256_loadu_pd (&V1r[index32i+20]);
    d2 = _mm256_add_pd (t0, d2);
    _mm256_storeu_pd (&V1r[index32i+20], d2);

    t0 = _mm256_loadu_pd (&V1r[index32i+24]);
    b3 = _mm256_add_pd (t0, b3);
    _mm256_storeu_pd (&V1r[index32i+24], b3);

    t0 = _mm256_loadu_pd (&V1r[index32i+28]);
    d3 = _mm256_add_pd (t0, d3);
    _mm256_storeu_pd (&V1r[index32i+28], d3);

    index32i += 32;
  }

  if (remain > 0)
  {
    _Complex double *V1_loc = &V1[16*nLoop];
    const double *V0r_loc   = &V1_4x[index32i   ];
    const double *V0i_loc   = &V1_4x[index32i+ 4];
    const double *VXr_loc   = &V1_4x[index32i+ 8];
    const double *VXi_loc   = &V1_4x[index32i+12];
    const double *VYr_loc   = &V1_4x[index32i+16];
    const double *VYi_loc   = &V1_4x[index32i+20];
    const double *VZr_loc   = &V1_4x[index32i+24];
    const double *VZi_loc   = &V1_4x[index32i+28];

    size_t index4i = 0;
    for (size_t i = 0; i < remain; i++)
    {
      V1_loc[index4i  ] += V0r_loc[i] + I*V0i_loc[i];
      V1_loc[index4i+1] += VXr_loc[i] + I*VXi_loc[i];
      V1_loc[index4i+2] += VYr_loc[i] + I*VYi_loc[i];
      V1_loc[index4i+3] += VZr_loc[i] + I*VZi_loc[i];
      index4i += 4;
    }
  }
}





#endif //NPME_USE_AVX

#if NPME_USE_AVX_512
inline void NPME_TransformQrealCoord_8x_AVX_512 (const size_t nCharge, 
  const double *charge, const double *coord, double *qReCrd_8x)
//input:  charge[nCharge], coord[3*nCharge] not aligned
//output: qReCrd_8x[4*M] = aligned at 64 bit boundary
//        qReCrd_8x[] = {q0, q1, q2, q3, q4, q5, q6, q7, 
//                       x0, x1, x2, x3, x4, x5, x6, x7, 
//                       y0, y1, y2, y3, y4, y5, y6, y7, 
//                       z0, z1, z2, z3, z4, z5, z6, z7, 
//                       q8, q9, q10,..
//if (nCharge%8 != 0), q is padded with zeros and x,y,z are padded with
//NPME_Pot_Xpad = large number to prevent |r1-r2| > 0
{
  const size_t remain = nCharge%8;
  const size_t nLoop  = (nCharge-remain)/8;


  size_t index8i  = 0;
  size_t index24i = 0;
  size_t index32i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    const double *X  = &coord[index24i];

    __m512d qVec, xVec, yVec, zVec;
    const __m512d t0  = _mm512_loadu_pd (&X[0]);
    const __m512d t1  = _mm512_loadu_pd (&X[8]);
    const __m512d t2  = _mm512_loadu_pd (&X[16]);
    qVec              = _mm512_loadu_pd (&charge[index8i]);

    NPME_mm512_8x3transpose_pd (t0, t1, t2, xVec, yVec, zVec);
    _mm512_store_pd (&qReCrd_8x[index32i   ], qVec);
    _mm512_store_pd (&qReCrd_8x[index32i+ 8], xVec);
    _mm512_store_pd (&qReCrd_8x[index32i+16], yVec);
    _mm512_store_pd (&qReCrd_8x[index32i+24], zVec);

    index8i  += 8;
    index24i += 24;
    index32i += 32;
  }

  if (remain > 0)
  {
    const double X = NPME_Pot_Xpad;
    double q8[8]  __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
    double x8[8]  __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};
    double y8[8]  __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};
    double z8[8]  __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};

    for (size_t i = 0; i < remain; i++)
    {
      size_t index = 8*nLoop+i;
      q8[i] = charge[index];
      x8[i] = coord[3*index  ];
      y8[i] = coord[3*index+1];
      z8[i] = coord[3*index+2];
    }
    __m512d qVec, xVec, yVec, zVec;
    qVec  = _mm512_load_pd (q8);
    xVec  = _mm512_load_pd (x8);
    yVec  = _mm512_load_pd (y8);
    zVec  = _mm512_load_pd (z8);

    _mm512_store_pd (&qReCrd_8x[index32i   ], qVec);
    _mm512_store_pd (&qReCrd_8x[index32i+ 8], xVec);
    _mm512_store_pd (&qReCrd_8x[index32i+16], yVec);
    _mm512_store_pd (&qReCrd_8x[index32i+24], zVec);
  }
}
inline void NPME_TransformQreal_8x_AVX_512 (const size_t nCharge, 
  const double *charge, double *charge_8x)
//input:  charge[nCharge] not aligned
//output: charge[8*M] = aligned at 64 bit boundary
//        charge[]    = {q0, q1, q2, q3, q4, q5, q6, q7,
//                       q8, q9, q10, ...
//if (nCharge%8 != 0), q is padded with zeros 
{
  const size_t remain = nCharge%8;
  const size_t nLoop  = (nCharge-remain)/8;


  size_t index8i  = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d qVec = _mm512_loadu_pd (&charge[index8i]);
    _mm512_store_pd (&charge_8x[index8i], qVec);
    index8i += 8;
  }

  if (remain > 0)
  {

    double q8[8]  __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < remain; i++)
    {
      size_t index = 8*nLoop+i;
      q8[i] = charge[index];
    }
    __m512d qVec;
    qVec  = _mm512_load_pd (q8);
    _mm512_store_pd (&charge_8x[index8i], qVec);
  }
}
inline void NPME_TransformQcomplexCoord_8x_AVX_512 (const size_t nCharge, 
  const _Complex double *charge, const double *coord, double *qCoCrd_8x)
//input:  charge[nCharge][2], coord[nCharge][3] not aligned
//output: qCoCrd_8x[5*M] = aligned at 64 bit boundary
//        qCoCrd_8x[] = {q0r, q1r, q2r, q3r, q4r, q5r, q6r, q7r, 
//                       q0i, q1i, q2i, q3i, q4i, q5i, q6i, q7i, 
//                       x0,  x1,  x2,  x3,  x4,  x5,  x6,  x7, 
//                       y0,  y1,  y2,  y3,  y4,  y5,  y6,  y7, 
//                       z0,  z1,  z2,  z3,  z4,  z5,  z6,  z7, 
//                       q8r, q9r, q10r,..
//if (nCharge%8 != 0), q is padded with zeros and x,y,z are padded with
//NPME_Pot_Xpad = large number to prevent |r1-r2| > 0
{
  const size_t remain       = nCharge%8;
  const size_t nLoop        = (nCharge-remain)/8;
  const double *chargeReal  = (const double *) charge;

  size_t index16i = 0;
  size_t index24i = 0;
  size_t index40i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    //complex charges
    __m512d t0, t1, t2, s0, s1, s2;
    t0 = _mm512_loadu_pd (&chargeReal[index16i  ]);
    t1 = _mm512_loadu_pd (&chargeReal[index16i+8]);
    NPME_mm512_8x2transpose_pd (t0, t1, s0, s1);
    _mm512_store_pd (&qCoCrd_8x[index40i   ], s0);
    _mm512_store_pd (&qCoCrd_8x[index40i+ 8], s1);

    //coord
    const double *X  = &coord[index24i];
    t0  = _mm512_loadu_pd (&X[0]);
    t1  = _mm512_loadu_pd (&X[8]);
    t2  = _mm512_loadu_pd (&X[16]);
    NPME_mm512_8x3transpose_pd (t0, t1, t2, s0, s1, s2);
    _mm512_store_pd (&qCoCrd_8x[index40i+16], s0);
    _mm512_store_pd (&qCoCrd_8x[index40i+24], s1);
    _mm512_store_pd (&qCoCrd_8x[index40i+32], s2);

    index16i += 16;
    index24i += 24;
    index40i += 40;
  }

  if (remain > 0)
  {
    const double X = NPME_Pot_Xpad;
    double q8R[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
    double q8I[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
    double x8[8]  __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};
    double y8[8]  __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};
    double z8[8]  __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};

    for (size_t i = 0; i < remain; i++)
    {
      size_t index = 8*nLoop+i;
      q8R[i] = creal(charge[index]);
      q8I[i] = cimag(charge[index]);
      x8 [i] = coord[3*index  ];
      y8 [i] = coord[3*index+1];
      z8 [i] = coord[3*index+2];
    }
    __m512d t0;
    t0 = _mm512_load_pd (q8R);  _mm512_store_pd (&qCoCrd_8x[index40i   ], t0);
    t0 = _mm512_load_pd (q8I);  _mm512_store_pd (&qCoCrd_8x[index40i+ 8], t0);
    t0 = _mm512_load_pd ( x8);  _mm512_store_pd (&qCoCrd_8x[index40i+16], t0);
    t0 = _mm512_load_pd ( y8);  _mm512_store_pd (&qCoCrd_8x[index40i+24], t0);
    t0 = _mm512_load_pd ( z8);  _mm512_store_pd (&qCoCrd_8x[index40i+32], t0);
  }
}
inline void NPME_TransformQcomplex_8x_AVX_512 (const size_t nCharge, 
  const _Complex double *charge, double *charge_8x)
//input:  charge[nCharge][2] not aligned
//output: charge_8x[4*M] = aligned at 64 bit boundary
//        charge_8x[]    = {q0r, q1r, q2r, q3r, q4r, q5r, q6r, q7r, 
//                          q0i, q1i, q2i, q3i, q4i, q5i, q6i, q7i, 
//                          q8r, q9r, q10r,..
//if (nCharge%8 != 0), q is padded with zeros 
{
  const size_t remain       = nCharge%8;
  const size_t nLoop        = (nCharge-remain)/8;
  const double *chargeReal  = (const double *) charge;

  size_t index16i  = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    //complex charges
    __m512d t0, t1, s0, s1;
    t0 = _mm512_loadu_pd (&chargeReal[index16i  ]);
    t1 = _mm512_loadu_pd (&chargeReal[index16i+8]);
    NPME_mm512_8x2transpose_pd (t0, t1, s0, s1);
    _mm512_store_pd (&charge_8x[index16i  ], s0);
    _mm512_store_pd (&charge_8x[index16i+8], s1);

    index16i  += 16;
  }

  if (remain > 0)
  {
    double q8R[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
    double q8I[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};

    for (size_t i = 0; i < remain; i++)
    {
      size_t index = 8*nLoop+i;
      q8R[i] = creal(charge[index]);
      q8I[i] = cimag(charge[index]);
    }
    __m512d t0;
    t0 = _mm512_load_pd (q8R);  _mm512_store_pd (&charge_8x[index16i  ], t0);
    t0 = _mm512_load_pd (q8I);  _mm512_store_pd (&charge_8x[index16i+8], t0);
  }
}

inline void NPME_TransformCoord_8x_AVX_512 (const size_t nCharge, 
  const double *coord, double *coord_8x)
//input:  coord[3*nCharge] not aligned
//output: coord_8x[3*M] = aligned at 64 bit boundary
//        coord_8x[]  = {x0, x1, x2, x3, x4, x5, x6, x7, 
//                       y0, y1, y2, y3, y4, y5, y6, y7, 
//                       z0, z1, z2, z3, z4, z5, z6, z7, 
//                       x8, x9, x10,..
//if (nCharge%8 != 0), x,y,z are padded with
//NPME_Pot_Xpad = large number to prevent |r1-r2| > 0
{
  const size_t remain = nCharge%8;
  const size_t nLoop  = (nCharge-remain)/8;

  size_t index24i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    const double *X  = &coord[index24i];

    __m512d xVec, yVec, zVec;
    const __m512d t0  = _mm512_loadu_pd (&X[ 0]);
    const __m512d t1  = _mm512_loadu_pd (&X[ 8]);
    const __m512d t2  = _mm512_loadu_pd (&X[16]);

    NPME_mm512_8x3transpose_pd (t0, t1, t2, xVec, yVec, zVec);
    _mm512_store_pd (&coord_8x[index24i   ], xVec);
    _mm512_store_pd (&coord_8x[index24i+ 8], yVec);
    _mm512_store_pd (&coord_8x[index24i+16], zVec);

    index24i += 24;
  }

  if (remain > 0)
  {
    const double X = NPME_Pot_Xpad;
    double x8[8]  __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};
    double y8[8]  __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};
    double z8[8]  __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};

    for (size_t i = 0; i < remain; i++)
    {
      size_t index = 8*nLoop+i;
      x8[i] = coord[3*index  ];
      y8[i] = coord[3*index+1];
      z8[i] = coord[3*index+2];
    }
    __m512d xVec, yVec, zVec;
    xVec  = _mm512_load_pd (x8);
    yVec  = _mm512_load_pd (y8);
    zVec  = _mm512_load_pd (z8);

    _mm512_store_pd (&coord_8x[index24i   ], xVec);
    _mm512_store_pd (&coord_8x[index24i+ 8], yVec);
    _mm512_store_pd (&coord_8x[index24i+16], zVec);
  }
}

inline void NPME_TransformRealV1_8x_2_V1_AVX_512 (const size_t nCharge, 
  const double *V1_8x, double *V1)
//input:  V1_8x[4*M] = = aligned at 64 bit boundary
//        V1_8x[] = {V0[0], V0[1], V0[2], V0[3], V0[4], V0[5], V0[6], V0[7],
//                   VX[0], VX[1], VX[2], VX[3], VX[4], VX[5], VX[6], VX[7],
//                   VY[0], VY[1], VY[2], VY[3], VY[4], VY[5], VY[6], VY[7],
//                   VZ[0], VZ[1], VZ[2], VZ[3], VZ[4], VZ[5], VZ[6], VZ[7],
//                   V0[8], V0[9], V0[10], ..
//output: V1[4*nCharge] = not aligned
//                      = {V0[0], VX[0], VY[0], VZ[0],
//                         V0[1], VX[1], VY[1], VZ[1], ..
{
  const size_t remain = nCharge%8;
  const size_t nLoop  = (nCharge-remain)/8;

  size_t index32i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    const __m512d V0  = _mm512_load_pd (&V1_8x[index32i   ]);
    const __m512d VX  = _mm512_load_pd (&V1_8x[index32i+ 8]);
    const __m512d VY  = _mm512_load_pd (&V1_8x[index32i+16]);
    const __m512d VZ  = _mm512_load_pd (&V1_8x[index32i+24]);

    __m512d t0, t1, t2, t3;
    NPME_mm512_4x8transpose_pd (V0, VX, VY, VZ, t0, t1, t2, t3);

    _mm512_storeu_pd (&V1[index32i   ], t0);
    _mm512_storeu_pd (&V1[index32i+ 8], t1);
    _mm512_storeu_pd (&V1[index32i+16], t2);
    _mm512_storeu_pd (&V1[index32i+24], t3);

    index32i += 32;
  }

  if (remain > 0)
  {
    double *V1_loc        = &V1[32*nLoop];
    const double *V0_loc  = &V1_8x[index32i   ];
    const double *VX_loc  = &V1_8x[index32i+ 8];
    const double *VY_loc  = &V1_8x[index32i+16];
    const double *VZ_loc  = &V1_8x[index32i+24];

    size_t index4i = 0;
    for (size_t i = 0; i < remain; i++)
    {
      V1_loc[index4i  ] = V0_loc[i];
      V1_loc[index4i+1] = VX_loc[i];
      V1_loc[index4i+2] = VY_loc[i];
      V1_loc[index4i+3] = VZ_loc[i];
      index4i += 4;
    }
  }
}



inline void NPME_TransformUpdateRealV1_8x_2_V1_AVX_512 (const size_t nCharge, 
  const double *V1_8x, double *V1)
//input:  V1_8x[4*M] = = aligned at 64 bit boundary
//        V1_8x[] = {V0[0], V0[1], V0[2], V0[3], V0[4], V0[5], V0[6], V0[7],
//                   VX[0], VX[1], VX[2], VX[3], VX[4], VX[5], VX[6], VX[7],
//                   VY[0], VY[1], VY[2], VY[3], VY[4], VY[5], VY[6], VY[7],
//                   VZ[0], VZ[1], VZ[2], VZ[3], VZ[4], VZ[5], VZ[6], VZ[7],
//                   V0[8], V0[9], V0[10], ..
//output: V1[4*nCharge] = not aligned
//                      = {V0[0], VX[0], VY[0], VZ[0],
//                         V0[1], VX[1], VY[1], VZ[1], ..
{
  const size_t remain = nCharge%8;
  const size_t nLoop  = (nCharge-remain)/8;

  size_t index32i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    const __m512d V0  = _mm512_load_pd (&V1_8x[index32i   ]);
    const __m512d VX  = _mm512_load_pd (&V1_8x[index32i+ 8]);
    const __m512d VY  = _mm512_load_pd (&V1_8x[index32i+16]);
    const __m512d VZ  = _mm512_load_pd (&V1_8x[index32i+24]);

    __m512d t0, t1, t2, t3;
    NPME_mm512_4x8transpose_pd (V0, VX, VY, VZ, t0, t1, t2, t3);

    __m512d s0  = _mm512_loadu_pd (&V1[index32i   ]);
    __m512d s1  = _mm512_loadu_pd (&V1[index32i+ 8]);
    __m512d s2  = _mm512_loadu_pd (&V1[index32i+16]);
    __m512d s3  = _mm512_loadu_pd (&V1[index32i+24]);

    t0 = _mm512_add_pd (s0, t0);
    t1 = _mm512_add_pd (s1, t1);
    t2 = _mm512_add_pd (s2, t2);
    t3 = _mm512_add_pd (s3, t3);

    _mm512_storeu_pd (&V1[index32i   ], t0);
    _mm512_storeu_pd (&V1[index32i+ 8], t1);
    _mm512_storeu_pd (&V1[index32i+16], t2);
    _mm512_storeu_pd (&V1[index32i+24], t3);

    index32i += 32;
  }

  if (remain > 0)
  {
    double *V1_loc        = &V1[32*nLoop];
    const double *V0_loc  = &V1_8x[index32i   ];
    const double *VX_loc  = &V1_8x[index32i+ 8];
    const double *VY_loc  = &V1_8x[index32i+16];
    const double *VZ_loc  = &V1_8x[index32i+24];

    size_t index4i = 0;
    for (size_t i = 0; i < remain; i++)
    {
      V1_loc[index4i  ] += V0_loc[i];
      V1_loc[index4i+1] += VX_loc[i];
      V1_loc[index4i+2] += VY_loc[i];
      V1_loc[index4i+3] += VZ_loc[i];
      index4i += 4;
    }
  }
}

inline void NPME_TransformComplexV1_8x_2_V1_AVX_512 (const size_t nCharge, 
  const double *V1_8x, _Complex double *V1)
//input:  V1_8x[4*M] = aligned at 64 bit boundary
//    V1_8x[] = {V0r[0], V0r[1], V0r[2], V0r[3], V0r[4], V0r[5], V0r[6], V0r[7],
//               V0i[0], V0i[1], V0i[2], V0i[3], V0i[4], V0i[5], V0i[6], V0i[7],
//               VXr[0], VXr[1], VXr[2], VXr[3], VXr[4], VXr[5], VXr[6], VXr[7],
//               VXi[0], VXi[1], VXi[2], VXi[3], VXi[4], VXi[5], VXi[6], VXi[7],
//               VYr[0], VYr[1], VYr[2], VYr[3], VYr[4], VYr[5], VYr[6], VYr[7],
//               VYi[0], VYi[1], VYi[2], VYi[3], VYi[4], VYi[5], VYi[6], VYi[7],
//               VZr[0], VZr[1], VZr[2], VZr[3], VZr[4], VZr[5], VZr[6], VZr[7],
//               VZi[0], VZi[1], VZi[2], VZi[3], VZi[4], VZi[5], VZi[6], VZi[7],
//               V0r[8], V0r[9], V0r[10],...
//output: V1[4*nCharge]= not aligned
//  = {V0r[0], V0i[0], VXr[0], VXi[0], VYr[0], VYi[0], VZr[0], VZi[0],
//     V0r[1], V0i[1], VXr[1], VXi[1], VYr[1], VYi[1], VZr[1], VZi[1],
//     ...
{
  const size_t remain = nCharge%8;
  const size_t nLoop  = (nCharge-remain)/8;

  double *V1r = (double *) V1;

  size_t index64i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d a0, a1, a2, a3, a4, a5, a6, a7;
    __m512d b0, b1, b2, b3, b4, b5, b6, b7;

    //V0r, V0i, VXr, VXi
    a0  = _mm512_load_pd (&V1_8x[index64i   ]);
    a1  = _mm512_load_pd (&V1_8x[index64i+ 8]);
    a2  = _mm512_load_pd (&V1_8x[index64i+16]);
    a3  = _mm512_load_pd (&V1_8x[index64i+24]);
    a4  = _mm512_load_pd (&V1_8x[index64i+32]);
    a5  = _mm512_load_pd (&V1_8x[index64i+40]);
    a6  = _mm512_load_pd (&V1_8x[index64i+48]);
    a7  = _mm512_load_pd (&V1_8x[index64i+56]);

    NPME_mm512_8x8transpose_pd (a0, a1, a2, a3, a4, a5, a6, a7,
                                         b0, b1, b2, b3, b4, b5, b6, b7);



    _mm512_storeu_pd (&V1r[index64i   ], b0);
    _mm512_storeu_pd (&V1r[index64i+ 8], b1);
    _mm512_storeu_pd (&V1r[index64i+16], b2);
    _mm512_storeu_pd (&V1r[index64i+24], b3);
    _mm512_storeu_pd (&V1r[index64i+32], b4);
    _mm512_storeu_pd (&V1r[index64i+40], b5);
    _mm512_storeu_pd (&V1r[index64i+48], b6);
    _mm512_storeu_pd (&V1r[index64i+56], b7);

    index64i += 64;
  }

  if (remain > 0)
  {
    _Complex double *V1_loc = &V1[32*nLoop];
    const double *V0r_loc   = &V1_8x[index64i   ];
    const double *V0i_loc   = &V1_8x[index64i+ 8];
    const double *VXr_loc   = &V1_8x[index64i+16];
    const double *VXi_loc   = &V1_8x[index64i+24];
    const double *VYr_loc   = &V1_8x[index64i+32];
    const double *VYi_loc   = &V1_8x[index64i+40];
    const double *VZr_loc   = &V1_8x[index64i+48];
    const double *VZi_loc   = &V1_8x[index64i+56];

    size_t index4i = 0;
    for (size_t i = 0; i < remain; i++)
    {
      V1_loc[index4i  ] = V0r_loc[i] + I*V0i_loc[i];
      V1_loc[index4i+1] = VXr_loc[i] + I*VXi_loc[i];
      V1_loc[index4i+2] = VYr_loc[i] + I*VYi_loc[i];
      V1_loc[index4i+3] = VZr_loc[i] + I*VZi_loc[i];
      index4i += 4;
    }
  }
}

inline void NPME_TransformUpdateComplexV1_8x_2_V1_AVX_512 (const size_t nCharge, 
  const double *V1_8x, _Complex double *V1)
//input:  V1_8x[4*M] = aligned at 64 bit boundary
//    V1_8x[] = {V0r[0], V0r[1], V0r[2], V0r[3], V0r[4], V0r[5], V0r[6], V0r[7],
//               V0i[0], V0i[1], V0i[2], V0i[3], V0i[4], V0i[5], V0i[6], V0i[7],
//               VXr[0], VXr[1], VXr[2], VXr[3], VXr[4], VXr[5], VXr[6], VXr[7],
//               VXi[0], VXi[1], VXi[2], VXi[3], VXi[4], VXi[5], VXi[6], VXi[7],
//               VYr[0], VYr[1], VYr[2], VYr[3], VYr[4], VYr[5], VYr[6], VYr[7],
//               VYi[0], VYi[1], VYi[2], VYi[3], VYi[4], VYi[5], VYi[6], VYi[7],
//               VZr[0], VZr[1], VZr[2], VZr[3], VZr[4], VZr[5], VZr[6], VZr[7],
//               VZi[0], VZi[1], VZi[2], VZi[3], VZi[4], VZi[5], VZi[6], VZi[7],
//               V0r[8], V0r[9], V0r[10],...
//output: V1[4*nCharge]= not aligned
//  = {V0r[0], V0i[0], VXr[0], VXi[0], VYr[0], VYi[0], VZr[0], VZi[0],
//     V0r[1], V0i[1], VXr[1], VXi[1], VYr[1], VYi[1], VZr[1], VZi[1],
//     ...
{
  const size_t remain = nCharge%8;
  const size_t nLoop  = (nCharge-remain)/8;

  double *V1r = (double *) V1;

  size_t index64i = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d a0, a1, a2, a3, a4, a5, a6, a7;
    __m512d b0, b1, b2, b3, b4, b5, b6, b7;

    //V0r, V0i, VXr, VXi
    a0  = _mm512_load_pd (&V1_8x[index64i   ]);
    a1  = _mm512_load_pd (&V1_8x[index64i+ 8]);
    a2  = _mm512_load_pd (&V1_8x[index64i+16]);
    a3  = _mm512_load_pd (&V1_8x[index64i+24]);
    a4  = _mm512_load_pd (&V1_8x[index64i+32]);
    a5  = _mm512_load_pd (&V1_8x[index64i+40]);
    a6  = _mm512_load_pd (&V1_8x[index64i+48]);
    a7  = _mm512_load_pd (&V1_8x[index64i+56]);

    NPME_mm512_8x8transpose_pd (a0, a1, a2, a3, a4, a5, a6, a7,
                                         b0, b1, b2, b3, b4, b5, b6, b7);

    __m512d t0;
    t0 = _mm512_loadu_pd (&V1r[index64i  ]);
    b0 = _mm512_add_pd (t0, b0);
    _mm512_storeu_pd (&V1r[index64i   ], b0);

    t0 = _mm512_loadu_pd (&V1r[index64i+ 8]);
    b1 = _mm512_add_pd (t0, b1);
    _mm512_storeu_pd (&V1r[index64i+ 8], b1);

    t0 = _mm512_loadu_pd (&V1r[index64i+16]);
    b2 = _mm512_add_pd (t0, b2);
    _mm512_storeu_pd (&V1r[index64i+16], b2);

    t0 = _mm512_loadu_pd (&V1r[index64i+24]);
    b3 = _mm512_add_pd (t0, b3);
    _mm512_storeu_pd (&V1r[index64i+24], b3);

    t0 = _mm512_loadu_pd (&V1r[index64i+32]);
    b4 = _mm512_add_pd (t0, b4);
    _mm512_storeu_pd (&V1r[index64i+32], b4);

    t0 = _mm512_loadu_pd (&V1r[index64i+40]);
    b5 = _mm512_add_pd (t0, b5);
    _mm512_storeu_pd (&V1r[index64i+40], b5);

    t0 = _mm512_loadu_pd (&V1r[index64i+48]);
    b6 = _mm512_add_pd (t0, b6);
    _mm512_storeu_pd (&V1r[index64i+48], b6);

    t0 = _mm512_loadu_pd (&V1r[index64i+56]);
    b7 = _mm512_add_pd (t0, b7);
    _mm512_storeu_pd (&V1r[index64i+56], b7);

    index64i += 64;
  }

  if (remain > 0)
  {
    _Complex double *V1_loc = &V1[32*nLoop];
    const double *V0r_loc   = &V1_8x[index64i   ];
    const double *V0i_loc   = &V1_8x[index64i+ 8];
    const double *VXr_loc   = &V1_8x[index64i+16];
    const double *VXi_loc   = &V1_8x[index64i+24];
    const double *VYr_loc   = &V1_8x[index64i+32];
    const double *VYi_loc   = &V1_8x[index64i+40];
    const double *VZr_loc   = &V1_8x[index64i+48];
    const double *VZi_loc   = &V1_8x[index64i+56];

    size_t index4i = 0;
    for (size_t i = 0; i < remain; i++)
    {
      V1_loc[index4i  ] += V0r_loc[i] + I*V0i_loc[i];
      V1_loc[index4i+1] += VXr_loc[i] + I*VXi_loc[i];
      V1_loc[index4i+2] += VYr_loc[i] + I*VYi_loc[i];
      V1_loc[index4i+3] += VZr_loc[i] + I*VZi_loc[i];
      index4i += 4;
    }
  }
}

inline void NPME_TransposeCoord_AVX_512 (const size_t nCharge, 
  const double *coord, double *xAlignTmp, double *yAlignTmp, double *zAlignTmp)
//input:  coord[3*nCharge] not aligned
//        nCharge must be a multiple of 8
//output: xAlignTmp[nCharge] = aligned at 64 bit boundary
//        yAlignTmp[nCharge] = aligned at 64 bit boundary
//        zAlignTmp[nCharge] = aligned at 64 bit boundary
{
  if (nCharge%8 != 0)
  {
    std::cout << "Error in NPME_TransposeCoord_AVX_512.\n";
    std::cout << "nCharge = " << nCharge << " is not a multiple of 8\n";
    exit(0);
  }

  const size_t m = nCharge/8;

  size_t indexCrd   = 0;
  size_t indexAlign = 0;
  for (size_t i = 0; i < m; i++)
  {
    const double *X  = &coord[indexCrd];
    indexCrd += 24;

    const __m512d t0 = _mm512_loadu_pd (&X[0]);
    const __m512d t1 = _mm512_loadu_pd (&X[8]);
    const __m512d t2 = _mm512_loadu_pd (&X[16]);

    __m512d x, y, z;
    NPME_mm512_8x3transpose_pd (t0, t1, t2, x, y, z);
    _mm512_store_pd (&xAlignTmp[indexAlign], x);
    _mm512_store_pd (&yAlignTmp[indexAlign], y);
    _mm512_store_pd (&zAlignTmp[indexAlign], z);
    indexAlign += 8;
  }
}


inline void NPME_TransposeV1_AVX_512 (const size_t nCharge, double *V1, 
  const double *V0_align, const double *VX_align, 
  const double *VY_align, const double *VZ_align)
//input:  nCharge must be a multiple of 8
//        V0_align[nCharge] = aligned at 64 bit boundary
//        VX_align[nCharge] = aligned at 64 bit boundary
//        VY_align[nCharge] = aligned at 64 bit boundary
//        VZ_align[nCharge] = aligned at 64 bit boundary
//output: V1[nCharge][4] = need not be aligned
{
  if (nCharge%8 != 0)
  {
    std::cout << "Error in NPME_TransposeV1_AVX_512.\n";
    std::cout << "nCharge = " << nCharge << " is not a multiple of 8\n";
    exit(0);
  }

  const size_t m = nCharge/8;

  size_t index8x  = 0;
  size_t index32x = 0;
  for (size_t i = 0; i < m; i++)
  {
    const __m512d V0_vec = _mm512_load_pd (&V0_align[index8x]);
    const __m512d VX_vec = _mm512_load_pd (&VX_align[index8x]);
    const __m512d VY_vec = _mm512_load_pd (&VY_align[index8x]);
    const __m512d VZ_vec = _mm512_load_pd (&VZ_align[index8x]);
    index8x += 8;

    __m512d t0, t1, t2, t3;
    NPME_mm512_4x8transpose_pd (V0_vec, VX_vec, VY_vec, VZ_vec,
      t0, t1, t2, t3);
    _mm512_storeu_pd (&V1[index32x   ], t0);
    _mm512_storeu_pd (&V1[index32x+ 8], t1);
    _mm512_storeu_pd (&V1[index32x+16], t2);
    _mm512_storeu_pd (&V1[index32x+24], t3);
    index32x += 32;
  }
}

inline void NPME_TransposeAddUpdateV1_AVX_512 (const size_t nCharge, double *V1, 
  const double *V0_align, const double *VX_align, 
  const double *VY_align, const double *VZ_align)
//similar to NPME_TransposeV1_AVX_512, but updates V1 by adding 
//instead of overwriting V1
//input:  nCharge must be a multiple of 8
//        V0_align[nCharge] = aligned at 64 bit boundary
//        VX_align[nCharge] = aligned at 64 bit boundary
//        VY_align[nCharge] = aligned at 64 bit boundary
//        VZ_align[nCharge] = aligned at 64 bit boundary
//output: V1[nCharge][4] = need not be aligned
{
  if (nCharge%8 != 0)
  {
    std::cout << "Error in NPME_TransposeAddUpdateV1_AVX_512.\n";
    std::cout << "nCharge = " << nCharge << " is not a multiple of 8\n";
    exit(0);
  }

  const size_t m = nCharge/8;

  size_t index8x  = 0;
  size_t index32x = 0;
  for (size_t i = 0; i < m; i++)
  {
    __m512d A_vec = _mm512_load_pd (&V0_align[index8x]);
    __m512d B_vec = _mm512_load_pd (&VX_align[index8x]);
    __m512d C_vec = _mm512_load_pd (&VY_align[index8x]);
    __m512d D_vec = _mm512_load_pd (&VZ_align[index8x]);
    index8x += 8;

    __m512d t0_vec, t1_vec, t2_vec, t3_vec;
    NPME_mm512_4x8transpose_pd (A_vec, B_vec, C_vec, D_vec,
      t0_vec, t1_vec, t2_vec, t3_vec);

    A_vec = _mm512_loadu_pd (&V1[index32x   ]);
    B_vec = _mm512_loadu_pd (&V1[index32x+8 ]);
    C_vec = _mm512_loadu_pd (&V1[index32x+16]);
    D_vec = _mm512_loadu_pd (&V1[index32x+24]);

    t0_vec = _mm512_add_pd (t0_vec, A_vec);
    t1_vec = _mm512_add_pd (t1_vec, B_vec);
    t2_vec = _mm512_add_pd (t2_vec, C_vec);
    t3_vec = _mm512_add_pd (t3_vec, D_vec);

    _mm512_storeu_pd (&V1[index32x   ], t0_vec);
    _mm512_storeu_pd (&V1[index32x+ 8], t1_vec);
    _mm512_storeu_pd (&V1[index32x+16], t2_vec);
    _mm512_storeu_pd (&V1[index32x+24], t3_vec);
    index32x += 32;
  }
}


inline void NPME_TransposeV1_AVX_512 (
  const size_t nCharge, _Complex double *V1, 
  const double *V0_r_align, const double *V0_i_align,
  const double *VX_r_align, const double *VX_i_align,
  const double *VY_r_align, const double *VY_i_align,
  const double *VZ_r_align, const double *VZ_i_align)
//input:  nCharge must be a multiple of 8
//        V0_align[nCharge] = aligned at 64 bit boundary
//        VX_align[nCharge] = aligned at 64 bit boundary
//        VY_align[nCharge] = aligned at 64 bit boundary
//        VZ_align[nCharge] = aligned at 64 bit boundary
//output: V1[nCharge][4] = need not be aligned
{
  if (nCharge%8 != 0)
  {
    std::cout << "Error in NPME_TransposeV1_AVX_512.\n";
    std::cout << "nCharge = " << nCharge << " is not a multiple of 8\n";
    exit(0);
  }

  const size_t m = nCharge/8;

  double *V1r = (double *) V1;

  size_t index8x  = 0;
  size_t index64x = 0;
  for (size_t i = 0; i < m; i++)
  {
    const __m512d V0_r_vec = _mm512_load_pd (&V0_r_align[index8x]);
    const __m512d VX_r_vec = _mm512_load_pd (&VX_r_align[index8x]);
    const __m512d VY_r_vec = _mm512_load_pd (&VY_r_align[index8x]);
    const __m512d VZ_r_vec = _mm512_load_pd (&VZ_r_align[index8x]);

    const __m512d V0_i_vec = _mm512_load_pd (&V0_i_align[index8x]);
    const __m512d VX_i_vec = _mm512_load_pd (&VX_i_align[index8x]);
    const __m512d VY_i_vec = _mm512_load_pd (&VY_i_align[index8x]);
    const __m512d VZ_i_vec = _mm512_load_pd (&VZ_i_align[index8x]);

    index8x += 8;

    __m512d t0, t1, t2, t3, t4, t5, t6, t7;
    NPME_mm512_8x8transpose_pd (
      V0_r_vec, V0_i_vec, VX_r_vec, VX_i_vec,
      VY_r_vec, VY_i_vec, VZ_r_vec, VZ_i_vec,
      t0, t1, t2, t3, t4, t5, t6, t7);


    _mm512_storeu_pd (&V1r[index64x   ], t0);
    _mm512_storeu_pd (&V1r[index64x+ 8], t1);

    _mm512_storeu_pd (&V1r[index64x+16], t2);
    _mm512_storeu_pd (&V1r[index64x+24], t3);

    _mm512_storeu_pd (&V1r[index64x+32], t4);
    _mm512_storeu_pd (&V1r[index64x+40], t5);

    _mm512_storeu_pd (&V1r[index64x+48], t6);
    _mm512_storeu_pd (&V1r[index64x+56], t7);


    index64x += 64;
  }
}

inline void NPME_TransposeAddUpdateV1_AVX_512 (
  const size_t nCharge, _Complex double *V1, 
  const double *V0_r_align, const double *V0_i_align,
  const double *VX_r_align, const double *VX_i_align,
  const double *VY_r_align, const double *VY_i_align,
  const double *VZ_r_align, const double *VZ_i_align)
//similar to NPME_TransposeV1_AVX, but updates V1 by adding 
//instead of overwriting V1
//input:  nCharge must be a multiple of 8
//        V0_align[nCharge] = aligned at 64 bit boundary
//        VX_align[nCharge] = aligned at 64 bit boundary
//        VY_align[nCharge] = aligned at 64 bit boundary
//        VZ_align[nCharge] = aligned at 64 bit boundary
//output: V1[nCharge][4] = need not be aligned
{
  if (nCharge%8 != 0)
  {
    std::cout << "Error in NPME_TransposeAddUpdateV1_AVX_512.\n";
    std::cout << "nCharge = " << nCharge << " is not a multiple of 8\n";
    exit(0);
  }

  const size_t m = nCharge/8;

  double *V1r = (double *) V1;

  size_t index8x  = 0;
  size_t index64x = 0;
  for (size_t i = 0; i < m; i++)
  {
    __m512d t0, t1, t2, t3, t4, t5, t6, t7;
    {
      const __m512d V0_r_vec = _mm512_load_pd (&V0_r_align[index8x]);
      const __m512d VX_r_vec = _mm512_load_pd (&VX_r_align[index8x]);
      const __m512d VY_r_vec = _mm512_load_pd (&VY_r_align[index8x]);
      const __m512d VZ_r_vec = _mm512_load_pd (&VZ_r_align[index8x]);

      const __m512d V0_i_vec = _mm512_load_pd (&V0_i_align[index8x]);
      const __m512d VX_i_vec = _mm512_load_pd (&VX_i_align[index8x]);
      const __m512d VY_i_vec = _mm512_load_pd (&VY_i_align[index8x]);
      const __m512d VZ_i_vec = _mm512_load_pd (&VZ_i_align[index8x]);

      index8x += 8;

      NPME_mm512_8x8transpose_pd (
        V0_r_vec, V0_i_vec, VX_r_vec, VX_i_vec,
        VY_r_vec, VY_i_vec, VZ_r_vec, VZ_i_vec,
        t0, t1, t2, t3, t4, t5, t6, t7);
    }

    __m512d Avec;

    Avec = _mm512_load_pd (&V1r[index64x   ]); t0 = _mm512_add_pd (t0, Avec);
    Avec = _mm512_load_pd (&V1r[index64x+ 8]); t1 = _mm512_add_pd (t1, Avec);

    Avec = _mm512_load_pd (&V1r[index64x+16]); t2 = _mm512_add_pd (t2, Avec);
    Avec = _mm512_load_pd (&V1r[index64x+24]); t3 = _mm512_add_pd (t3, Avec);

    Avec = _mm512_load_pd (&V1r[index64x+32]); t4 = _mm512_add_pd (t4, Avec);
    Avec = _mm512_load_pd (&V1r[index64x+40]); t5 = _mm512_add_pd (t5, Avec);

    Avec = _mm512_load_pd (&V1r[index64x+48]); t6 = _mm512_add_pd (t6, Avec);
    Avec = _mm512_load_pd (&V1r[index64x+56]); t7 = _mm512_add_pd (t7, Avec);

    _mm512_storeu_pd (&V1r[index64x   ], t0);
    _mm512_storeu_pd (&V1r[index64x+ 8], t1);

    _mm512_storeu_pd (&V1r[index64x+16], t2);
    _mm512_storeu_pd (&V1r[index64x+24], t3);

    _mm512_storeu_pd (&V1r[index64x+32], t4);
    _mm512_storeu_pd (&V1r[index64x+40], t5);

    _mm512_storeu_pd (&V1r[index64x+48], t6);
    _mm512_storeu_pd (&V1r[index64x+56], t7);


    index64x += 64;
  }
}

inline void NPME_Real2Complex_AVX_512 (const size_t N, _Complex double *z, 
  const double *xAlignTmp, const double *yAlignTmp)
//input:  xAlignTmp[N] = aligned at 32 bit boundary
//        yAlignTmp[N] = aligned at 32 bit boundary
//        N must be a multiple of 4
//output: z[N] need not be aligned
//        where z[N] = x[N] + I*y[N]
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Real2Complex_AVX_512.\n";
    std::cout << "N = " << N << " is not a multiple of 8\n";
    exit(0);
  }

  const size_t m  = N/8;
  double *zr      = (double *) z;
  //zr[2*N] = zr[8*m]

  size_t indexCrd   = 0;
  size_t indexAlign = 0;
  for (size_t i = 0; i < m; i++)
  {
    const __m512d x = _mm512_load_pd (&xAlignTmp[indexAlign]);
    const __m512d y = _mm512_load_pd (&yAlignTmp[indexAlign]);
    indexAlign += 8;

    __m512d t0, t1;
    NPME_mm512_2x8transpose_pd (x, y, t0, t1);

    double *X  = &zr[indexCrd];
    _mm512_storeu_pd (&X[0], t0);
    _mm512_storeu_pd (&X[8], t1);
    indexCrd += 16;
  }
}


inline void NPME_Complex2Real_AVX_512 (const size_t N, const _Complex double *z, 
  double *xAlignTmp, double *yAlignTmp)
//input:  z[N] need not be aligned
//        N must be a multiple of 4
//output: xAlignTmp[N] = aligned at 32 bit boundary
//        yAlignTmp[N] = aligned at 32 bit boundary
//        where z[N] = x[N] + I*y[N]
{
  if (N%8 != 0)
  {
    std::cout << "Error in NPME_Complex2Real_AVX_512.\n";
    std::cout << "N = " << N << " is not a multiple of 8\n";
    exit(0);
  }

  const size_t m    = N/8;
  const double *zr  = (const double *) z;
  //zr[2*N] = zr[8*m]

  size_t indexCrd   = 0;
  size_t indexAlign = 0;
  for (size_t i = 0; i < m; i++)
  {
    const double *X  = &zr[indexCrd];
    const __m512d t0 = _mm512_loadu_pd (&X[0]);
    const __m512d t1 = _mm512_loadu_pd (&X[8]);
    indexCrd += 16;

    __m512d x, y;
    NPME_mm512_8x2transpose_pd (t0, t1, x, y);
    _mm512_store_pd (&xAlignTmp[indexAlign], x);
    _mm512_store_pd (&yAlignTmp[indexAlign], y);
    indexAlign += 8;
  }
}

#endif //NPME_USE_AVX_512

}//end namespace NPME_Library





#endif // NPME_POTENTIAL_SUPPORT_FUNCTIONS_H


