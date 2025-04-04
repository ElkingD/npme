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
#include <vector>

#include <immintrin.h>



#include "NPME_Constant.h"
#include "NPME_PotentialHelmholtz.h"
#include "NPME_VectorIntrinsic.h"
#include "NPME_PotentialSupportFunctions.h"
#include "NPME_SupportFunctions.h"
#include "NPME_FunctionDerivMatch.h"
#include "NPME_PartitionBox.h"
#include "NPME_PartitionEmbeddedBox.h"
#include "NPME_AlignedArray.h"





namespace NPME_Library
{
void NPME_PotHelmholtz_MacroSelf_V1 (const _Complex double k0,
  const size_t nCharge, const double *coord, const _Complex double *Q1, 
  _Complex double *V1, const int nProc, const int vecOption, 
  const size_t blockSize)
{
  if (blockSize%8 != 0)
  {
    printf("Error in NPME_PotHelmholtz_MacroSelf_V1.\n");
    printf("blockSize = %lu is not a multiple of 8\n", blockSize);
    exit(0);
  }

  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotHelmholtz_MacroSelf_V1.\n");
    printf("blockSize = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      blockSize, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  const size_t remain = nCharge%blockSize;
  size_t nBlock       = (nCharge-remain)/blockSize;
  if (remain > 0)
    nBlock++;

  memset(V1, 0, 4*nCharge*sizeof(_Complex double));


  const size_t nPair = (nBlock*(nBlock+1))/2;

  size_t k;
  #pragma omp parallel shared(V1, Q1, coord, nBlock) private(k) default(none) num_threads(nProc)
  {
    _Complex double V1loc_1 [4*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
    _Complex double V1loc_2 [4*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));

    #pragma omp for schedule(static)
    for (k = 0; k < nPair; k++)
    {
      size_t i, j;
      NPME_ind2D_symmetric2_index_2_pq (i, j, k);

      if (i != j)
      {
        size_t nCharge1 = blockSize;
        size_t nCharge2 = blockSize;

        if ( (remain > 0) && (i == nBlock - 1) )  nCharge1 = remain;
        if ( (remain > 0) && (j == nBlock - 1) )  nCharge2 = remain;

        const size_t index_i        = i*blockSize;
        const _Complex double *Q1_1 = &Q1[index_i];
        const double *coord1        = &coord[3*index_i];
        _Complex double *V1_1       = &V1[4*index_i];

        const size_t index_j        = j*blockSize;
        const _Complex double *Q1_2 = &Q1[index_j];
        const double *coord2        = &coord[3*index_j];
       _Complex  double *V1_2       = &V1[4*index_j];

        NPME_PotHelmholtz_Pair_V1 (k0, 
          nCharge1, coord1, Q1_1, 
          nCharge2, coord2, Q1_2, 
          V1loc_1, V1loc_2, vecOption);

        #pragma omp critical (update_NPME_PotHelmholtz_MacroSelf_V1)
        {
          for (size_t n = 0; n < 4*nCharge1; n++)   V1_1[n] += V1loc_1[n];
          for (size_t n = 0; n < 4*nCharge2; n++)   V1_2[n] += V1loc_2[n];
        }
      }
      else
      {
        size_t nCharge1 = blockSize;

        if ( (remain > 0) && (i == nBlock - 1) )  nCharge1 = remain;

        const size_t index_i        = i*blockSize;
        const _Complex double *Q1_1 = &Q1[index_i];
        const double *coord1        = &coord[3*index_i];
        _Complex double *V1_1       = &V1[4*index_i];

        NPME_PotHelmholtz_Self_V1 (k0, nCharge1, 
          coord1, Q1_1, V1loc_1, vecOption);

        #pragma omp critical (update_NPME_PotHelmholtz_MacroSelf_V1)
        {
          for (size_t n = 0; n < 4*nCharge1; n++)   V1_1[n] += V1loc_1[n];
        }
      }
    }
  }
}


void NPME_PotHelmholtz_SR_DM_ClusterElement_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir,
  const NPME_Library::NPME_ClusterPair& cluster,
  const double *coord, const _Complex double *charge,
  _Complex double *V1, _Complex double *V2, 
  int vecOption, size_t blockSize, bool zeroArray)
//input:  coord[3*nCharge], charge[nCharge], cluster (single element)
//output: V1[cluster.nPointPerCluster1][4]
//        V2[cluster.nPointPerCluster2][4]
{
  const size_t nChargeA1  = cluster.nPointPerCluster1;
  const size_t nChargeA2  = cluster.nPointPerCluster2;
  const size_t startIndA1 = cluster.pointStartA1;
  const size_t startIndA2 = cluster.pointStartA2;

  if (zeroArray)
  {
    memset(V1, 0, 4*nChargeA1*sizeof(_Complex double));
    memset(V2, 0, 4*nChargeA2*sizeof(_Complex double));
  }

  for (size_t k = 0; k < cluster.pairB.size(); k++)
  {
    const size_t nChargeB1    = cluster.pairB[k].nPointPerCell1;
    const size_t nChargeB2    = cluster.pairB[k].nPointPerCell2;
    const size_t startIndB1   = cluster.pairB[k].startPointIndex1;
    const size_t startIndB2   = cluster.pairB[k].startPointIndex2;
    const size_t locStartInd1 = startIndB1 - startIndA1;
    const size_t locStartInd2 = startIndB2 - startIndA2;

    const double *coordB1           = &coord[3*startIndB1];
    const double *coordB2           = &coord[3*startIndB2];
    const _Complex double *chargeB1 = &charge[startIndB1];
    const _Complex double *chargeB2 = &charge[startIndB2];
    _Complex double *VB1            = &V1[4*locStartInd1];
    _Complex double *VB2            = &V2[4*locStartInd2];

    bool zeroArrayB = 0;
    if (cluster.pairB[k].cellIndex1 != cluster.pairB[k].cellIndex2)
    {
      NPME_PotHelmholtz_SR_DM_LargePair_V1 (k0,
        Nder, a, b, Rdir,
        nChargeB1, coordB1, chargeB1, 
        nChargeB2, coordB2, chargeB2, 
        VB1, VB2, vecOption, blockSize, zeroArrayB);
    }
    else
    {
      NPME_PotHelmholtz_SR_DM_LargeSelf_V1 (k0,
        Nder, a, b, Rdir,
        nChargeB1, coordB1, chargeB1, VB1, vecOption, blockSize, zeroArrayB);
    }
  }
}



void NPME_PotHelmholtz_SR_DM_DirectSum_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir,
  const size_t nCharge, const double *coord, const _Complex double *charge, 
  _Complex double *V, const int nProc, const int vecOption,
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster, 
  const size_t blockSize)
{
  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotHelmholtz_SR_DM_DirectSum_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotHelmholtz_SR_DM_DirectSum_V1\n";
    sprintf(str, "blockSize = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n",
      blockSize, NPME_Pot_MaxChgBlock_V1);
    std::cout << str;
    exit(0);
  }

  memset(V, 0, 4*nCharge*sizeof(_Complex double));

  const size_t nCluster = cluster.size();

  size_t k;
  #pragma omp parallel shared(V, charge, coord, cluster, a, b) private(k) default(none) num_threads(nProc)
  {
    #pragma omp for schedule(dynamic)
    for (k = 0; k < nCluster; k++)
    {
      bool zeroArray = 1;
      const size_t nChargeA1  = cluster[k].nPointPerCluster1;
      const size_t nChargeA2  = cluster[k].nPointPerCluster2;
      const size_t startIndA1 = cluster[k].pointStartA1;
      const size_t startIndA2 = cluster[k].pointStartA2;

      NPME_AlignedArrayDoubleComplex Vmem_1 (4*nChargeA1, 64);
      NPME_AlignedArrayDoubleComplex Vmem_2 (4*nChargeA2, 64);

      _Complex double *VA1             = &V[4*startIndA1];
      _Complex double *VA2             = &V[4*startIndA2];
      _Complex double *V1loc_1         = Vmem_1.GetPtr();
      _Complex double *V1loc_2         = Vmem_2.GetPtr();


      NPME_PotHelmholtz_SR_DM_ClusterElement_V1 (k0, Nder, a, b, Rdir,
            cluster[k], coord, charge, V1loc_1, V1loc_2, 
            vecOption, blockSize, zeroArray);

      #pragma omp critical (update_NPME_PotHelmholtz_SR_DM_DirectSum_V1)
      {
        for (size_t n = 0; n < 4*nChargeA1; n++)   VA1[n] += V1loc_1[n];
        for (size_t n = 0; n < 4*nChargeA2; n++)   VA2[n] += V1loc_2[n];
      }
    }
  }
}

void NPME_PotHelmholtz_Pair_V1 (const _Complex double k0,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (zeroArray)
  {
    memset(V1, 0, 4*nCharge1*sizeof(_Complex double));
    memset(V2, 0, 4*nCharge2*sizeof(_Complex double));
  }
  for (size_t i = 0; i < nCharge1; i++)
  {
    const double x1 = coord1[3*i  ];
    const double y1 = coord1[3*i+1];
    const double z1 = coord1[3*i+2];

    for (size_t j = 0; j < nCharge2; j++)
    {
      const double x            = x1 - coord2[3*j  ];
      const double y            = y1 - coord2[3*j+1];
      const double z            = z1 - coord2[3*j+2];
      const double r2           = x*x + y*y + z*z;
      const double r            = sqrt(fabs(r2));

      const _Complex double f0  = cexp(I*k0*r)/r;
      const _Complex double f1  = (I*k0*f0 - f0/r)/r;
      const _Complex double fX  = x*f1;
      const _Complex double fY  = y*f1;
      const _Complex double fZ  = z*f1;


      V1[4*i  ] += f0*q2[j];
      V1[4*i+1] += fX*q2[j];
      V1[4*i+2] += fY*q2[j];
      V1[4*i+3] += fZ*q2[j];

      V2[4*j  ] += f0*q1[i];
      V2[4*j+1] -= fX*q1[i];
      V2[4*j+2] -= fY*q1[i];
      V2[4*j+3] -= fZ*q1[i];
    }
  }
}


void NPME_PotHelmholtz_Self_V1 (const _Complex double k0,
  const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (zeroArray)
    memset(V, 0, 4*nCharge*sizeof(_Complex double));

  for (size_t i = 0; i < nCharge; i++)
  {
    const double x1 = coord[3*i  ];
    const double y1 = coord[3*i+1];
    const double z1 = coord[3*i+2];

    for (size_t j = 0; j < i; j++)
    {
      const double x            = x1 - coord[3*j  ];
      const double y            = y1 - coord[3*j+1];
      const double z            = z1 - coord[3*j+2];

      const double r2           = x*x + y*y + z*z;
      const double r            = sqrt(fabs(r2));

      const _Complex double f0  = cexp(I*k0*r)/r;
      const _Complex double f1  = (I*k0*f0 - f0/r)/r;
      const _Complex double fX  = x*f1;
      const _Complex double fY  = y*f1;
      const _Complex double fZ  = z*f1;

      V[4*i  ] += f0*q[j];
      V[4*i+1] += fX*q[j];
      V[4*i+2] += fY*q[j];
      V[4*i+3] += fZ*q[j];

      V[4*j  ] += f0*q[i];
      V[4*j+1] -= fX*q[i];
      V[4*j+2] -= fY*q[i];
      V[4*j+3] -= fZ*q[i];
    }
  }
}


void NPME_PotHelmholtz_SR_DM_Pair_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        a[Nder+1], b[Nder+1]
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (zeroArray)
  {
    memset(V1, 0, 4*nCharge1*sizeof(_Complex double));
    memset(V2, 0, 4*nCharge2*sizeof(_Complex double));
  }

  for (size_t i = 0; i < nCharge1; i++)
  {
    const double x1 = coord1[3*i  ];
    const double y1 = coord1[3*i+1];
    const double z1 = coord1[3*i+2];

    for (size_t j = 0; j < nCharge2; j++)
    {
      const double x            = x1 - coord2[3*j  ];
      const double y            = y1 - coord2[3*j+1];
      const double z            = z1 - coord2[3*j+2];
      const double r2           = x*x + y*y + z*z;
      const double r            = sqrt(fabs(r2));


      _Complex double f0, f1;
      if (r < Rdir)
      {
        _Complex double f0_exact  = cexp(I*k0*r)/r;
        _Complex double f1_exact  = (I*k0*f0_exact - f0_exact/r)/r;

        f0 = f0_exact - NPME_FunctionDerivMatch_EvenSeriesComplex (f1, 
                              Nder, a, b, r2);
        f1 = f1_exact - f1;
      }
      else
      {
        f0 = 0;
        f1 = 0;
      }
      const _Complex double fX  = x*f1;
      const _Complex double fY  = y*f1;
      const _Complex double fZ  = z*f1;

      V1[4*i  ] += f0*q2[j];
      V1[4*i+1] += fX*q2[j];
      V1[4*i+2] += fY*q2[j];
      V1[4*i+3] += fZ*q2[j];

      V2[4*j  ] += f0*q1[i];
      V2[4*j+1] -= fX*q1[i];
      V2[4*j+2] -= fY*q1[i];
      V2[4*j+3] -= fZ*q1[i];
    }
  }
}


void NPME_PotHelmholtz_SR_DM_Self_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir, const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//        a[Nder+1], b[Nder+1]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (zeroArray)
    memset(V, 0, 4*nCharge*sizeof(_Complex double));

  for (size_t i = 0; i < nCharge; i++)
  {
    const double x1 = coord[3*i  ];
    const double y1 = coord[3*i+1];
    const double z1 = coord[3*i+2];

    for (size_t j = 0; j < i; j++)
    {
      const double x            = x1 - coord[3*j  ];
      const double y            = y1 - coord[3*j+1];
      const double z            = z1 - coord[3*j+2];

      const double r2           = x*x + y*y + z*z;
      const double r            = sqrt(fabs(r2));

      _Complex double f0, f1;
      if (r < Rdir)
      {
        _Complex double f0_exact  = cexp(I*k0*r)/r;
        _Complex double f1_exact  = (I*k0*f0_exact - f0_exact/r)/r;

        f0 = f0_exact - NPME_FunctionDerivMatch_EvenSeriesComplex (f1, 
                              Nder, a, b, r2);
        f1 = f1_exact - f1;
      }
      else
      {
        f0 = 0;
        f1 = 0;
      }
      const _Complex double fX  = x*f1;
      const _Complex double fY  = y*f1;
      const _Complex double fZ  = z*f1;

      V[4*i  ] += f0*q[j];
      V[4*i+1] += fX*q[j];
      V[4*i+2] += fY*q[j];
      V[4*i+3] += fZ*q[j];

      V[4*j  ] += f0*q[i];
      V[4*j+1] -= fX*q[i];
      V[4*j+2] -= fY*q[i];
      V[4*j+3] -= fZ*q[i];
    }
  }
}



#if NPME_USE_AVX


void NPME_PotHelmholtz_AVX (const _Complex double k0,
  __m256d& f0r_Vec, __m256d& f0i_Vec, __m256d& fXr_Vec, __m256d& fXi_Vec,
  __m256d& fYr_Vec, __m256d& fYi_Vec, __m256d& fZr_Vec, __m256d& fZi_Vec,
  const __m256d& xVec, const __m256d& yVec, const __m256d& zVec)
{
  const __m256d k0rVec  = _mm256_set1_pd( creal(k0) );
  const __m256d nk0iVec = _mm256_set1_pd(-cimag(k0) );

  __m256d r2Vec = _mm256_mul_pd  (xVec, xVec);
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
  __m256d rVec  = _mm256_sqrt_pd (r2Vec);

  __m256d f1r_Vec, f1i_Vec;
  {
    __m256d sinVec, cosVec, expVec;
    sinVec   = _mm256_sincos_pd (&cosVec, _mm256_mul_pd (k0rVec, rVec));
    expVec   = _mm256_exp_pd  (_mm256_mul_pd (nk0iVec, rVec));
    expVec   = _mm256_div_pd  (expVec, rVec);
    f0r_Vec  = _mm256_mul_pd  (expVec,    cosVec);
    f0i_Vec  = _mm256_mul_pd  (expVec,    sinVec);

    //g = f0/r
    __m256d gr_Vec = _mm256_div_pd  (f0r_Vec , rVec);
    __m256d gi_Vec = _mm256_div_pd  (f0i_Vec , rVec);

    //f1 = 1/r df0/dr
    #if NPME_USE_AVX_FMA
    {
      f1r_Vec  = _mm256_fmadd_pd  ( k0rVec, f0i_Vec,  gr_Vec);
      f1r_Vec  = _mm256_fmsub_pd  (nk0iVec, f0r_Vec, f1r_Vec);
      f1i_Vec  = _mm256_fmsub_pd  (nk0iVec, f0i_Vec,  gi_Vec);
      f1i_Vec  = _mm256_fmadd_pd  ( k0rVec, f0r_Vec, f1i_Vec);
    }
    #else
    {
      f1r_Vec  = _mm256_add_pd  (_mm256_mul_pd ( k0rVec, f0i_Vec),  gr_Vec);
      f1r_Vec  = _mm256_sub_pd  (_mm256_mul_pd (nk0iVec, f0r_Vec), f1r_Vec);
      f1i_Vec  = _mm256_sub_pd  (_mm256_mul_pd (nk0iVec, f0i_Vec),  gi_Vec);
      f1i_Vec  = _mm256_add_pd  (_mm256_mul_pd ( k0rVec, f0r_Vec), f1i_Vec);
    }
    #endif

    f1r_Vec = _mm256_div_pd  (f1r_Vec, rVec);
    f1i_Vec = _mm256_div_pd  (f1i_Vec, rVec);
  }

  fXr_Vec = _mm256_mul_pd  (f1r_Vec, xVec);
  fYr_Vec = _mm256_mul_pd  (f1r_Vec, yVec);
  fZr_Vec = _mm256_mul_pd  (f1r_Vec, zVec);

  fXi_Vec = _mm256_mul_pd  (f1i_Vec, xVec);
  fYi_Vec = _mm256_mul_pd  (f1i_Vec, yVec);
  fZi_Vec = _mm256_mul_pd  (f1i_Vec, zVec);
}



void NPME_PotHelmholtz_Pair_V1_AVX (const _Complex double k0,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotHelmholtz_Pair_V1_AVX.\n";
    sprintf(str, "nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    std::cout << str;
    exit(0);
  }

  const size_t remain2    = nCharge2%4;
  const size_t nLoop2     = (nCharge2 - remain2)/4;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  double qCoCrd_4x[5*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_4x    [8*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));

  memset(V2_4x, 0, 32*nLoop2wRemainder*sizeof(double));
  NPME_TransformQcomplexCoord_4x_AVX (nCharge2, q2, coord2, qCoCrd_4x);

  for (size_t i = 0; i < nCharge1; i++)
  {
    const double *r1        = &coord1[3*i];
    const __m256d q1r_Vec   = _mm256_set1_pd(creal( q1[i]));
    const __m256d q1i_Vec   = _mm256_set1_pd(cimag( q1[i]));
    const __m256d q1Nr_Vec  = _mm256_set1_pd(creal(-q1[i]));

    const __m256d x1Vec     = _mm256_set1_pd(r1[0]);
    const __m256d y1Vec     = _mm256_set1_pd(r1[1]);
    const __m256d z1Vec     = _mm256_set1_pd(r1[2]);

    __m256d V0r_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VXr_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VYr_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VZr_1_Vec       = _mm256_set1_pd(0.0);

    __m256d V0i_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VXi_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VYi_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VZi_1_Vec       = _mm256_set1_pd(0.0);

    size_t index20j = 0;
    size_t index32j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      const __m256d q2r_Vec = _mm256_load_pd (&qCoCrd_4x[index20j   ]);
      const __m256d q2i_Vec = _mm256_load_pd (&qCoCrd_4x[index20j+ 4]);

      __m256d f0r_Vec, fXr_Vec, fYr_Vec, fZr_Vec;
      __m256d f0i_Vec, fXi_Vec, fYi_Vec, fZi_Vec;
      //calc f0, fX, fY, fZ
      {
        __m256d xVec        = _mm256_load_pd (&qCoCrd_4x[index20j+ 8]);
        __m256d yVec        = _mm256_load_pd (&qCoCrd_4x[index20j+12]);
        __m256d zVec        = _mm256_load_pd (&qCoCrd_4x[index20j+16]);

        xVec = _mm256_sub_pd (x1Vec, xVec);
        yVec = _mm256_sub_pd (y1Vec, yVec);
        zVec = _mm256_sub_pd (z1Vec, zVec);

        NPME_PotHelmholtz_AVX (k0,
          f0r_Vec, f0i_Vec, fXr_Vec, fXi_Vec,
          fYr_Vec, fYi_Vec, fZr_Vec, fZi_Vec,
          xVec, yVec, zVec);
      }

      //V   = f*q
      //V_r = f_r*q_r - f_i*q_i
      //V_i = f_r*q_i + f_i*q_r
  
      __m256d V0r_2_Vec = _mm256_load_pd (&V2_4x[index32j   ]);
      __m256d V0i_2_Vec = _mm256_load_pd (&V2_4x[index32j+ 4]);
      __m256d VXr_2_Vec = _mm256_load_pd (&V2_4x[index32j+ 8]);
      __m256d VXi_2_Vec = _mm256_load_pd (&V2_4x[index32j+12]);
      __m256d VYr_2_Vec = _mm256_load_pd (&V2_4x[index32j+16]);
      __m256d VYi_2_Vec = _mm256_load_pd (&V2_4x[index32j+20]);
      __m256d VZr_2_Vec = _mm256_load_pd (&V2_4x[index32j+24]);
      __m256d VZi_2_Vec = _mm256_load_pd (&V2_4x[index32j+28]);

      #if NPME_USE_AVX_FMA
      {
        //V1 contribution
        V0r_1_Vec  = _mm256_fmsub_pd  (f0i_Vec, q2i_Vec, V0r_1_Vec);
        V0r_1_Vec  = _mm256_fmsub_pd  (f0r_Vec, q2r_Vec, V0r_1_Vec);
        VXr_1_Vec  = _mm256_fmsub_pd  (fXi_Vec, q2i_Vec, VXr_1_Vec);
        VXr_1_Vec  = _mm256_fmsub_pd  (fXr_Vec, q2r_Vec, VXr_1_Vec);
        VYr_1_Vec  = _mm256_fmsub_pd  (fYi_Vec, q2i_Vec, VYr_1_Vec);
        VYr_1_Vec  = _mm256_fmsub_pd  (fYr_Vec, q2r_Vec, VYr_1_Vec);
        VZr_1_Vec  = _mm256_fmsub_pd  (fZi_Vec, q2i_Vec, VZr_1_Vec);
        VZr_1_Vec  = _mm256_fmsub_pd  (fZr_Vec, q2r_Vec, VZr_1_Vec);

        V0i_1_Vec  = _mm256_fmadd_pd  (f0r_Vec, q2i_Vec, V0i_1_Vec);
        V0i_1_Vec  = _mm256_fmadd_pd  (f0i_Vec, q2r_Vec, V0i_1_Vec);
        VXi_1_Vec  = _mm256_fmadd_pd  (fXr_Vec, q2i_Vec, VXi_1_Vec);
        VXi_1_Vec  = _mm256_fmadd_pd  (fXi_Vec, q2r_Vec, VXi_1_Vec);
        VYi_1_Vec  = _mm256_fmadd_pd  (fYr_Vec, q2i_Vec, VYi_1_Vec);
        VYi_1_Vec  = _mm256_fmadd_pd  (fYi_Vec, q2r_Vec, VYi_1_Vec);
        VZi_1_Vec  = _mm256_fmadd_pd  (fZr_Vec, q2i_Vec, VZi_1_Vec);
        VZi_1_Vec  = _mm256_fmadd_pd  (fZi_Vec, q2r_Vec, VZi_1_Vec);

        //V2 contribution
        V0r_2_Vec  = _mm256_fmsub_pd  (f0i_Vec, q1i_Vec, V0r_2_Vec);
        V0r_2_Vec  = _mm256_fmsub_pd  (f0r_Vec, q1r_Vec, V0r_2_Vec);
        VXr_2_Vec  = _mm256_fmsub_pd  (fXr_Vec, q1r_Vec, VXr_2_Vec);
        VXr_2_Vec  = _mm256_fmsub_pd  (fXi_Vec, q1i_Vec, VXr_2_Vec);
        VYr_2_Vec  = _mm256_fmsub_pd  (fYr_Vec, q1r_Vec, VYr_2_Vec);
        VYr_2_Vec  = _mm256_fmsub_pd  (fYi_Vec, q1i_Vec, VYr_2_Vec);
        VZr_2_Vec  = _mm256_fmsub_pd  (fZr_Vec, q1r_Vec, VZr_2_Vec);
        VZr_2_Vec  = _mm256_fmsub_pd  (fZi_Vec, q1i_Vec, VZr_2_Vec);

        V0i_2_Vec  = _mm256_fmadd_pd  (f0r_Vec, q1i_Vec, V0i_2_Vec);
        V0i_2_Vec  = _mm256_fmadd_pd  (f0i_Vec, q1r_Vec, V0i_2_Vec);
        VXi_2_Vec  = _mm256_fmsub_pd  (fXr_Vec,  q1i_Vec, VXi_2_Vec);
        VXi_2_Vec  = _mm256_fmsub_pd  (fXi_Vec, q1Nr_Vec, VXi_2_Vec);
        VYi_2_Vec  = _mm256_fmsub_pd  (fYr_Vec,  q1i_Vec, VYi_2_Vec);
        VYi_2_Vec  = _mm256_fmsub_pd  (fYi_Vec, q1Nr_Vec, VYi_2_Vec);
        VZi_2_Vec  = _mm256_fmsub_pd  (fZr_Vec,  q1i_Vec, VZi_2_Vec);
        VZi_2_Vec  = _mm256_fmsub_pd  (fZi_Vec, q1Nr_Vec, VZi_2_Vec);
      }
      #else
      {
        //V1 contribution
        V0r_1_Vec = _mm256_sub_pd(_mm256_mul_pd(f0i_Vec, q2i_Vec), V0r_1_Vec);
        V0r_1_Vec = _mm256_sub_pd(_mm256_mul_pd(f0r_Vec, q2r_Vec), V0r_1_Vec);
        VXr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q2i_Vec), VXr_1_Vec);
        VXr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec, q2r_Vec), VXr_1_Vec);
        VYr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q2i_Vec), VYr_1_Vec);
        VYr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec, q2r_Vec), VYr_1_Vec);
        VZr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q2i_Vec), VZr_1_Vec);
        VZr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec, q2r_Vec), VZr_1_Vec);

        V0i_1_Vec = _mm256_add_pd(_mm256_mul_pd(f0r_Vec, q2i_Vec), V0i_1_Vec);
        V0i_1_Vec = _mm256_add_pd(_mm256_mul_pd(f0i_Vec, q2r_Vec), V0i_1_Vec);
        VXi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fXr_Vec, q2i_Vec), VXi_1_Vec);
        VXi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fXi_Vec, q2r_Vec), VXi_1_Vec);
        VYi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fYr_Vec, q2i_Vec), VYi_1_Vec);
        VYi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fYi_Vec, q2r_Vec), VYi_1_Vec);
        VZi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fZr_Vec, q2i_Vec), VZi_1_Vec);
        VZi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fZi_Vec, q2r_Vec), VZi_1_Vec);

        //V2 contribution
        V0r_2_Vec = _mm256_sub_pd(_mm256_mul_pd(f0i_Vec, q1i_Vec), V0r_2_Vec);
        V0r_2_Vec = _mm256_sub_pd(_mm256_mul_pd(f0r_Vec, q1r_Vec), V0r_2_Vec);
        VXr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec, q1r_Vec), VXr_2_Vec);
        VXr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q1i_Vec), VXr_2_Vec);
        VYr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec, q1r_Vec), VYr_2_Vec);
        VYr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q1i_Vec), VYr_2_Vec);
        VZr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec, q1r_Vec), VZr_2_Vec);
        VZr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q1i_Vec), VZr_2_Vec);

        V0i_2_Vec = _mm256_add_pd(_mm256_mul_pd(f0r_Vec,  q1i_Vec), V0i_2_Vec);
        V0i_2_Vec = _mm256_add_pd(_mm256_mul_pd(f0i_Vec,  q1r_Vec), V0i_2_Vec);
        VXi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec,  q1i_Vec), VXi_2_Vec);
        VXi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q1Nr_Vec), VXi_2_Vec);
        VYi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec,  q1i_Vec), VYi_2_Vec);
        VYi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q1Nr_Vec), VYi_2_Vec);
        VZi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec,  q1i_Vec), VZi_2_Vec);
        VZi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q1Nr_Vec), VZi_2_Vec);
      }
      #endif


      _mm256_store_pd (&V2_4x[index32j   ], V0r_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+ 4], V0i_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+ 8], VXr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+12], VXi_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+16], VYr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+20], VYi_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+24], VZr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+28], VZi_2_Vec);


      index20j += 20;
      index32j += 32;
    }

    //accumulate and store V1 contributions
    double sum_r[4];
    double sum_i[4];
    NPME_mm256_4x4HorizontalSum_pd (sum_r, V0r_1_Vec, VXr_1_Vec, 
                                       VYr_1_Vec, VZr_1_Vec);
    NPME_mm256_4x4HorizontalSum_pd (sum_i, V0i_1_Vec, VXi_1_Vec, 
                                       VYi_1_Vec, VZi_1_Vec);

    if (zeroArray)
    {
      V1[4*i  ]  = sum_r[0] + I*sum_i[0];
      V1[4*i+1]  = sum_r[1] + I*sum_i[1];
      V1[4*i+2]  = sum_r[2] + I*sum_i[2];
      V1[4*i+3]  = sum_r[3] + I*sum_i[3];
    }
    else
    {
      V1[4*i  ]  += sum_r[0] + I*sum_i[0];
      V1[4*i+1]  += sum_r[1] + I*sum_i[1];
      V1[4*i+2]  += sum_r[2] + I*sum_i[2];
      V1[4*i+3]  += sum_r[3] + I*sum_i[3];
    }
  }

  if (zeroArray)
    NPME_TransformComplexV1_4x_2_V1_AVX (nCharge2, V2_4x, &V2[0]);
  else
    NPME_TransformUpdateComplexV1_4x_2_V1_AVX (nCharge2, V2_4x, &V2[0]);
}



void NPME_PotHelmholtz_Self_V1_AVX (const _Complex double k0,
  const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotHelmholtz_Self_V1_AVX.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  double qCoCrd_4x[5*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  NPME_TransformQcomplexCoord_4x_AVX (nCharge, q, coord, qCoCrd_4x);

  double V2_4x[8*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  {
    const size_t remain    = nCharge%4;
    const size_t nLoop     = (nCharge - remain)/4;
    size_t nLoopwRemainder = nLoop;
    if (remain > 0)
      nLoopwRemainder++;

    memset(V2_4x, 0, 32*nLoopwRemainder*sizeof(double));
  }


  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r1        = &coord[3*i];
    __m256d q1r_Vec         = _mm256_set1_pd(creal( q[i]));
    __m256d q1i_Vec         = _mm256_set1_pd(cimag( q[i]));
    __m256d q1Nr_Vec        = _mm256_set1_pd(creal(-q[i]));

    const __m256d x1Vec     = _mm256_set1_pd(r1[0]);
    const __m256d y1Vec     = _mm256_set1_pd(r1[1]);
    const __m256d z1Vec     = _mm256_set1_pd(r1[2]);

    __m256d V0r_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VXr_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VYr_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VZr_1_Vec       = _mm256_set1_pd(0.0);

    __m256d V0i_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VXi_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VYi_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VZi_1_Vec       = _mm256_set1_pd(0.0);

    const size_t remain2  = (i)%4;
    const size_t nLoop2   = (i-remain2)/4;

    size_t index20j = 0;
    size_t index32j = 0;
    for (size_t j = 0; j < nLoop2; j++)
    {
      const __m256d q2r_Vec = _mm256_load_pd (&qCoCrd_4x[index20j   ]);
      const __m256d q2i_Vec = _mm256_load_pd (&qCoCrd_4x[index20j+ 4]);

      __m256d f0r_Vec, fXr_Vec, fYr_Vec, fZr_Vec;
      __m256d f0i_Vec, fXi_Vec, fYi_Vec, fZi_Vec;
      //calc f0, fX, fY, fZ
      {
        __m256d xVec        = _mm256_load_pd (&qCoCrd_4x[index20j+ 8]);
        __m256d yVec        = _mm256_load_pd (&qCoCrd_4x[index20j+12]);
        __m256d zVec        = _mm256_load_pd (&qCoCrd_4x[index20j+16]);

        xVec = _mm256_sub_pd (x1Vec, xVec);
        yVec = _mm256_sub_pd (y1Vec, yVec);
        zVec = _mm256_sub_pd (z1Vec, zVec);

        NPME_PotHelmholtz_AVX (k0,
          f0r_Vec, f0i_Vec, fXr_Vec, fXi_Vec,
          fYr_Vec, fYi_Vec, fZr_Vec, fZi_Vec,
          xVec, yVec, zVec);
      }

      //V   = f*q
      //V_r = f_r*q_r - f_i*q_i
      //V_i = f_r*q_i + f_i*q_r
  
      __m256d V0r_2_Vec = _mm256_load_pd (&V2_4x[index32j   ]);
      __m256d V0i_2_Vec = _mm256_load_pd (&V2_4x[index32j+ 4]);
      __m256d VXr_2_Vec = _mm256_load_pd (&V2_4x[index32j+ 8]);
      __m256d VXi_2_Vec = _mm256_load_pd (&V2_4x[index32j+12]);
      __m256d VYr_2_Vec = _mm256_load_pd (&V2_4x[index32j+16]);
      __m256d VYi_2_Vec = _mm256_load_pd (&V2_4x[index32j+20]);
      __m256d VZr_2_Vec = _mm256_load_pd (&V2_4x[index32j+24]);
      __m256d VZi_2_Vec = _mm256_load_pd (&V2_4x[index32j+28]);


      //V   = f*q
      //V_r = f_r*q_r - f_i*q_i
      //V_i = f_r*q_i + f_i*q_r
  
      #if NPME_USE_AVX_FMA
      {
        //V1 contribution
        V0r_1_Vec  = _mm256_fmsub_pd  (f0i_Vec, q2i_Vec, V0r_1_Vec);
        V0r_1_Vec  = _mm256_fmsub_pd  (f0r_Vec, q2r_Vec, V0r_1_Vec);
        VXr_1_Vec  = _mm256_fmsub_pd  (fXi_Vec, q2i_Vec, VXr_1_Vec);
        VXr_1_Vec  = _mm256_fmsub_pd  (fXr_Vec, q2r_Vec, VXr_1_Vec);
        VYr_1_Vec  = _mm256_fmsub_pd  (fYi_Vec, q2i_Vec, VYr_1_Vec);
        VYr_1_Vec  = _mm256_fmsub_pd  (fYr_Vec, q2r_Vec, VYr_1_Vec);
        VZr_1_Vec  = _mm256_fmsub_pd  (fZi_Vec, q2i_Vec, VZr_1_Vec);
        VZr_1_Vec  = _mm256_fmsub_pd  (fZr_Vec, q2r_Vec, VZr_1_Vec);

        V0i_1_Vec  = _mm256_fmadd_pd  (f0r_Vec, q2i_Vec, V0i_1_Vec);
        V0i_1_Vec  = _mm256_fmadd_pd  (f0i_Vec, q2r_Vec, V0i_1_Vec);
        VXi_1_Vec  = _mm256_fmadd_pd  (fXr_Vec, q2i_Vec, VXi_1_Vec);
        VXi_1_Vec  = _mm256_fmadd_pd  (fXi_Vec, q2r_Vec, VXi_1_Vec);
        VYi_1_Vec  = _mm256_fmadd_pd  (fYr_Vec, q2i_Vec, VYi_1_Vec);
        VYi_1_Vec  = _mm256_fmadd_pd  (fYi_Vec, q2r_Vec, VYi_1_Vec);
        VZi_1_Vec  = _mm256_fmadd_pd  (fZr_Vec, q2i_Vec, VZi_1_Vec);
        VZi_1_Vec  = _mm256_fmadd_pd  (fZi_Vec, q2r_Vec, VZi_1_Vec);

        //V2 contribution
        V0r_2_Vec  = _mm256_fmsub_pd  (f0i_Vec, q1i_Vec, V0r_2_Vec);
        V0r_2_Vec  = _mm256_fmsub_pd  (f0r_Vec, q1r_Vec, V0r_2_Vec);
        VXr_2_Vec  = _mm256_fmsub_pd  (fXr_Vec, q1r_Vec, VXr_2_Vec);
        VXr_2_Vec  = _mm256_fmsub_pd  (fXi_Vec, q1i_Vec, VXr_2_Vec);
        VYr_2_Vec  = _mm256_fmsub_pd  (fYr_Vec, q1r_Vec, VYr_2_Vec);
        VYr_2_Vec  = _mm256_fmsub_pd  (fYi_Vec, q1i_Vec, VYr_2_Vec);
        VZr_2_Vec  = _mm256_fmsub_pd  (fZr_Vec, q1r_Vec, VZr_2_Vec);
        VZr_2_Vec  = _mm256_fmsub_pd  (fZi_Vec, q1i_Vec, VZr_2_Vec);

        V0i_2_Vec  = _mm256_fmadd_pd  (f0r_Vec, q1i_Vec, V0i_2_Vec);
        V0i_2_Vec  = _mm256_fmadd_pd  (f0i_Vec, q1r_Vec, V0i_2_Vec);
        VXi_2_Vec  = _mm256_fmsub_pd  (fXr_Vec,  q1i_Vec, VXi_2_Vec);
        VXi_2_Vec  = _mm256_fmsub_pd  (fXi_Vec, q1Nr_Vec, VXi_2_Vec);
        VYi_2_Vec  = _mm256_fmsub_pd  (fYr_Vec,  q1i_Vec, VYi_2_Vec);
        VYi_2_Vec  = _mm256_fmsub_pd  (fYi_Vec, q1Nr_Vec, VYi_2_Vec);
        VZi_2_Vec  = _mm256_fmsub_pd  (fZr_Vec,  q1i_Vec, VZi_2_Vec);
        VZi_2_Vec  = _mm256_fmsub_pd  (fZi_Vec, q1Nr_Vec, VZi_2_Vec);
      }
      #else
      {
        //V1 contribution
        V0r_1_Vec = _mm256_sub_pd(_mm256_mul_pd(f0i_Vec, q2i_Vec), V0r_1_Vec);
        V0r_1_Vec = _mm256_sub_pd(_mm256_mul_pd(f0r_Vec, q2r_Vec), V0r_1_Vec);
        VXr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q2i_Vec), VXr_1_Vec);
        VXr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec, q2r_Vec), VXr_1_Vec);
        VYr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q2i_Vec), VYr_1_Vec);
        VYr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec, q2r_Vec), VYr_1_Vec);
        VZr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q2i_Vec), VZr_1_Vec);
        VZr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec, q2r_Vec), VZr_1_Vec);

        V0i_1_Vec = _mm256_add_pd(_mm256_mul_pd(f0r_Vec, q2i_Vec), V0i_1_Vec);
        V0i_1_Vec = _mm256_add_pd(_mm256_mul_pd(f0i_Vec, q2r_Vec), V0i_1_Vec);
        VXi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fXr_Vec, q2i_Vec), VXi_1_Vec);
        VXi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fXi_Vec, q2r_Vec), VXi_1_Vec);
        VYi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fYr_Vec, q2i_Vec), VYi_1_Vec);
        VYi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fYi_Vec, q2r_Vec), VYi_1_Vec);
        VZi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fZr_Vec, q2i_Vec), VZi_1_Vec);
        VZi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fZi_Vec, q2r_Vec), VZi_1_Vec);

        //V2 contribution
        V0r_2_Vec = _mm256_sub_pd(_mm256_mul_pd(f0i_Vec, q1i_Vec), V0r_2_Vec);
        V0r_2_Vec = _mm256_sub_pd(_mm256_mul_pd(f0r_Vec, q1r_Vec), V0r_2_Vec);
        VXr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec, q1r_Vec), VXr_2_Vec);
        VXr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q1i_Vec), VXr_2_Vec);
        VYr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec, q1r_Vec), VYr_2_Vec);
        VYr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q1i_Vec), VYr_2_Vec);
        VZr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec, q1r_Vec), VZr_2_Vec);
        VZr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q1i_Vec), VZr_2_Vec);

        V0i_2_Vec = _mm256_add_pd(_mm256_mul_pd(f0r_Vec,  q1i_Vec), V0i_2_Vec);
        V0i_2_Vec = _mm256_add_pd(_mm256_mul_pd(f0i_Vec,  q1r_Vec), V0i_2_Vec);
        VXi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec,  q1i_Vec), VXi_2_Vec);
        VXi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q1Nr_Vec), VXi_2_Vec);
        VYi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec,  q1i_Vec), VYi_2_Vec);
        VYi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q1Nr_Vec), VYi_2_Vec);
        VZi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec,  q1i_Vec), VZi_2_Vec);
        VZi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q1Nr_Vec), VZi_2_Vec);
      }
      #endif

      _mm256_store_pd (&V2_4x[index32j   ], V0r_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+ 4], V0i_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+ 8], VXr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+12], VXi_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+16], VYr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+20], VYi_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+24], VZr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+28], VZi_2_Vec);

      index20j += 20;
      index32j += 32;
    }

    if (remain2 > 0)
    {
      const size_t indexStart     = 4*nLoop2;
      const _Complex double *qLoc = &q[indexStart];
      const double *crdLoc        = &coord[3*indexStart];
      const double X              = r1[0]+1.0;
      double x2Array[4]  __attribute__((aligned(64))) = {X, X, X, X};
      double y2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double z2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double q2rArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};
      double q2iArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};
      double mskArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};

      for (size_t k = 0; k < remain2; k++)
      {
        q2rArray[k] = creal(qLoc[k]);
        q2iArray[k] = cimag(qLoc[k]);
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
        mskArray[k] = 1.0;
      }
      __m256d q2r_Vec   = _mm256_load_pd (q2rArray);
      __m256d q2i_Vec   = _mm256_load_pd (q2iArray);
      __m256d xVec      = _mm256_load_pd (x2Array);
      __m256d yVec      = _mm256_load_pd (y2Array);
      __m256d zVec      = _mm256_load_pd (z2Array);
      __m256d mskVec    = _mm256_load_pd (mskArray);

      //apply mask to q1
      q1r_Vec   = _mm256_mul_pd (mskVec, q1r_Vec);
      q1i_Vec   = _mm256_mul_pd (mskVec, q1i_Vec);
      q1Nr_Vec  = _mm256_mul_pd (mskVec, q1Nr_Vec);

      __m256d V0r_2_Vec = _mm256_load_pd (&V2_4x[index32j   ]);
      __m256d V0i_2_Vec = _mm256_load_pd (&V2_4x[index32j+ 4]);
      __m256d VXr_2_Vec = _mm256_load_pd (&V2_4x[index32j+ 8]);
      __m256d VXi_2_Vec = _mm256_load_pd (&V2_4x[index32j+12]);
      __m256d VYr_2_Vec = _mm256_load_pd (&V2_4x[index32j+16]);
      __m256d VYi_2_Vec = _mm256_load_pd (&V2_4x[index32j+20]);
      __m256d VZr_2_Vec = _mm256_load_pd (&V2_4x[index32j+24]);
      __m256d VZi_2_Vec = _mm256_load_pd (&V2_4x[index32j+28]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      __m256d f0r_Vec, fXr_Vec, fYr_Vec, fZr_Vec;
      __m256d f0i_Vec, fXi_Vec, fYi_Vec, fZi_Vec;
      //calc f0, fX, fY, fZ
      {
        NPME_PotHelmholtz_AVX (k0,
          f0r_Vec, f0i_Vec, fXr_Vec, fXi_Vec,
          fYr_Vec, fYi_Vec, fZr_Vec, fZi_Vec,
          xVec, yVec, zVec);
      }
      
      #if NPME_USE_AVX_FMA
      {
        //V1 contribution
        V0r_1_Vec  = _mm256_fmsub_pd  (f0i_Vec, q2i_Vec, V0r_1_Vec);
        V0r_1_Vec  = _mm256_fmsub_pd  (f0r_Vec, q2r_Vec, V0r_1_Vec);
        VXr_1_Vec  = _mm256_fmsub_pd  (fXi_Vec, q2i_Vec, VXr_1_Vec);
        VXr_1_Vec  = _mm256_fmsub_pd  (fXr_Vec, q2r_Vec, VXr_1_Vec);
        VYr_1_Vec  = _mm256_fmsub_pd  (fYi_Vec, q2i_Vec, VYr_1_Vec);
        VYr_1_Vec  = _mm256_fmsub_pd  (fYr_Vec, q2r_Vec, VYr_1_Vec);
        VZr_1_Vec  = _mm256_fmsub_pd  (fZi_Vec, q2i_Vec, VZr_1_Vec);
        VZr_1_Vec  = _mm256_fmsub_pd  (fZr_Vec, q2r_Vec, VZr_1_Vec);

        V0i_1_Vec  = _mm256_fmadd_pd  (f0r_Vec, q2i_Vec, V0i_1_Vec);
        V0i_1_Vec  = _mm256_fmadd_pd  (f0i_Vec, q2r_Vec, V0i_1_Vec);
        VXi_1_Vec  = _mm256_fmadd_pd  (fXr_Vec, q2i_Vec, VXi_1_Vec);
        VXi_1_Vec  = _mm256_fmadd_pd  (fXi_Vec, q2r_Vec, VXi_1_Vec);
        VYi_1_Vec  = _mm256_fmadd_pd  (fYr_Vec, q2i_Vec, VYi_1_Vec);
        VYi_1_Vec  = _mm256_fmadd_pd  (fYi_Vec, q2r_Vec, VYi_1_Vec);
        VZi_1_Vec  = _mm256_fmadd_pd  (fZr_Vec, q2i_Vec, VZi_1_Vec);
        VZi_1_Vec  = _mm256_fmadd_pd  (fZi_Vec, q2r_Vec, VZi_1_Vec);

        //V2 contribution
        V0r_2_Vec  = _mm256_fmsub_pd  (f0i_Vec, q1i_Vec, V0r_2_Vec);
        V0r_2_Vec  = _mm256_fmsub_pd  (f0r_Vec, q1r_Vec, V0r_2_Vec);
        VXr_2_Vec  = _mm256_fmsub_pd  (fXr_Vec, q1r_Vec, VXr_2_Vec);
        VXr_2_Vec  = _mm256_fmsub_pd  (fXi_Vec, q1i_Vec, VXr_2_Vec);
        VYr_2_Vec  = _mm256_fmsub_pd  (fYr_Vec, q1r_Vec, VYr_2_Vec);
        VYr_2_Vec  = _mm256_fmsub_pd  (fYi_Vec, q1i_Vec, VYr_2_Vec);
        VZr_2_Vec  = _mm256_fmsub_pd  (fZr_Vec, q1r_Vec, VZr_2_Vec);
        VZr_2_Vec  = _mm256_fmsub_pd  (fZi_Vec, q1i_Vec, VZr_2_Vec);

        V0i_2_Vec  = _mm256_fmadd_pd  (f0r_Vec, q1i_Vec, V0i_2_Vec);
        V0i_2_Vec  = _mm256_fmadd_pd  (f0i_Vec, q1r_Vec, V0i_2_Vec);
        VXi_2_Vec  = _mm256_fmsub_pd  (fXr_Vec,  q1i_Vec, VXi_2_Vec);
        VXi_2_Vec  = _mm256_fmsub_pd  (fXi_Vec, q1Nr_Vec, VXi_2_Vec);
        VYi_2_Vec  = _mm256_fmsub_pd  (fYr_Vec,  q1i_Vec, VYi_2_Vec);
        VYi_2_Vec  = _mm256_fmsub_pd  (fYi_Vec, q1Nr_Vec, VYi_2_Vec);
        VZi_2_Vec  = _mm256_fmsub_pd  (fZr_Vec,  q1i_Vec, VZi_2_Vec);
        VZi_2_Vec  = _mm256_fmsub_pd  (fZi_Vec, q1Nr_Vec, VZi_2_Vec);
      }
      #else
      {
        //V1 contribution
        V0r_1_Vec = _mm256_sub_pd(_mm256_mul_pd(f0i_Vec, q2i_Vec), V0r_1_Vec);
        V0r_1_Vec = _mm256_sub_pd(_mm256_mul_pd(f0r_Vec, q2r_Vec), V0r_1_Vec);
        VXr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q2i_Vec), VXr_1_Vec);
        VXr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec, q2r_Vec), VXr_1_Vec);
        VYr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q2i_Vec), VYr_1_Vec);
        VYr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec, q2r_Vec), VYr_1_Vec);
        VZr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q2i_Vec), VZr_1_Vec);
        VZr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec, q2r_Vec), VZr_1_Vec);

        V0i_1_Vec = _mm256_add_pd(_mm256_mul_pd(f0r_Vec, q2i_Vec), V0i_1_Vec);
        V0i_1_Vec = _mm256_add_pd(_mm256_mul_pd(f0i_Vec, q2r_Vec), V0i_1_Vec);
        VXi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fXr_Vec, q2i_Vec), VXi_1_Vec);
        VXi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fXi_Vec, q2r_Vec), VXi_1_Vec);
        VYi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fYr_Vec, q2i_Vec), VYi_1_Vec);
        VYi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fYi_Vec, q2r_Vec), VYi_1_Vec);
        VZi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fZr_Vec, q2i_Vec), VZi_1_Vec);
        VZi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fZi_Vec, q2r_Vec), VZi_1_Vec);

        //V2 contribution
        V0r_2_Vec = _mm256_sub_pd(_mm256_mul_pd(f0i_Vec, q1i_Vec), V0r_2_Vec);
        V0r_2_Vec = _mm256_sub_pd(_mm256_mul_pd(f0r_Vec, q1r_Vec), V0r_2_Vec);
        VXr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec, q1r_Vec), VXr_2_Vec);
        VXr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q1i_Vec), VXr_2_Vec);
        VYr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec, q1r_Vec), VYr_2_Vec);
        VYr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q1i_Vec), VYr_2_Vec);
        VZr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec, q1r_Vec), VZr_2_Vec);
        VZr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q1i_Vec), VZr_2_Vec);

        V0i_2_Vec = _mm256_add_pd(_mm256_mul_pd(f0r_Vec,  q1i_Vec), V0i_2_Vec);
        V0i_2_Vec = _mm256_add_pd(_mm256_mul_pd(f0i_Vec,  q1r_Vec), V0i_2_Vec);
        VXi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec,  q1i_Vec), VXi_2_Vec);
        VXi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q1Nr_Vec), VXi_2_Vec);
        VYi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec,  q1i_Vec), VYi_2_Vec);
        VYi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q1Nr_Vec), VYi_2_Vec);
        VZi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec,  q1i_Vec), VZi_2_Vec);
        VZi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q1Nr_Vec), VZi_2_Vec);
      }
      #endif

      _mm256_store_pd (&V2_4x[index32j   ], V0r_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+ 4], V0i_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+ 8], VXr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+12], VXi_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+16], VYr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+20], VYi_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+24], VZr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+28], VZi_2_Vec);
    }


    //accumulate and store V1 contributions
    double sum_r[4];
    double sum_i[4];
    NPME_mm256_4x4HorizontalSum_pd (sum_r, V0r_1_Vec, VXr_1_Vec, 
                                       VYr_1_Vec, VZr_1_Vec);
    NPME_mm256_4x4HorizontalSum_pd (sum_i, V0i_1_Vec, VXi_1_Vec, 
                                       VYi_1_Vec, VZi_1_Vec);

    if (zeroArray)
    {
      V[4*i  ]  = sum_r[0] + I*sum_i[0];
      V[4*i+1]  = sum_r[1] + I*sum_i[1];
      V[4*i+2]  = sum_r[2] + I*sum_i[2];
      V[4*i+3]  = sum_r[3] + I*sum_i[3];
    }
    else
    {
      V[4*i  ]  += sum_r[0] + I*sum_i[0];
      V[4*i+1]  += sum_r[1] + I*sum_i[1];
      V[4*i+2]  += sum_r[2] + I*sum_i[2];
      V[4*i+3]  += sum_r[3] + I*sum_i[3];
    }
  }

  NPME_TransformUpdateComplexV1_4x_2_V1_AVX (nCharge, V2_4x, V);
}





void NPME_PotHelmholtz_SR_DM_Radial_AVX (__m256d& f0rVec, __m256d& f0iVec,
  __m256d& f1rVec, __m256d& f1iVec, const __m256d& rVec, 
  const __m256d& k0rVec, const __m256d& nk0iVec)
{
  __m256d sinVec, cosVec, expVec;
  sinVec  = _mm256_sincos_pd (&cosVec, _mm256_mul_pd (k0rVec, rVec));

  expVec  = _mm256_exp_pd  (_mm256_mul_pd (nk0iVec, rVec));
  expVec  = _mm256_div_pd  (expVec, rVec);

  f0rVec  = _mm256_mul_pd  (expVec,    cosVec);
  f0iVec  = _mm256_mul_pd  (expVec,    sinVec);


  //g = f0/r
  __m256d grVec = _mm256_div_pd  (f0rVec , rVec);
  __m256d giVec = _mm256_div_pd  (f0iVec , rVec);

  //f1 = 1/r df0/dr
  #if NPME_USE_AVX_FMA
  {
    f1rVec  = _mm256_fmadd_pd  ( k0rVec, f0iVec,  grVec);
    f1rVec  = _mm256_fmsub_pd  (nk0iVec, f0rVec, f1rVec);
    f1iVec  = _mm256_fmsub_pd  (nk0iVec, f0iVec,  giVec);
    f1iVec  = _mm256_fmadd_pd  ( k0rVec, f0rVec, f1iVec);
  }
  #else
  {
    f1rVec  = _mm256_add_pd  (_mm256_mul_pd ( k0rVec, f0iVec),  grVec);
    f1rVec  = _mm256_sub_pd  (_mm256_mul_pd (nk0iVec, f0rVec), f1rVec);
    f1iVec  = _mm256_sub_pd  (_mm256_mul_pd (nk0iVec, f0iVec),  giVec);
    f1iVec  = _mm256_add_pd  (_mm256_mul_pd ( k0rVec, f0rVec), f1iVec);
  }
  #endif

  f1rVec = _mm256_div_pd  (f1rVec, rVec);
  f1iVec = _mm256_div_pd  (f1iVec, rVec);
}
void NPME_PotHelmholtz_SR_DM_AVX (const _Complex double k0,
  __m256d& f0r_Vec, __m256d& f0i_Vec, __m256d& fXr_Vec, __m256d& fXi_Vec,
  __m256d& fYr_Vec, __m256d& fYi_Vec, __m256d& fZr_Vec, __m256d& fZi_Vec,
  const __m256d& xVec, const __m256d& yVec, const __m256d& zVec,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir)
{
  const __m256d RdirVec = _mm256_set1_pd( Rdir );
  const __m256d k0rVec  = _mm256_set1_pd( creal(k0) );
  const __m256d nk0iVec = _mm256_set1_pd(-cimag(k0) );
  const __m256d zeroVec = _mm256_set1_pd(0.0);

  __m256d r2Vec = _mm256_mul_pd  (xVec, xVec);
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
  __m256d rVec  = _mm256_sqrt_pd (r2Vec);

  __m256d f1r_Vec, f1i_Vec;
  {
    __m256d f0r_AVec, f0i_AVec;
    __m256d f0r_BVec, f0i_BVec;
    __m256d f1r_AVec, f1i_AVec;
    __m256d f1r_BVec, f1i_BVec;

    NPME_FunctionDerivMatch_EvenSeriesComplex_AVX (f0r_AVec, f0i_AVec, 
      f1r_AVec, f1i_AVec, r2Vec, Nder, a, b);
    NPME_PotHelmholtz_SR_DM_Radial_AVX (f0r_BVec, f0i_BVec, 
      f1r_BVec, f1i_BVec, rVec, k0rVec, nk0iVec);

    f0r_BVec = _mm256_sub_pd (f0r_BVec, f0r_AVec);
    f0i_BVec = _mm256_sub_pd (f0i_BVec, f0i_AVec);
    f1r_BVec = _mm256_sub_pd (f1r_BVec, f1r_AVec);
    f1i_BVec = _mm256_sub_pd (f1i_BVec, f1i_AVec);

    //use (f0r_BVec, f0i_BVec) if r < Rdir
    //use (zeroVec,  zeroVec)  if r > Rdir
    {
      __m256d t0, dless, dmore;
      t0      = _mm256_cmp_pd (rVec, RdirVec, 1);

      dless   = _mm256_and_pd    (t0, f0r_BVec);
      dmore   = _mm256_andnot_pd (t0, zeroVec);
      f0r_Vec = _mm256_add_pd (dless, dmore);

      dless   = _mm256_and_pd    (t0, f0i_BVec);
      dmore   = _mm256_andnot_pd (t0, zeroVec);
      f0i_Vec = _mm256_add_pd (dless, dmore);

      dless   = _mm256_and_pd    (t0, f1r_BVec);
      dmore   = _mm256_andnot_pd (t0, zeroVec);
      f1r_Vec = _mm256_add_pd (dless, dmore);

      dless   = _mm256_and_pd    (t0, f1i_BVec);
      dmore   = _mm256_andnot_pd (t0, zeroVec);
      f1i_Vec = _mm256_add_pd (dless, dmore);
    }

  }

  fXr_Vec = _mm256_mul_pd  (f1r_Vec, xVec);
  fYr_Vec = _mm256_mul_pd  (f1r_Vec, yVec);
  fZr_Vec = _mm256_mul_pd  (f1r_Vec, zVec);

  fXi_Vec = _mm256_mul_pd  (f1i_Vec, xVec);
  fYi_Vec = _mm256_mul_pd  (f1i_Vec, yVec);
  fZi_Vec = _mm256_mul_pd  (f1i_Vec, zVec);
}

void NPME_PotHelmholtz_SR_DM_Pair_V1_AVX (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotHelmholtz_Pair_V1_AVX.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  const size_t remain2    = nCharge2%4;
  const size_t nLoop2     = (nCharge2 - remain2)/4;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  double qCoCrd_4x[5*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_4x    [8*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));

  memset(V2_4x, 0, 32*nLoop2wRemainder*sizeof(double));
  NPME_TransformQcomplexCoord_4x_AVX (nCharge2, q2, coord2, qCoCrd_4x);

  for (size_t i = 0; i < nCharge1; i++)
  {
    const double *r1        = &coord1[3*i];
    const __m256d q1r_Vec   = _mm256_set1_pd(creal( q1[i]));
    const __m256d q1i_Vec   = _mm256_set1_pd(cimag( q1[i]));
    const __m256d q1Nr_Vec  = _mm256_set1_pd(creal(-q1[i]));

    const __m256d x1Vec     = _mm256_set1_pd(r1[0]);
    const __m256d y1Vec     = _mm256_set1_pd(r1[1]);
    const __m256d z1Vec     = _mm256_set1_pd(r1[2]);

    __m256d V0r_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VXr_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VYr_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VZr_1_Vec       = _mm256_set1_pd(0.0);

    __m256d V0i_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VXi_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VYi_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VZi_1_Vec       = _mm256_set1_pd(0.0);

    size_t index20j = 0;
    size_t index32j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      const __m256d q2r_Vec = _mm256_load_pd (&qCoCrd_4x[index20j   ]);
      const __m256d q2i_Vec = _mm256_load_pd (&qCoCrd_4x[index20j+ 4]);

      __m256d f0r_Vec, fXr_Vec, fYr_Vec, fZr_Vec;
      __m256d f0i_Vec, fXi_Vec, fYi_Vec, fZi_Vec;
      //calc f0, fX, fY, fZ
      {
        __m256d xVec        = _mm256_load_pd (&qCoCrd_4x[index20j+ 8]);
        __m256d yVec        = _mm256_load_pd (&qCoCrd_4x[index20j+12]);
        __m256d zVec        = _mm256_load_pd (&qCoCrd_4x[index20j+16]);

        xVec = _mm256_sub_pd (x1Vec, xVec);
        yVec = _mm256_sub_pd (y1Vec, yVec);
        zVec = _mm256_sub_pd (z1Vec, zVec);

        NPME_PotHelmholtz_SR_DM_AVX (k0,
          f0r_Vec, f0i_Vec, fXr_Vec, fXi_Vec,
          fYr_Vec, fYi_Vec, fZr_Vec, fZi_Vec,
          xVec, yVec, zVec, Nder, a, b, Rdir);
      }

      //V   = f*q
      //V_r = f_r*q_r - f_i*q_i
      //V_i = f_r*q_i + f_i*q_r
  
      __m256d V0r_2_Vec = _mm256_load_pd (&V2_4x[index32j   ]);
      __m256d V0i_2_Vec = _mm256_load_pd (&V2_4x[index32j+ 4]);
      __m256d VXr_2_Vec = _mm256_load_pd (&V2_4x[index32j+ 8]);
      __m256d VXi_2_Vec = _mm256_load_pd (&V2_4x[index32j+12]);
      __m256d VYr_2_Vec = _mm256_load_pd (&V2_4x[index32j+16]);
      __m256d VYi_2_Vec = _mm256_load_pd (&V2_4x[index32j+20]);
      __m256d VZr_2_Vec = _mm256_load_pd (&V2_4x[index32j+24]);
      __m256d VZi_2_Vec = _mm256_load_pd (&V2_4x[index32j+28]);

      #if NPME_USE_AVX_FMA
      {
        //V1 contribution
        V0r_1_Vec  = _mm256_fmsub_pd  (f0i_Vec, q2i_Vec, V0r_1_Vec);
        V0r_1_Vec  = _mm256_fmsub_pd  (f0r_Vec, q2r_Vec, V0r_1_Vec);
        VXr_1_Vec  = _mm256_fmsub_pd  (fXi_Vec, q2i_Vec, VXr_1_Vec);
        VXr_1_Vec  = _mm256_fmsub_pd  (fXr_Vec, q2r_Vec, VXr_1_Vec);
        VYr_1_Vec  = _mm256_fmsub_pd  (fYi_Vec, q2i_Vec, VYr_1_Vec);
        VYr_1_Vec  = _mm256_fmsub_pd  (fYr_Vec, q2r_Vec, VYr_1_Vec);
        VZr_1_Vec  = _mm256_fmsub_pd  (fZi_Vec, q2i_Vec, VZr_1_Vec);
        VZr_1_Vec  = _mm256_fmsub_pd  (fZr_Vec, q2r_Vec, VZr_1_Vec);

        V0i_1_Vec  = _mm256_fmadd_pd  (f0r_Vec, q2i_Vec, V0i_1_Vec);
        V0i_1_Vec  = _mm256_fmadd_pd  (f0i_Vec, q2r_Vec, V0i_1_Vec);
        VXi_1_Vec  = _mm256_fmadd_pd  (fXr_Vec, q2i_Vec, VXi_1_Vec);
        VXi_1_Vec  = _mm256_fmadd_pd  (fXi_Vec, q2r_Vec, VXi_1_Vec);
        VYi_1_Vec  = _mm256_fmadd_pd  (fYr_Vec, q2i_Vec, VYi_1_Vec);
        VYi_1_Vec  = _mm256_fmadd_pd  (fYi_Vec, q2r_Vec, VYi_1_Vec);
        VZi_1_Vec  = _mm256_fmadd_pd  (fZr_Vec, q2i_Vec, VZi_1_Vec);
        VZi_1_Vec  = _mm256_fmadd_pd  (fZi_Vec, q2r_Vec, VZi_1_Vec);

        //V2 contribution
        V0r_2_Vec  = _mm256_fmsub_pd  (f0i_Vec, q1i_Vec, V0r_2_Vec);
        V0r_2_Vec  = _mm256_fmsub_pd  (f0r_Vec, q1r_Vec, V0r_2_Vec);
        VXr_2_Vec  = _mm256_fmsub_pd  (fXr_Vec, q1r_Vec, VXr_2_Vec);
        VXr_2_Vec  = _mm256_fmsub_pd  (fXi_Vec, q1i_Vec, VXr_2_Vec);
        VYr_2_Vec  = _mm256_fmsub_pd  (fYr_Vec, q1r_Vec, VYr_2_Vec);
        VYr_2_Vec  = _mm256_fmsub_pd  (fYi_Vec, q1i_Vec, VYr_2_Vec);
        VZr_2_Vec  = _mm256_fmsub_pd  (fZr_Vec, q1r_Vec, VZr_2_Vec);
        VZr_2_Vec  = _mm256_fmsub_pd  (fZi_Vec, q1i_Vec, VZr_2_Vec);

        V0i_2_Vec  = _mm256_fmadd_pd  (f0r_Vec, q1i_Vec, V0i_2_Vec);
        V0i_2_Vec  = _mm256_fmadd_pd  (f0i_Vec, q1r_Vec, V0i_2_Vec);
        VXi_2_Vec  = _mm256_fmsub_pd  (fXr_Vec,  q1i_Vec, VXi_2_Vec);
        VXi_2_Vec  = _mm256_fmsub_pd  (fXi_Vec, q1Nr_Vec, VXi_2_Vec);
        VYi_2_Vec  = _mm256_fmsub_pd  (fYr_Vec,  q1i_Vec, VYi_2_Vec);
        VYi_2_Vec  = _mm256_fmsub_pd  (fYi_Vec, q1Nr_Vec, VYi_2_Vec);
        VZi_2_Vec  = _mm256_fmsub_pd  (fZr_Vec,  q1i_Vec, VZi_2_Vec);
        VZi_2_Vec  = _mm256_fmsub_pd  (fZi_Vec, q1Nr_Vec, VZi_2_Vec);
      }
      #else
      {
        //V1 contribution
        V0r_1_Vec = _mm256_sub_pd(_mm256_mul_pd(f0i_Vec, q2i_Vec), V0r_1_Vec);
        V0r_1_Vec = _mm256_sub_pd(_mm256_mul_pd(f0r_Vec, q2r_Vec), V0r_1_Vec);
        VXr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q2i_Vec), VXr_1_Vec);
        VXr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec, q2r_Vec), VXr_1_Vec);
        VYr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q2i_Vec), VYr_1_Vec);
        VYr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec, q2r_Vec), VYr_1_Vec);
        VZr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q2i_Vec), VZr_1_Vec);
        VZr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec, q2r_Vec), VZr_1_Vec);

        V0i_1_Vec = _mm256_add_pd(_mm256_mul_pd(f0r_Vec, q2i_Vec), V0i_1_Vec);
        V0i_1_Vec = _mm256_add_pd(_mm256_mul_pd(f0i_Vec, q2r_Vec), V0i_1_Vec);
        VXi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fXr_Vec, q2i_Vec), VXi_1_Vec);
        VXi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fXi_Vec, q2r_Vec), VXi_1_Vec);
        VYi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fYr_Vec, q2i_Vec), VYi_1_Vec);
        VYi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fYi_Vec, q2r_Vec), VYi_1_Vec);
        VZi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fZr_Vec, q2i_Vec), VZi_1_Vec);
        VZi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fZi_Vec, q2r_Vec), VZi_1_Vec);

        //V2 contribution
        V0r_2_Vec = _mm256_sub_pd(_mm256_mul_pd(f0i_Vec, q1i_Vec), V0r_2_Vec);
        V0r_2_Vec = _mm256_sub_pd(_mm256_mul_pd(f0r_Vec, q1r_Vec), V0r_2_Vec);
        VXr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec, q1r_Vec), VXr_2_Vec);
        VXr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q1i_Vec), VXr_2_Vec);
        VYr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec, q1r_Vec), VYr_2_Vec);
        VYr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q1i_Vec), VYr_2_Vec);
        VZr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec, q1r_Vec), VZr_2_Vec);
        VZr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q1i_Vec), VZr_2_Vec);

        V0i_2_Vec = _mm256_add_pd(_mm256_mul_pd(f0r_Vec,  q1i_Vec), V0i_2_Vec);
        V0i_2_Vec = _mm256_add_pd(_mm256_mul_pd(f0i_Vec,  q1r_Vec), V0i_2_Vec);
        VXi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec,  q1i_Vec), VXi_2_Vec);
        VXi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q1Nr_Vec), VXi_2_Vec);
        VYi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec,  q1i_Vec), VYi_2_Vec);
        VYi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q1Nr_Vec), VYi_2_Vec);
        VZi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec,  q1i_Vec), VZi_2_Vec);
        VZi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q1Nr_Vec), VZi_2_Vec);
      }
      #endif


      _mm256_store_pd (&V2_4x[index32j   ], V0r_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+ 4], V0i_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+ 8], VXr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+12], VXi_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+16], VYr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+20], VYi_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+24], VZr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+28], VZi_2_Vec);


      index20j += 20;
      index32j += 32;
    }

    //accumulate and store V1 contributions
    double sum_r[4];
    double sum_i[4];
    NPME_mm256_4x4HorizontalSum_pd (sum_r, V0r_1_Vec, VXr_1_Vec, 
                                       VYr_1_Vec, VZr_1_Vec);
    NPME_mm256_4x4HorizontalSum_pd (sum_i, V0i_1_Vec, VXi_1_Vec, 
                                       VYi_1_Vec, VZi_1_Vec);

    if (zeroArray)
    {
      V1[4*i  ]  = sum_r[0] + I*sum_i[0];
      V1[4*i+1]  = sum_r[1] + I*sum_i[1];
      V1[4*i+2]  = sum_r[2] + I*sum_i[2];
      V1[4*i+3]  = sum_r[3] + I*sum_i[3];
    }
    else
    {
      V1[4*i  ]  += sum_r[0] + I*sum_i[0];
      V1[4*i+1]  += sum_r[1] + I*sum_i[1];
      V1[4*i+2]  += sum_r[2] + I*sum_i[2];
      V1[4*i+3]  += sum_r[3] + I*sum_i[3];
    }

  }

  if (zeroArray)
    NPME_TransformComplexV1_4x_2_V1_AVX (nCharge2, V2_4x, &V2[0]);
  else
    NPME_TransformUpdateComplexV1_4x_2_V1_AVX (nCharge2, V2_4x, &V2[0]);
}



void NPME_PotHelmholtz_SR_DM_Self_V1_AVX (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir, const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotHelmholtz_Self_V1_AVX.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  double qCoCrd_4x[5*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  NPME_TransformQcomplexCoord_4x_AVX (nCharge, q, coord, qCoCrd_4x);

  double V2_4x[8*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  {
    const size_t remain    = nCharge%4;
    const size_t nLoop     = (nCharge - remain)/4;
    size_t nLoopwRemainder = nLoop;
    if (remain > 0)
      nLoopwRemainder++;

    memset(V2_4x, 0, 32*nLoopwRemainder*sizeof(double));
  }


  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r1        = &coord[3*i];
    __m256d q1r_Vec         = _mm256_set1_pd(creal( q[i]));
    __m256d q1i_Vec         = _mm256_set1_pd(cimag( q[i]));
    __m256d q1Nr_Vec        = _mm256_set1_pd(creal(-q[i]));

    const __m256d x1Vec     = _mm256_set1_pd(r1[0]);
    const __m256d y1Vec     = _mm256_set1_pd(r1[1]);
    const __m256d z1Vec     = _mm256_set1_pd(r1[2]);

    __m256d V0r_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VXr_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VYr_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VZr_1_Vec       = _mm256_set1_pd(0.0);

    __m256d V0i_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VXi_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VYi_1_Vec       = _mm256_set1_pd(0.0);
    __m256d VZi_1_Vec       = _mm256_set1_pd(0.0);

    const size_t remain2  = (i)%4;
    const size_t nLoop2   = (i-remain2)/4;

    size_t index20j = 0;
    size_t index32j = 0;
    for (size_t j = 0; j < nLoop2; j++)
    {
      const __m256d q2r_Vec = _mm256_load_pd (&qCoCrd_4x[index20j   ]);
      const __m256d q2i_Vec = _mm256_load_pd (&qCoCrd_4x[index20j+ 4]);

      __m256d f0r_Vec, fXr_Vec, fYr_Vec, fZr_Vec;
      __m256d f0i_Vec, fXi_Vec, fYi_Vec, fZi_Vec;
      //calc f0, fX, fY, fZ
      {
        __m256d xVec        = _mm256_load_pd (&qCoCrd_4x[index20j+ 8]);
        __m256d yVec        = _mm256_load_pd (&qCoCrd_4x[index20j+12]);
        __m256d zVec        = _mm256_load_pd (&qCoCrd_4x[index20j+16]);

        xVec = _mm256_sub_pd (x1Vec, xVec);
        yVec = _mm256_sub_pd (y1Vec, yVec);
        zVec = _mm256_sub_pd (z1Vec, zVec);

        NPME_PotHelmholtz_SR_DM_AVX (k0,
          f0r_Vec, f0i_Vec, fXr_Vec, fXi_Vec,
          fYr_Vec, fYi_Vec, fZr_Vec, fZi_Vec,
          xVec, yVec, zVec, Nder, a, b, Rdir);
      }

      //V   = f*q
      //V_r = f_r*q_r - f_i*q_i
      //V_i = f_r*q_i + f_i*q_r
  
      __m256d V0r_2_Vec = _mm256_load_pd (&V2_4x[index32j   ]);
      __m256d V0i_2_Vec = _mm256_load_pd (&V2_4x[index32j+ 4]);
      __m256d VXr_2_Vec = _mm256_load_pd (&V2_4x[index32j+ 8]);
      __m256d VXi_2_Vec = _mm256_load_pd (&V2_4x[index32j+12]);
      __m256d VYr_2_Vec = _mm256_load_pd (&V2_4x[index32j+16]);
      __m256d VYi_2_Vec = _mm256_load_pd (&V2_4x[index32j+20]);
      __m256d VZr_2_Vec = _mm256_load_pd (&V2_4x[index32j+24]);
      __m256d VZi_2_Vec = _mm256_load_pd (&V2_4x[index32j+28]);


      //V   = f*q
      //V_r = f_r*q_r - f_i*q_i
      //V_i = f_r*q_i + f_i*q_r
  
      #if NPME_USE_AVX_FMA
      {
        //V1 contribution
        V0r_1_Vec  = _mm256_fmsub_pd  (f0i_Vec, q2i_Vec, V0r_1_Vec);
        V0r_1_Vec  = _mm256_fmsub_pd  (f0r_Vec, q2r_Vec, V0r_1_Vec);
        VXr_1_Vec  = _mm256_fmsub_pd  (fXi_Vec, q2i_Vec, VXr_1_Vec);
        VXr_1_Vec  = _mm256_fmsub_pd  (fXr_Vec, q2r_Vec, VXr_1_Vec);
        VYr_1_Vec  = _mm256_fmsub_pd  (fYi_Vec, q2i_Vec, VYr_1_Vec);
        VYr_1_Vec  = _mm256_fmsub_pd  (fYr_Vec, q2r_Vec, VYr_1_Vec);
        VZr_1_Vec  = _mm256_fmsub_pd  (fZi_Vec, q2i_Vec, VZr_1_Vec);
        VZr_1_Vec  = _mm256_fmsub_pd  (fZr_Vec, q2r_Vec, VZr_1_Vec);

        V0i_1_Vec  = _mm256_fmadd_pd  (f0r_Vec, q2i_Vec, V0i_1_Vec);
        V0i_1_Vec  = _mm256_fmadd_pd  (f0i_Vec, q2r_Vec, V0i_1_Vec);
        VXi_1_Vec  = _mm256_fmadd_pd  (fXr_Vec, q2i_Vec, VXi_1_Vec);
        VXi_1_Vec  = _mm256_fmadd_pd  (fXi_Vec, q2r_Vec, VXi_1_Vec);
        VYi_1_Vec  = _mm256_fmadd_pd  (fYr_Vec, q2i_Vec, VYi_1_Vec);
        VYi_1_Vec  = _mm256_fmadd_pd  (fYi_Vec, q2r_Vec, VYi_1_Vec);
        VZi_1_Vec  = _mm256_fmadd_pd  (fZr_Vec, q2i_Vec, VZi_1_Vec);
        VZi_1_Vec  = _mm256_fmadd_pd  (fZi_Vec, q2r_Vec, VZi_1_Vec);

        //V2 contribution
        V0r_2_Vec  = _mm256_fmsub_pd  (f0i_Vec, q1i_Vec, V0r_2_Vec);
        V0r_2_Vec  = _mm256_fmsub_pd  (f0r_Vec, q1r_Vec, V0r_2_Vec);
        VXr_2_Vec  = _mm256_fmsub_pd  (fXr_Vec, q1r_Vec, VXr_2_Vec);
        VXr_2_Vec  = _mm256_fmsub_pd  (fXi_Vec, q1i_Vec, VXr_2_Vec);
        VYr_2_Vec  = _mm256_fmsub_pd  (fYr_Vec, q1r_Vec, VYr_2_Vec);
        VYr_2_Vec  = _mm256_fmsub_pd  (fYi_Vec, q1i_Vec, VYr_2_Vec);
        VZr_2_Vec  = _mm256_fmsub_pd  (fZr_Vec, q1r_Vec, VZr_2_Vec);
        VZr_2_Vec  = _mm256_fmsub_pd  (fZi_Vec, q1i_Vec, VZr_2_Vec);

        V0i_2_Vec  = _mm256_fmadd_pd  (f0r_Vec, q1i_Vec, V0i_2_Vec);
        V0i_2_Vec  = _mm256_fmadd_pd  (f0i_Vec, q1r_Vec, V0i_2_Vec);
        VXi_2_Vec  = _mm256_fmsub_pd  (fXr_Vec,  q1i_Vec, VXi_2_Vec);
        VXi_2_Vec  = _mm256_fmsub_pd  (fXi_Vec, q1Nr_Vec, VXi_2_Vec);
        VYi_2_Vec  = _mm256_fmsub_pd  (fYr_Vec,  q1i_Vec, VYi_2_Vec);
        VYi_2_Vec  = _mm256_fmsub_pd  (fYi_Vec, q1Nr_Vec, VYi_2_Vec);
        VZi_2_Vec  = _mm256_fmsub_pd  (fZr_Vec,  q1i_Vec, VZi_2_Vec);
        VZi_2_Vec  = _mm256_fmsub_pd  (fZi_Vec, q1Nr_Vec, VZi_2_Vec);
      }
      #else
      {
        //V1 contribution
        V0r_1_Vec = _mm256_sub_pd(_mm256_mul_pd(f0i_Vec, q2i_Vec), V0r_1_Vec);
        V0r_1_Vec = _mm256_sub_pd(_mm256_mul_pd(f0r_Vec, q2r_Vec), V0r_1_Vec);
        VXr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q2i_Vec), VXr_1_Vec);
        VXr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec, q2r_Vec), VXr_1_Vec);
        VYr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q2i_Vec), VYr_1_Vec);
        VYr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec, q2r_Vec), VYr_1_Vec);
        VZr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q2i_Vec), VZr_1_Vec);
        VZr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec, q2r_Vec), VZr_1_Vec);

        V0i_1_Vec = _mm256_add_pd(_mm256_mul_pd(f0r_Vec, q2i_Vec), V0i_1_Vec);
        V0i_1_Vec = _mm256_add_pd(_mm256_mul_pd(f0i_Vec, q2r_Vec), V0i_1_Vec);
        VXi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fXr_Vec, q2i_Vec), VXi_1_Vec);
        VXi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fXi_Vec, q2r_Vec), VXi_1_Vec);
        VYi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fYr_Vec, q2i_Vec), VYi_1_Vec);
        VYi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fYi_Vec, q2r_Vec), VYi_1_Vec);
        VZi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fZr_Vec, q2i_Vec), VZi_1_Vec);
        VZi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fZi_Vec, q2r_Vec), VZi_1_Vec);

        //V2 contribution
        V0r_2_Vec = _mm256_sub_pd(_mm256_mul_pd(f0i_Vec, q1i_Vec), V0r_2_Vec);
        V0r_2_Vec = _mm256_sub_pd(_mm256_mul_pd(f0r_Vec, q1r_Vec), V0r_2_Vec);
        VXr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec, q1r_Vec), VXr_2_Vec);
        VXr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q1i_Vec), VXr_2_Vec);
        VYr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec, q1r_Vec), VYr_2_Vec);
        VYr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q1i_Vec), VYr_2_Vec);
        VZr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec, q1r_Vec), VZr_2_Vec);
        VZr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q1i_Vec), VZr_2_Vec);

        V0i_2_Vec = _mm256_add_pd(_mm256_mul_pd(f0r_Vec,  q1i_Vec), V0i_2_Vec);
        V0i_2_Vec = _mm256_add_pd(_mm256_mul_pd(f0i_Vec,  q1r_Vec), V0i_2_Vec);
        VXi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec,  q1i_Vec), VXi_2_Vec);
        VXi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q1Nr_Vec), VXi_2_Vec);
        VYi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec,  q1i_Vec), VYi_2_Vec);
        VYi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q1Nr_Vec), VYi_2_Vec);
        VZi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec,  q1i_Vec), VZi_2_Vec);
        VZi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q1Nr_Vec), VZi_2_Vec);
      }
      #endif

      _mm256_store_pd (&V2_4x[index32j   ], V0r_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+ 4], V0i_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+ 8], VXr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+12], VXi_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+16], VYr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+20], VYi_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+24], VZr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+28], VZi_2_Vec);

      index20j += 20;
      index32j += 32;
    }

    if (remain2 > 0)
    {
      const size_t indexStart     = 4*nLoop2;
      const _Complex double *qLoc = &q[indexStart];
      const double *crdLoc        = &coord[3*indexStart];
      const double X              = r1[0]+1.0;
      double x2Array[4]  __attribute__((aligned(64))) = {X, X, X, X};
      double y2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double z2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double q2rArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};
      double q2iArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};
      double mskArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};


      for (size_t k = 0; k < remain2; k++)
      {
        q2rArray[k] = creal(qLoc[k]);
        q2iArray[k] = cimag(qLoc[k]);
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
        mskArray[k] = 1.0;
      }
      __m256d q2r_Vec   = _mm256_load_pd (q2rArray);
      __m256d q2i_Vec   = _mm256_load_pd (q2iArray);
      __m256d xVec      = _mm256_load_pd (x2Array);
      __m256d yVec      = _mm256_load_pd (y2Array);
      __m256d zVec      = _mm256_load_pd (z2Array);
      __m256d mskVec    = _mm256_load_pd (mskArray);

      //apply mask to q1
      q1r_Vec   = _mm256_mul_pd (mskVec, q1r_Vec);
      q1i_Vec   = _mm256_mul_pd (mskVec, q1i_Vec);
      q1Nr_Vec  = _mm256_mul_pd (mskVec, q1Nr_Vec);

      __m256d V0r_2_Vec = _mm256_load_pd (&V2_4x[index32j   ]);
      __m256d V0i_2_Vec = _mm256_load_pd (&V2_4x[index32j+ 4]);
      __m256d VXr_2_Vec = _mm256_load_pd (&V2_4x[index32j+ 8]);
      __m256d VXi_2_Vec = _mm256_load_pd (&V2_4x[index32j+12]);
      __m256d VYr_2_Vec = _mm256_load_pd (&V2_4x[index32j+16]);
      __m256d VYi_2_Vec = _mm256_load_pd (&V2_4x[index32j+20]);
      __m256d VZr_2_Vec = _mm256_load_pd (&V2_4x[index32j+24]);
      __m256d VZi_2_Vec = _mm256_load_pd (&V2_4x[index32j+28]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      __m256d f0r_Vec, fXr_Vec, fYr_Vec, fZr_Vec;
      __m256d f0i_Vec, fXi_Vec, fYi_Vec, fZi_Vec;
      //calc f0, fX, fY, fZ
      {
        NPME_PotHelmholtz_SR_DM_AVX (k0,
          f0r_Vec, f0i_Vec, fXr_Vec, fXi_Vec,
          fYr_Vec, fYi_Vec, fZr_Vec, fZi_Vec,
          xVec, yVec, zVec, Nder, a, b, Rdir);
      }
      
      #if NPME_USE_AVX_FMA
      {
        //V1 contribution
        V0r_1_Vec  = _mm256_fmsub_pd  (f0i_Vec, q2i_Vec, V0r_1_Vec);
        V0r_1_Vec  = _mm256_fmsub_pd  (f0r_Vec, q2r_Vec, V0r_1_Vec);
        VXr_1_Vec  = _mm256_fmsub_pd  (fXi_Vec, q2i_Vec, VXr_1_Vec);
        VXr_1_Vec  = _mm256_fmsub_pd  (fXr_Vec, q2r_Vec, VXr_1_Vec);
        VYr_1_Vec  = _mm256_fmsub_pd  (fYi_Vec, q2i_Vec, VYr_1_Vec);
        VYr_1_Vec  = _mm256_fmsub_pd  (fYr_Vec, q2r_Vec, VYr_1_Vec);
        VZr_1_Vec  = _mm256_fmsub_pd  (fZi_Vec, q2i_Vec, VZr_1_Vec);
        VZr_1_Vec  = _mm256_fmsub_pd  (fZr_Vec, q2r_Vec, VZr_1_Vec);

        V0i_1_Vec  = _mm256_fmadd_pd  (f0r_Vec, q2i_Vec, V0i_1_Vec);
        V0i_1_Vec  = _mm256_fmadd_pd  (f0i_Vec, q2r_Vec, V0i_1_Vec);
        VXi_1_Vec  = _mm256_fmadd_pd  (fXr_Vec, q2i_Vec, VXi_1_Vec);
        VXi_1_Vec  = _mm256_fmadd_pd  (fXi_Vec, q2r_Vec, VXi_1_Vec);
        VYi_1_Vec  = _mm256_fmadd_pd  (fYr_Vec, q2i_Vec, VYi_1_Vec);
        VYi_1_Vec  = _mm256_fmadd_pd  (fYi_Vec, q2r_Vec, VYi_1_Vec);
        VZi_1_Vec  = _mm256_fmadd_pd  (fZr_Vec, q2i_Vec, VZi_1_Vec);
        VZi_1_Vec  = _mm256_fmadd_pd  (fZi_Vec, q2r_Vec, VZi_1_Vec);

        //V2 contribution
        V0r_2_Vec  = _mm256_fmsub_pd  (f0i_Vec, q1i_Vec, V0r_2_Vec);
        V0r_2_Vec  = _mm256_fmsub_pd  (f0r_Vec, q1r_Vec, V0r_2_Vec);
        VXr_2_Vec  = _mm256_fmsub_pd  (fXr_Vec, q1r_Vec, VXr_2_Vec);
        VXr_2_Vec  = _mm256_fmsub_pd  (fXi_Vec, q1i_Vec, VXr_2_Vec);
        VYr_2_Vec  = _mm256_fmsub_pd  (fYr_Vec, q1r_Vec, VYr_2_Vec);
        VYr_2_Vec  = _mm256_fmsub_pd  (fYi_Vec, q1i_Vec, VYr_2_Vec);
        VZr_2_Vec  = _mm256_fmsub_pd  (fZr_Vec, q1r_Vec, VZr_2_Vec);
        VZr_2_Vec  = _mm256_fmsub_pd  (fZi_Vec, q1i_Vec, VZr_2_Vec);

        V0i_2_Vec  = _mm256_fmadd_pd  (f0r_Vec, q1i_Vec, V0i_2_Vec);
        V0i_2_Vec  = _mm256_fmadd_pd  (f0i_Vec, q1r_Vec, V0i_2_Vec);
        VXi_2_Vec  = _mm256_fmsub_pd  (fXr_Vec,  q1i_Vec, VXi_2_Vec);
        VXi_2_Vec  = _mm256_fmsub_pd  (fXi_Vec, q1Nr_Vec, VXi_2_Vec);
        VYi_2_Vec  = _mm256_fmsub_pd  (fYr_Vec,  q1i_Vec, VYi_2_Vec);
        VYi_2_Vec  = _mm256_fmsub_pd  (fYi_Vec, q1Nr_Vec, VYi_2_Vec);
        VZi_2_Vec  = _mm256_fmsub_pd  (fZr_Vec,  q1i_Vec, VZi_2_Vec);
        VZi_2_Vec  = _mm256_fmsub_pd  (fZi_Vec, q1Nr_Vec, VZi_2_Vec);
      }
      #else
      {
        //V1 contribution
        V0r_1_Vec = _mm256_sub_pd(_mm256_mul_pd(f0i_Vec, q2i_Vec), V0r_1_Vec);
        V0r_1_Vec = _mm256_sub_pd(_mm256_mul_pd(f0r_Vec, q2r_Vec), V0r_1_Vec);
        VXr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q2i_Vec), VXr_1_Vec);
        VXr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec, q2r_Vec), VXr_1_Vec);
        VYr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q2i_Vec), VYr_1_Vec);
        VYr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec, q2r_Vec), VYr_1_Vec);
        VZr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q2i_Vec), VZr_1_Vec);
        VZr_1_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec, q2r_Vec), VZr_1_Vec);

        V0i_1_Vec = _mm256_add_pd(_mm256_mul_pd(f0r_Vec, q2i_Vec), V0i_1_Vec);
        V0i_1_Vec = _mm256_add_pd(_mm256_mul_pd(f0i_Vec, q2r_Vec), V0i_1_Vec);
        VXi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fXr_Vec, q2i_Vec), VXi_1_Vec);
        VXi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fXi_Vec, q2r_Vec), VXi_1_Vec);
        VYi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fYr_Vec, q2i_Vec), VYi_1_Vec);
        VYi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fYi_Vec, q2r_Vec), VYi_1_Vec);
        VZi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fZr_Vec, q2i_Vec), VZi_1_Vec);
        VZi_1_Vec = _mm256_add_pd(_mm256_mul_pd(fZi_Vec, q2r_Vec), VZi_1_Vec);

        //V2 contribution
        V0r_2_Vec = _mm256_sub_pd(_mm256_mul_pd(f0i_Vec, q1i_Vec), V0r_2_Vec);
        V0r_2_Vec = _mm256_sub_pd(_mm256_mul_pd(f0r_Vec, q1r_Vec), V0r_2_Vec);
        VXr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec, q1r_Vec), VXr_2_Vec);
        VXr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q1i_Vec), VXr_2_Vec);
        VYr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec, q1r_Vec), VYr_2_Vec);
        VYr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q1i_Vec), VYr_2_Vec);
        VZr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec, q1r_Vec), VZr_2_Vec);
        VZr_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q1i_Vec), VZr_2_Vec);

        V0i_2_Vec = _mm256_add_pd(_mm256_mul_pd(f0r_Vec,  q1i_Vec), V0i_2_Vec);
        V0i_2_Vec = _mm256_add_pd(_mm256_mul_pd(f0i_Vec,  q1r_Vec), V0i_2_Vec);
        VXi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXr_Vec,  q1i_Vec), VXi_2_Vec);
        VXi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fXi_Vec, q1Nr_Vec), VXi_2_Vec);
        VYi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYr_Vec,  q1i_Vec), VYi_2_Vec);
        VYi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fYi_Vec, q1Nr_Vec), VYi_2_Vec);
        VZi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZr_Vec,  q1i_Vec), VZi_2_Vec);
        VZi_2_Vec = _mm256_sub_pd(_mm256_mul_pd(fZi_Vec, q1Nr_Vec), VZi_2_Vec);
      }
      #endif

      _mm256_store_pd (&V2_4x[index32j   ], V0r_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+ 4], V0i_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+ 8], VXr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+12], VXi_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+16], VYr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+20], VYi_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+24], VZr_2_Vec);
      _mm256_store_pd (&V2_4x[index32j+28], VZi_2_Vec);
    }


    //accumulate and store V1 contributions
    double sum_r[4];
    double sum_i[4];
    NPME_mm256_4x4HorizontalSum_pd (sum_r, V0r_1_Vec, VXr_1_Vec, 
                                       VYr_1_Vec, VZr_1_Vec);
    NPME_mm256_4x4HorizontalSum_pd (sum_i, V0i_1_Vec, VXi_1_Vec, 
                                       VYi_1_Vec, VZi_1_Vec);

    if (zeroArray)
    {
      V[4*i  ]  = sum_r[0] + I*sum_i[0];
      V[4*i+1]  = sum_r[1] + I*sum_i[1];
      V[4*i+2]  = sum_r[2] + I*sum_i[2];
      V[4*i+3]  = sum_r[3] + I*sum_i[3];
    }
    else
    {
      V[4*i  ]  += sum_r[0] + I*sum_i[0];
      V[4*i+1]  += sum_r[1] + I*sum_i[1];
      V[4*i+2]  += sum_r[2] + I*sum_i[2];
      V[4*i+3]  += sum_r[3] + I*sum_i[3];
    }
  }

  NPME_TransformUpdateComplexV1_4x_2_V1_AVX (nCharge, V2_4x, V);
}





#endif


#if NPME_USE_AVX_512


void NPME_PotHelmholtz_AVX_512 (const _Complex double k0,
  __m512d& f0r_Vec, __m512d& f0i_Vec, __m512d& fXr_Vec, __m512d& fXi_Vec,
  __m512d& fYr_Vec, __m512d& fYi_Vec, __m512d& fZr_Vec, __m512d& fZi_Vec,
  const __m512d& xVec, const __m512d& yVec, const __m512d& zVec)
{
  const __m512d k0rVec  = _mm512_set1_pd( creal(k0) );
  const __m512d nk0iVec = _mm512_set1_pd(-cimag(k0) );

  __m512d rVec;
  rVec  = _mm512_mul_pd  (xVec, xVec);
  rVec  = _mm512_fmadd_pd  (yVec, yVec, rVec);
  rVec  = _mm512_fmadd_pd  (zVec, zVec, rVec);
  rVec  = _mm512_sqrt_pd (rVec);

  __m512d f1r_Vec, f1i_Vec;
  {
    __m512d sinVec, cosVec, expVec;
    sinVec   = _mm512_sincos_pd (&cosVec, _mm512_mul_pd (k0rVec, rVec));
    expVec   = _mm512_exp_pd  (_mm512_mul_pd (nk0iVec, rVec));
    expVec   = _mm512_div_pd  (expVec, rVec);
    f0r_Vec  = _mm512_mul_pd  (expVec,    cosVec);
    f0i_Vec  = _mm512_mul_pd  (expVec,    sinVec);

    //g = f0/r
    __m512d gr_Vec = _mm512_div_pd  (f0r_Vec , rVec);
    __m512d gi_Vec = _mm512_div_pd  (f0i_Vec , rVec);

    //f1 = 1/r df0/dr
    f1r_Vec = _mm512_fmadd_pd  ( k0rVec, f0i_Vec,  gr_Vec);
    f1r_Vec = _mm512_fmsub_pd  (nk0iVec, f0r_Vec, f1r_Vec);
    f1i_Vec = _mm512_fmsub_pd  (nk0iVec, f0i_Vec,  gi_Vec);
    f1i_Vec = _mm512_fmadd_pd  ( k0rVec, f0r_Vec, f1i_Vec);

    f1r_Vec = _mm512_div_pd  (f1r_Vec, rVec);
    f1i_Vec = _mm512_div_pd  (f1i_Vec, rVec);
  }

  fXr_Vec = _mm512_mul_pd  (f1r_Vec, xVec);
  fYr_Vec = _mm512_mul_pd  (f1r_Vec, yVec);
  fZr_Vec = _mm512_mul_pd  (f1r_Vec, zVec);

  fXi_Vec = _mm512_mul_pd  (f1i_Vec, xVec);
  fYi_Vec = _mm512_mul_pd  (f1i_Vec, yVec);
  fZi_Vec = _mm512_mul_pd  (f1i_Vec, zVec);
}


void NPME_PotHelmholtz_Pair_V1_AVX_512 (const _Complex double k0,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotHelmholtz_Pair_V1_AVX_512.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  const size_t remain2    = nCharge2%8;
  const size_t nLoop2     = (nCharge2 - remain2)/8;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  double qCoCrd_8x[5*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_8x    [8*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));

  memset(V2_8x, 0, 64*nLoop2wRemainder*sizeof(double));
  NPME_TransformQcomplexCoord_8x_AVX_512 (nCharge2, q2, coord2, qCoCrd_8x);

  for (size_t i = 0; i < nCharge1; i++)
  {
    const double *r1        = &coord1[3*i];
    const __m512d q1r_Vec   = _mm512_set1_pd(creal( q1[i]));
    const __m512d q1i_Vec   = _mm512_set1_pd(cimag( q1[i]));
    const __m512d q1Nr_Vec  = _mm512_set1_pd(creal(-q1[i]));

    const __m512d x1Vec     = _mm512_set1_pd(r1[0]);
    const __m512d y1Vec     = _mm512_set1_pd(r1[1]);
    const __m512d z1Vec     = _mm512_set1_pd(r1[2]);

    __m512d V0r_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VXr_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VYr_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VZr_1_Vec       = _mm512_set1_pd(0.0);

    __m512d V0i_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VXi_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VYi_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VZi_1_Vec       = _mm512_set1_pd(0.0);

    size_t index40j = 0;
    size_t index64j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      const __m512d q2r_Vec = _mm512_load_pd (&qCoCrd_8x[index40j   ]);
      const __m512d q2i_Vec = _mm512_load_pd (&qCoCrd_8x[index40j+ 8]);

      __m512d f0r_Vec, fXr_Vec, fYr_Vec, fZr_Vec;
      __m512d f0i_Vec, fXi_Vec, fYi_Vec, fZi_Vec;
      //calc f0, fX, fY, fZ
      {
        __m512d xVec        = _mm512_load_pd (&qCoCrd_8x[index40j+16]);
        __m512d yVec        = _mm512_load_pd (&qCoCrd_8x[index40j+24]);
        __m512d zVec        = _mm512_load_pd (&qCoCrd_8x[index40j+32]);

        xVec = _mm512_sub_pd (x1Vec, xVec);
        yVec = _mm512_sub_pd (y1Vec, yVec);
        zVec = _mm512_sub_pd (z1Vec, zVec);

        NPME_PotHelmholtz_AVX_512 (k0,
          f0r_Vec, f0i_Vec, fXr_Vec, fXi_Vec,
          fYr_Vec, fYi_Vec, fZr_Vec, fZi_Vec,
          xVec, yVec, zVec);
      }

      //V   = f*q
      //V_r = f_r*q_r - f_i*q_i
      //V_i = f_r*q_i + f_i*q_r
        __m512d V0r_2_Vec = _mm512_load_pd (&V2_8x[index64j   ]);
      __m512d V0i_2_Vec = _mm512_load_pd (&V2_8x[index64j+ 8]);
      __m512d VXr_2_Vec = _mm512_load_pd (&V2_8x[index64j+16]);
      __m512d VXi_2_Vec = _mm512_load_pd (&V2_8x[index64j+24]);
      __m512d VYr_2_Vec = _mm512_load_pd (&V2_8x[index64j+32]);
      __m512d VYi_2_Vec = _mm512_load_pd (&V2_8x[index64j+40]);
      __m512d VZr_2_Vec = _mm512_load_pd (&V2_8x[index64j+48]);
      __m512d VZi_2_Vec = _mm512_load_pd (&V2_8x[index64j+56]);

      //V1 contribution
      V0r_1_Vec  = _mm512_fmsub_pd  (f0i_Vec, q2i_Vec, V0r_1_Vec);
      V0r_1_Vec  = _mm512_fmsub_pd  (f0r_Vec, q2r_Vec, V0r_1_Vec);
      VXr_1_Vec  = _mm512_fmsub_pd  (fXi_Vec, q2i_Vec, VXr_1_Vec);
      VXr_1_Vec  = _mm512_fmsub_pd  (fXr_Vec, q2r_Vec, VXr_1_Vec);
      VYr_1_Vec  = _mm512_fmsub_pd  (fYi_Vec, q2i_Vec, VYr_1_Vec);
      VYr_1_Vec  = _mm512_fmsub_pd  (fYr_Vec, q2r_Vec, VYr_1_Vec);
      VZr_1_Vec  = _mm512_fmsub_pd  (fZi_Vec, q2i_Vec, VZr_1_Vec);
      VZr_1_Vec  = _mm512_fmsub_pd  (fZr_Vec, q2r_Vec, VZr_1_Vec);

      V0i_1_Vec  = _mm512_fmadd_pd  (f0r_Vec, q2i_Vec, V0i_1_Vec);
      V0i_1_Vec  = _mm512_fmadd_pd  (f0i_Vec, q2r_Vec, V0i_1_Vec);
      VXi_1_Vec  = _mm512_fmadd_pd  (fXr_Vec, q2i_Vec, VXi_1_Vec);
      VXi_1_Vec  = _mm512_fmadd_pd  (fXi_Vec, q2r_Vec, VXi_1_Vec);
      VYi_1_Vec  = _mm512_fmadd_pd  (fYr_Vec, q2i_Vec, VYi_1_Vec);
      VYi_1_Vec  = _mm512_fmadd_pd  (fYi_Vec, q2r_Vec, VYi_1_Vec);
      VZi_1_Vec  = _mm512_fmadd_pd  (fZr_Vec, q2i_Vec, VZi_1_Vec);
      VZi_1_Vec  = _mm512_fmadd_pd  (fZi_Vec, q2r_Vec, VZi_1_Vec);

      //V2 contribution
      V0r_2_Vec  = _mm512_fmsub_pd  (f0i_Vec, q1i_Vec, V0r_2_Vec);
      V0r_2_Vec  = _mm512_fmsub_pd  (f0r_Vec, q1r_Vec, V0r_2_Vec);
      VXr_2_Vec  = _mm512_fmsub_pd  (fXr_Vec, q1r_Vec, VXr_2_Vec);
      VXr_2_Vec  = _mm512_fmsub_pd  (fXi_Vec, q1i_Vec, VXr_2_Vec);
      VYr_2_Vec  = _mm512_fmsub_pd  (fYr_Vec, q1r_Vec, VYr_2_Vec);
      VYr_2_Vec  = _mm512_fmsub_pd  (fYi_Vec, q1i_Vec, VYr_2_Vec);
      VZr_2_Vec  = _mm512_fmsub_pd  (fZr_Vec, q1r_Vec, VZr_2_Vec);
      VZr_2_Vec  = _mm512_fmsub_pd  (fZi_Vec, q1i_Vec, VZr_2_Vec);

      V0i_2_Vec  = _mm512_fmadd_pd  (f0r_Vec, q1i_Vec, V0i_2_Vec);
      V0i_2_Vec  = _mm512_fmadd_pd  (f0i_Vec, q1r_Vec, V0i_2_Vec);
      VXi_2_Vec  = _mm512_fmsub_pd  (fXr_Vec,  q1i_Vec, VXi_2_Vec);
      VXi_2_Vec  = _mm512_fmsub_pd  (fXi_Vec, q1Nr_Vec, VXi_2_Vec);
      VYi_2_Vec  = _mm512_fmsub_pd  (fYr_Vec,  q1i_Vec, VYi_2_Vec);
      VYi_2_Vec  = _mm512_fmsub_pd  (fYi_Vec, q1Nr_Vec, VYi_2_Vec);
      VZi_2_Vec  = _mm512_fmsub_pd  (fZr_Vec,  q1i_Vec, VZi_2_Vec);
      VZi_2_Vec  = _mm512_fmsub_pd  (fZi_Vec, q1Nr_Vec, VZi_2_Vec);

      _mm512_store_pd (&V2_8x[index64j   ], V0r_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+ 8], V0i_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+16], VXr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+24], VXi_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+32], VYr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+40], VYi_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+48], VZr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+56], VZi_2_Vec);

      index40j += 40;
      index64j += 64;
    }

    //accumulate and store V1 contributions
    double sum_r[4];
    double sum_i[4];
    NPME_mm512_4x8HorizontalSum_pd (sum_r, V0r_1_Vec, VXr_1_Vec, 
                                       VYr_1_Vec, VZr_1_Vec);
    NPME_mm512_4x8HorizontalSum_pd (sum_i, V0i_1_Vec, VXi_1_Vec, 
                                       VYi_1_Vec, VZi_1_Vec);

    if (zeroArray)
    {
      V1[4*i  ]  = sum_r[0] + I*sum_i[0];
      V1[4*i+1]  = sum_r[1] + I*sum_i[1];
      V1[4*i+2]  = sum_r[2] + I*sum_i[2];
      V1[4*i+3]  = sum_r[3] + I*sum_i[3];
    }
    else
    {
      V1[4*i  ]  += sum_r[0] + I*sum_i[0];
      V1[4*i+1]  += sum_r[1] + I*sum_i[1];
      V1[4*i+2]  += sum_r[2] + I*sum_i[2];
      V1[4*i+3]  += sum_r[3] + I*sum_i[3];
    }
  }

  if (zeroArray)
    NPME_TransformComplexV1_8x_2_V1_AVX_512 (nCharge2, V2_8x, &V2[0]);
  else
    NPME_TransformUpdateComplexV1_8x_2_V1_AVX_512 (nCharge2, V2_8x, &V2[0]);
}


void NPME_PotHelmholtz_Self_V1_AVX_512 (const _Complex double k0,
  const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotHelmholtz_Self_V1_AVX_512.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  double qCoCrd_8x[5*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  NPME_TransformQcomplexCoord_8x_AVX_512 (nCharge, q, coord, qCoCrd_8x);

  double V2_8x[8*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  {
    const size_t remain    = nCharge%8;
    const size_t nLoop     = (nCharge - remain)/8;
    size_t nLoopwRemainder = nLoop;
    if (remain > 0)
      nLoopwRemainder++;

    memset(V2_8x, 0, 64*nLoopwRemainder*sizeof(double));
  }

  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r1        = &coord[3*i];
    __m512d q1r_Vec         = _mm512_set1_pd(creal( q[i]));
    __m512d q1i_Vec         = _mm512_set1_pd(cimag( q[i]));
    __m512d q1Nr_Vec        = _mm512_set1_pd(creal(-q[i]));

    const __m512d x1Vec     = _mm512_set1_pd(r1[0]);
    const __m512d y1Vec     = _mm512_set1_pd(r1[1]);
    const __m512d z1Vec     = _mm512_set1_pd(r1[2]);

    __m512d V0r_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VXr_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VYr_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VZr_1_Vec       = _mm512_set1_pd(0.0);

    __m512d V0i_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VXi_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VYi_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VZi_1_Vec       = _mm512_set1_pd(0.0);

    const size_t remain2  = (i)%8;
    const size_t nLoop2   = (i-remain2)/8;

    size_t index40j = 0;
    size_t index64j = 0;
    for (size_t j = 0; j < nLoop2; j++)
    {
      const __m512d q2r_Vec = _mm512_load_pd (&qCoCrd_8x[index40j   ]);
      const __m512d q2i_Vec = _mm512_load_pd (&qCoCrd_8x[index40j+ 8]);

      __m512d f0r_Vec, fXr_Vec, fYr_Vec, fZr_Vec;
      __m512d f0i_Vec, fXi_Vec, fYi_Vec, fZi_Vec;
      //calc f0, fX, fY, fZ
      {
        __m512d xVec        = _mm512_load_pd (&qCoCrd_8x[index40j+16]);
        __m512d yVec        = _mm512_load_pd (&qCoCrd_8x[index40j+24]);
        __m512d zVec        = _mm512_load_pd (&qCoCrd_8x[index40j+32]);

        xVec = _mm512_sub_pd (x1Vec, xVec);
        yVec = _mm512_sub_pd (y1Vec, yVec);
        zVec = _mm512_sub_pd (z1Vec, zVec);

        NPME_PotHelmholtz_AVX_512 (k0,
          f0r_Vec, f0i_Vec, fXr_Vec, fXi_Vec,
          fYr_Vec, fYi_Vec, fZr_Vec, fZi_Vec,
          xVec, yVec, zVec);
      }

      //V   = f*q
      //V_r = f_r*q_r - f_i*q_i
      //V_i = f_r*q_i + f_i*q_r
  
      __m512d V0r_2_Vec = _mm512_load_pd (&V2_8x[index64j   ]);
      __m512d V0i_2_Vec = _mm512_load_pd (&V2_8x[index64j+ 8]);
      __m512d VXr_2_Vec = _mm512_load_pd (&V2_8x[index64j+16]);
      __m512d VXi_2_Vec = _mm512_load_pd (&V2_8x[index64j+24]);
      __m512d VYr_2_Vec = _mm512_load_pd (&V2_8x[index64j+32]);
      __m512d VYi_2_Vec = _mm512_load_pd (&V2_8x[index64j+40]);
      __m512d VZr_2_Vec = _mm512_load_pd (&V2_8x[index64j+48]);
      __m512d VZi_2_Vec = _mm512_load_pd (&V2_8x[index64j+56]);


      //V   = f*q
      //V_r = f_r*q_r - f_i*q_i
      //V_i = f_r*q_i + f_i*q_r
      //V1 contribution
      V0r_1_Vec  = _mm512_fmsub_pd  (f0i_Vec, q2i_Vec, V0r_1_Vec);
      V0r_1_Vec  = _mm512_fmsub_pd  (f0r_Vec, q2r_Vec, V0r_1_Vec);
      VXr_1_Vec  = _mm512_fmsub_pd  (fXi_Vec, q2i_Vec, VXr_1_Vec);
      VXr_1_Vec  = _mm512_fmsub_pd  (fXr_Vec, q2r_Vec, VXr_1_Vec);
      VYr_1_Vec  = _mm512_fmsub_pd  (fYi_Vec, q2i_Vec, VYr_1_Vec);
      VYr_1_Vec  = _mm512_fmsub_pd  (fYr_Vec, q2r_Vec, VYr_1_Vec);
      VZr_1_Vec  = _mm512_fmsub_pd  (fZi_Vec, q2i_Vec, VZr_1_Vec);
      VZr_1_Vec  = _mm512_fmsub_pd  (fZr_Vec, q2r_Vec, VZr_1_Vec);

      V0i_1_Vec  = _mm512_fmadd_pd  (f0r_Vec, q2i_Vec, V0i_1_Vec);
      V0i_1_Vec  = _mm512_fmadd_pd  (f0i_Vec, q2r_Vec, V0i_1_Vec);
      VXi_1_Vec  = _mm512_fmadd_pd  (fXr_Vec, q2i_Vec, VXi_1_Vec);
      VXi_1_Vec  = _mm512_fmadd_pd  (fXi_Vec, q2r_Vec, VXi_1_Vec);
      VYi_1_Vec  = _mm512_fmadd_pd  (fYr_Vec, q2i_Vec, VYi_1_Vec);
      VYi_1_Vec  = _mm512_fmadd_pd  (fYi_Vec, q2r_Vec, VYi_1_Vec);
      VZi_1_Vec  = _mm512_fmadd_pd  (fZr_Vec, q2i_Vec, VZi_1_Vec);
      VZi_1_Vec  = _mm512_fmadd_pd  (fZi_Vec, q2r_Vec, VZi_1_Vec);

      //V2 contribution
      V0r_2_Vec  = _mm512_fmsub_pd  (f0i_Vec, q1i_Vec, V0r_2_Vec);
      V0r_2_Vec  = _mm512_fmsub_pd  (f0r_Vec, q1r_Vec, V0r_2_Vec);
      VXr_2_Vec  = _mm512_fmsub_pd  (fXr_Vec, q1r_Vec, VXr_2_Vec);
      VXr_2_Vec  = _mm512_fmsub_pd  (fXi_Vec, q1i_Vec, VXr_2_Vec);
      VYr_2_Vec  = _mm512_fmsub_pd  (fYr_Vec, q1r_Vec, VYr_2_Vec);
      VYr_2_Vec  = _mm512_fmsub_pd  (fYi_Vec, q1i_Vec, VYr_2_Vec);
      VZr_2_Vec  = _mm512_fmsub_pd  (fZr_Vec, q1r_Vec, VZr_2_Vec);
      VZr_2_Vec  = _mm512_fmsub_pd  (fZi_Vec, q1i_Vec, VZr_2_Vec);

      V0i_2_Vec  = _mm512_fmadd_pd  (f0r_Vec, q1i_Vec, V0i_2_Vec);
      V0i_2_Vec  = _mm512_fmadd_pd  (f0i_Vec, q1r_Vec, V0i_2_Vec);
      VXi_2_Vec  = _mm512_fmsub_pd  (fXr_Vec,  q1i_Vec, VXi_2_Vec);
      VXi_2_Vec  = _mm512_fmsub_pd  (fXi_Vec, q1Nr_Vec, VXi_2_Vec);
      VYi_2_Vec  = _mm512_fmsub_pd  (fYr_Vec,  q1i_Vec, VYi_2_Vec);
      VYi_2_Vec  = _mm512_fmsub_pd  (fYi_Vec, q1Nr_Vec, VYi_2_Vec);
      VZi_2_Vec  = _mm512_fmsub_pd  (fZr_Vec,  q1i_Vec, VZi_2_Vec);
      VZi_2_Vec  = _mm512_fmsub_pd  (fZi_Vec, q1Nr_Vec, VZi_2_Vec);

      _mm512_store_pd (&V2_8x[index64j   ], V0r_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+ 8], V0i_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+16], VXr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+24], VXi_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+32], VYr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+40], VYi_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+48], VZr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+56], VZi_2_Vec);

      index40j += 40;
      index64j += 64;
    }

    if (remain2 > 0)
    {
      const size_t indexStart     = 8*nLoop2;
      const _Complex double *qLoc = &q[indexStart];
      const double *crdLoc        = &coord[3*indexStart];
      const double X              = r1[0]+1;
      double x2Array[8]  __attribute__((aligned(64))) = {X,X,X,X,X,X,X,X};
      double y2Array[8]  __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};
      double z2Array[8]  __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};
      double q2rArray[8] __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};
      double q2iArray[8] __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};
      double mskArray[8] __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};

      for (size_t k = 0; k < remain2; k++)
      {
        q2rArray[k] = creal(qLoc[k]);
        q2iArray[k] = cimag(qLoc[k]);
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
        mskArray[k] = 1.0;
      }
      __m512d q2r_Vec   = _mm512_load_pd (q2rArray);
      __m512d q2i_Vec   = _mm512_load_pd (q2iArray);
      __m512d xVec      = _mm512_load_pd (x2Array);
      __m512d yVec      = _mm512_load_pd (y2Array);
      __m512d zVec      = _mm512_load_pd (z2Array);
      __m512d mskVec    = _mm512_load_pd (mskArray);

      //apply mask to q1
      q1r_Vec   = _mm512_mul_pd (mskVec, q1r_Vec);
      q1i_Vec   = _mm512_mul_pd (mskVec, q1i_Vec);
      q1Nr_Vec  = _mm512_mul_pd (mskVec, q1Nr_Vec);

      __m512d V0r_2_Vec = _mm512_load_pd (&V2_8x[index64j   ]);
      __m512d V0i_2_Vec = _mm512_load_pd (&V2_8x[index64j+ 8]);
      __m512d VXr_2_Vec = _mm512_load_pd (&V2_8x[index64j+16]);
      __m512d VXi_2_Vec = _mm512_load_pd (&V2_8x[index64j+24]);
      __m512d VYr_2_Vec = _mm512_load_pd (&V2_8x[index64j+32]);
      __m512d VYi_2_Vec = _mm512_load_pd (&V2_8x[index64j+40]);
      __m512d VZr_2_Vec = _mm512_load_pd (&V2_8x[index64j+48]);
      __m512d VZi_2_Vec = _mm512_load_pd (&V2_8x[index64j+56]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      __m512d f0r_Vec, fXr_Vec, fYr_Vec, fZr_Vec;
      __m512d f0i_Vec, fXi_Vec, fYi_Vec, fZi_Vec;
      //calc f0, fX, fY, fZ
      {
        NPME_PotHelmholtz_AVX_512 (k0,
          f0r_Vec, f0i_Vec, fXr_Vec, fXi_Vec,
          fYr_Vec, fYi_Vec, fZr_Vec, fZi_Vec,
          xVec, yVec, zVec);
      }
      
      //V1 contribution
      V0r_1_Vec  = _mm512_fmsub_pd  (f0i_Vec, q2i_Vec, V0r_1_Vec);
      V0r_1_Vec  = _mm512_fmsub_pd  (f0r_Vec, q2r_Vec, V0r_1_Vec);
      VXr_1_Vec  = _mm512_fmsub_pd  (fXi_Vec, q2i_Vec, VXr_1_Vec);
      VXr_1_Vec  = _mm512_fmsub_pd  (fXr_Vec, q2r_Vec, VXr_1_Vec);
      VYr_1_Vec  = _mm512_fmsub_pd  (fYi_Vec, q2i_Vec, VYr_1_Vec);
      VYr_1_Vec  = _mm512_fmsub_pd  (fYr_Vec, q2r_Vec, VYr_1_Vec);
      VZr_1_Vec  = _mm512_fmsub_pd  (fZi_Vec, q2i_Vec, VZr_1_Vec);
      VZr_1_Vec  = _mm512_fmsub_pd  (fZr_Vec, q2r_Vec, VZr_1_Vec);

      V0i_1_Vec  = _mm512_fmadd_pd  (f0r_Vec, q2i_Vec, V0i_1_Vec);
      V0i_1_Vec  = _mm512_fmadd_pd  (f0i_Vec, q2r_Vec, V0i_1_Vec);
      VXi_1_Vec  = _mm512_fmadd_pd  (fXr_Vec, q2i_Vec, VXi_1_Vec);
      VXi_1_Vec  = _mm512_fmadd_pd  (fXi_Vec, q2r_Vec, VXi_1_Vec);
      VYi_1_Vec  = _mm512_fmadd_pd  (fYr_Vec, q2i_Vec, VYi_1_Vec);
      VYi_1_Vec  = _mm512_fmadd_pd  (fYi_Vec, q2r_Vec, VYi_1_Vec);
      VZi_1_Vec  = _mm512_fmadd_pd  (fZr_Vec, q2i_Vec, VZi_1_Vec);
      VZi_1_Vec  = _mm512_fmadd_pd  (fZi_Vec, q2r_Vec, VZi_1_Vec);

      //V2 contribution
      V0r_2_Vec  = _mm512_fmsub_pd  (f0i_Vec, q1i_Vec, V0r_2_Vec);
      V0r_2_Vec  = _mm512_fmsub_pd  (f0r_Vec, q1r_Vec, V0r_2_Vec);
      VXr_2_Vec  = _mm512_fmsub_pd  (fXr_Vec, q1r_Vec, VXr_2_Vec);
      VXr_2_Vec  = _mm512_fmsub_pd  (fXi_Vec, q1i_Vec, VXr_2_Vec);
      VYr_2_Vec  = _mm512_fmsub_pd  (fYr_Vec, q1r_Vec, VYr_2_Vec);
      VYr_2_Vec  = _mm512_fmsub_pd  (fYi_Vec, q1i_Vec, VYr_2_Vec);
      VZr_2_Vec  = _mm512_fmsub_pd  (fZr_Vec, q1r_Vec, VZr_2_Vec);
      VZr_2_Vec  = _mm512_fmsub_pd  (fZi_Vec, q1i_Vec, VZr_2_Vec);

      V0i_2_Vec  = _mm512_fmadd_pd  (f0r_Vec, q1i_Vec, V0i_2_Vec);
      V0i_2_Vec  = _mm512_fmadd_pd  (f0i_Vec, q1r_Vec, V0i_2_Vec);
      VXi_2_Vec  = _mm512_fmsub_pd  (fXr_Vec,  q1i_Vec, VXi_2_Vec);
      VXi_2_Vec  = _mm512_fmsub_pd  (fXi_Vec, q1Nr_Vec, VXi_2_Vec);
      VYi_2_Vec  = _mm512_fmsub_pd  (fYr_Vec,  q1i_Vec, VYi_2_Vec);
      VYi_2_Vec  = _mm512_fmsub_pd  (fYi_Vec, q1Nr_Vec, VYi_2_Vec);
      VZi_2_Vec  = _mm512_fmsub_pd  (fZr_Vec,  q1i_Vec, VZi_2_Vec);
      VZi_2_Vec  = _mm512_fmsub_pd  (fZi_Vec, q1Nr_Vec, VZi_2_Vec);

      _mm512_store_pd (&V2_8x[index64j   ], V0r_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+ 8], V0i_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+16], VXr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+24], VXi_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+32], VYr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+40], VYi_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+48], VZr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+56], VZi_2_Vec);
    }


    //accumulate and store V1 contributions
    double sum_r[4];
    double sum_i[4];
    NPME_mm512_4x8HorizontalSum_pd (sum_r, V0r_1_Vec, VXr_1_Vec, 
                                       VYr_1_Vec, VZr_1_Vec);
    NPME_mm512_4x8HorizontalSum_pd (sum_i, V0i_1_Vec, VXi_1_Vec, 
                                       VYi_1_Vec, VZi_1_Vec);

    if (zeroArray)
    {
      V[4*i  ]  = sum_r[0] + I*sum_i[0];
      V[4*i+1]  = sum_r[1] + I*sum_i[1];
      V[4*i+2]  = sum_r[2] + I*sum_i[2];
      V[4*i+3]  = sum_r[3] + I*sum_i[3];
    }
    else
    {
      V[4*i  ]  += sum_r[0] + I*sum_i[0];
      V[4*i+1]  += sum_r[1] + I*sum_i[1];
      V[4*i+2]  += sum_r[2] + I*sum_i[2];
      V[4*i+3]  += sum_r[3] + I*sum_i[3];
    }
  }

  NPME_TransformUpdateComplexV1_8x_2_V1_AVX_512 (nCharge, V2_8x, V);
}

void NPME_PotHelmholtz_SR_DM_Radial_AVX_512 (
  __m512d& f0rVec, __m512d& f0iVec,
  __m512d& f1rVec, __m512d& f1iVec, const __m512d& rVec, 
  const __m512d& k0rVec, const __m512d& nk0iVec)
{
  __m512d sinVec, cosVec, expVec;
  sinVec  = _mm512_sincos_pd (&cosVec, _mm512_mul_pd (k0rVec, rVec));

  expVec  = _mm512_exp_pd  (_mm512_mul_pd (nk0iVec, rVec));
  expVec  = _mm512_div_pd  (expVec, rVec);

  f0rVec  = _mm512_mul_pd  (expVec,    cosVec);
  f0iVec  = _mm512_mul_pd  (expVec,    sinVec);


  //g = f0/r
  __m512d grVec = _mm512_div_pd  (f0rVec , rVec);
  __m512d giVec = _mm512_div_pd  (f0iVec , rVec);

  //f1 = 1/r df0/dr
  #if NPME_USE_AVX_512_FMA
  {
    f1rVec  = _mm512_fmadd_pd  ( k0rVec, f0iVec,  grVec);
    f1rVec  = _mm512_fmsub_pd  (nk0iVec, f0rVec, f1rVec);
    f1iVec  = _mm512_fmsub_pd  (nk0iVec, f0iVec,  giVec);
    f1iVec  = _mm512_fmadd_pd  ( k0rVec, f0rVec, f1iVec);
  }
  #else
  {
    f1rVec  = _mm512_add_pd  (_mm512_mul_pd ( k0rVec, f0iVec),  grVec);
    f1rVec  = _mm512_sub_pd  (_mm512_mul_pd (nk0iVec, f0rVec), f1rVec);
    f1iVec  = _mm512_sub_pd  (_mm512_mul_pd (nk0iVec, f0iVec),  giVec);
    f1iVec  = _mm512_add_pd  (_mm512_mul_pd ( k0rVec, f0rVec), f1iVec);
  }
  #endif

  f1rVec = _mm512_div_pd  (f1rVec, rVec);
  f1iVec = _mm512_div_pd  (f1iVec, rVec);
}
void NPME_PotHelmholtz_SR_DM_AVX_512 (const _Complex double k0,
  __m512d& f0r_Vec, __m512d& f0i_Vec, __m512d& fXr_Vec, __m512d& fXi_Vec,
  __m512d& fYr_Vec, __m512d& fYi_Vec, __m512d& fZr_Vec, __m512d& fZi_Vec,
  const __m512d& xVec, const __m512d& yVec, const __m512d& zVec,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir)
{
  const __m512d RdirVec = _mm512_set1_pd( Rdir );
  const __m512d k0rVec  = _mm512_set1_pd( creal(k0) );
  const __m512d nk0iVec = _mm512_set1_pd(-cimag(k0) );
  const __m512d zeroVec = _mm512_set1_pd(0.0);

  __m512d r2Vec = _mm512_mul_pd  (xVec, xVec);
  r2Vec  = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
  r2Vec  = _mm512_fmadd_pd  (zVec, zVec, r2Vec);

  __m512d rVec  = _mm512_sqrt_pd (r2Vec);

  __m512d f1r_Vec, f1i_Vec;
  {
    __m512d f0r_AVec, f0i_AVec;
    __m512d f0r_BVec, f0i_BVec;
    __m512d f1r_AVec, f1i_AVec;
    __m512d f1r_BVec, f1i_BVec;

    NPME_FunctionDerivMatch_EvenSeriesComplex_AVX_512 (f0r_AVec, f0i_AVec, 
      f1r_AVec, f1i_AVec, r2Vec, Nder, a, b);
    NPME_PotHelmholtz_SR_DM_Radial_AVX_512 (f0r_BVec, f0i_BVec, 
      f1r_BVec, f1i_BVec, rVec, k0rVec, nk0iVec);

    f0r_BVec = _mm512_sub_pd (f0r_BVec, f0r_AVec);
    f0i_BVec = _mm512_sub_pd (f0i_BVec, f0i_AVec);
    f1r_BVec = _mm512_sub_pd (f1r_BVec, f1r_AVec);
    f1i_BVec = _mm512_sub_pd (f1i_BVec, f1i_AVec);

    //use (f0_BVec) if r < Rdir
    //use (zeroVec) if r > Rdir
    {
      __mmask8 maskVec = _mm512_cmp_pd_mask (rVec, RdirVec, 1);
      f0r_Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f0r_BVec);
      f0i_Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f0i_BVec);
      f1r_Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f1r_BVec);
      f1i_Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f1i_BVec);
    }
  }

  fXr_Vec = _mm512_mul_pd  (f1r_Vec, xVec);
  fYr_Vec = _mm512_mul_pd  (f1r_Vec, yVec);
  fZr_Vec = _mm512_mul_pd  (f1r_Vec, zVec);

  fXi_Vec = _mm512_mul_pd  (f1i_Vec, xVec);
  fYi_Vec = _mm512_mul_pd  (f1i_Vec, yVec);
  fZi_Vec = _mm512_mul_pd  (f1i_Vec, zVec);
}

void NPME_PotHelmholtz_SR_DM_Pair_V1_AVX_512 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotHelmholtz_Pair_V1_AVX_512.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  const size_t remain2    = nCharge2%8;
  const size_t nLoop2     = (nCharge2 - remain2)/8;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  double qCoCrd_8x[5*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_8x    [8*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));

  memset(V2_8x, 0, 64*nLoop2wRemainder*sizeof(double));
  NPME_TransformQcomplexCoord_8x_AVX_512 (nCharge2, q2, coord2, qCoCrd_8x);

  for (size_t i = 0; i < nCharge1; i++)
  {
    const double *r1        = &coord1[3*i];
    const __m512d q1r_Vec   = _mm512_set1_pd(creal( q1[i]));
    const __m512d q1i_Vec   = _mm512_set1_pd(cimag( q1[i]));
    const __m512d q1Nr_Vec  = _mm512_set1_pd(creal(-q1[i]));

    const __m512d x1Vec     = _mm512_set1_pd(r1[0]);
    const __m512d y1Vec     = _mm512_set1_pd(r1[1]);
    const __m512d z1Vec     = _mm512_set1_pd(r1[2]);

    __m512d V0r_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VXr_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VYr_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VZr_1_Vec       = _mm512_set1_pd(0.0);

    __m512d V0i_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VXi_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VYi_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VZi_1_Vec       = _mm512_set1_pd(0.0);

    size_t index40j = 0;
    size_t index64j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      const __m512d q2r_Vec = _mm512_load_pd (&qCoCrd_8x[index40j   ]);
      const __m512d q2i_Vec = _mm512_load_pd (&qCoCrd_8x[index40j+ 8]);

      __m512d f0r_Vec, fXr_Vec, fYr_Vec, fZr_Vec;
      __m512d f0i_Vec, fXi_Vec, fYi_Vec, fZi_Vec;
      //calc f0, fX, fY, fZ
      {
        __m512d xVec        = _mm512_load_pd (&qCoCrd_8x[index40j+16]);
        __m512d yVec        = _mm512_load_pd (&qCoCrd_8x[index40j+24]);
        __m512d zVec        = _mm512_load_pd (&qCoCrd_8x[index40j+32]);

        xVec = _mm512_sub_pd (x1Vec, xVec);
        yVec = _mm512_sub_pd (y1Vec, yVec);
        zVec = _mm512_sub_pd (z1Vec, zVec);

        NPME_PotHelmholtz_SR_DM_AVX_512 (k0,
          f0r_Vec, f0i_Vec, fXr_Vec, fXi_Vec,
          fYr_Vec, fYi_Vec, fZr_Vec, fZi_Vec,
          xVec, yVec, zVec, Nder, a, b, Rdir);
      }

      //V   = f*q
      //V_r = f_r*q_r - f_i*q_i
      //V_i = f_r*q_i + f_i*q_r
        __m512d V0r_2_Vec = _mm512_load_pd (&V2_8x[index64j   ]);
      __m512d V0i_2_Vec = _mm512_load_pd (&V2_8x[index64j+ 8]);
      __m512d VXr_2_Vec = _mm512_load_pd (&V2_8x[index64j+16]);
      __m512d VXi_2_Vec = _mm512_load_pd (&V2_8x[index64j+24]);
      __m512d VYr_2_Vec = _mm512_load_pd (&V2_8x[index64j+32]);
      __m512d VYi_2_Vec = _mm512_load_pd (&V2_8x[index64j+40]);
      __m512d VZr_2_Vec = _mm512_load_pd (&V2_8x[index64j+48]);
      __m512d VZi_2_Vec = _mm512_load_pd (&V2_8x[index64j+56]);

      //V1 contribution
      V0r_1_Vec  = _mm512_fmsub_pd  (f0i_Vec, q2i_Vec, V0r_1_Vec);
      V0r_1_Vec  = _mm512_fmsub_pd  (f0r_Vec, q2r_Vec, V0r_1_Vec);
      VXr_1_Vec  = _mm512_fmsub_pd  (fXi_Vec, q2i_Vec, VXr_1_Vec);
      VXr_1_Vec  = _mm512_fmsub_pd  (fXr_Vec, q2r_Vec, VXr_1_Vec);
      VYr_1_Vec  = _mm512_fmsub_pd  (fYi_Vec, q2i_Vec, VYr_1_Vec);
      VYr_1_Vec  = _mm512_fmsub_pd  (fYr_Vec, q2r_Vec, VYr_1_Vec);
      VZr_1_Vec  = _mm512_fmsub_pd  (fZi_Vec, q2i_Vec, VZr_1_Vec);
      VZr_1_Vec  = _mm512_fmsub_pd  (fZr_Vec, q2r_Vec, VZr_1_Vec);

      V0i_1_Vec  = _mm512_fmadd_pd  (f0r_Vec, q2i_Vec, V0i_1_Vec);
      V0i_1_Vec  = _mm512_fmadd_pd  (f0i_Vec, q2r_Vec, V0i_1_Vec);
      VXi_1_Vec  = _mm512_fmadd_pd  (fXr_Vec, q2i_Vec, VXi_1_Vec);
      VXi_1_Vec  = _mm512_fmadd_pd  (fXi_Vec, q2r_Vec, VXi_1_Vec);
      VYi_1_Vec  = _mm512_fmadd_pd  (fYr_Vec, q2i_Vec, VYi_1_Vec);
      VYi_1_Vec  = _mm512_fmadd_pd  (fYi_Vec, q2r_Vec, VYi_1_Vec);
      VZi_1_Vec  = _mm512_fmadd_pd  (fZr_Vec, q2i_Vec, VZi_1_Vec);
      VZi_1_Vec  = _mm512_fmadd_pd  (fZi_Vec, q2r_Vec, VZi_1_Vec);

      //V2 contribution
      V0r_2_Vec  = _mm512_fmsub_pd  (f0i_Vec, q1i_Vec, V0r_2_Vec);
      V0r_2_Vec  = _mm512_fmsub_pd  (f0r_Vec, q1r_Vec, V0r_2_Vec);
      VXr_2_Vec  = _mm512_fmsub_pd  (fXr_Vec, q1r_Vec, VXr_2_Vec);
      VXr_2_Vec  = _mm512_fmsub_pd  (fXi_Vec, q1i_Vec, VXr_2_Vec);
      VYr_2_Vec  = _mm512_fmsub_pd  (fYr_Vec, q1r_Vec, VYr_2_Vec);
      VYr_2_Vec  = _mm512_fmsub_pd  (fYi_Vec, q1i_Vec, VYr_2_Vec);
      VZr_2_Vec  = _mm512_fmsub_pd  (fZr_Vec, q1r_Vec, VZr_2_Vec);
      VZr_2_Vec  = _mm512_fmsub_pd  (fZi_Vec, q1i_Vec, VZr_2_Vec);

      V0i_2_Vec  = _mm512_fmadd_pd  (f0r_Vec, q1i_Vec, V0i_2_Vec);
      V0i_2_Vec  = _mm512_fmadd_pd  (f0i_Vec, q1r_Vec, V0i_2_Vec);
      VXi_2_Vec  = _mm512_fmsub_pd  (fXr_Vec,  q1i_Vec, VXi_2_Vec);
      VXi_2_Vec  = _mm512_fmsub_pd  (fXi_Vec, q1Nr_Vec, VXi_2_Vec);
      VYi_2_Vec  = _mm512_fmsub_pd  (fYr_Vec,  q1i_Vec, VYi_2_Vec);
      VYi_2_Vec  = _mm512_fmsub_pd  (fYi_Vec, q1Nr_Vec, VYi_2_Vec);
      VZi_2_Vec  = _mm512_fmsub_pd  (fZr_Vec,  q1i_Vec, VZi_2_Vec);
      VZi_2_Vec  = _mm512_fmsub_pd  (fZi_Vec, q1Nr_Vec, VZi_2_Vec);

      _mm512_store_pd (&V2_8x[index64j   ], V0r_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+ 8], V0i_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+16], VXr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+24], VXi_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+32], VYr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+40], VYi_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+48], VZr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+56], VZi_2_Vec);

      index40j += 40;
      index64j += 64;
    }

    //accumulate and store V1 contributions
    double sum_r[4];
    double sum_i[4];
    NPME_mm512_4x8HorizontalSum_pd (sum_r, V0r_1_Vec, VXr_1_Vec, 
                                       VYr_1_Vec, VZr_1_Vec);
    NPME_mm512_4x8HorizontalSum_pd (sum_i, V0i_1_Vec, VXi_1_Vec, 
                                       VYi_1_Vec, VZi_1_Vec);

    if (zeroArray)
    {
      V1[4*i  ]  = sum_r[0] + I*sum_i[0];
      V1[4*i+1]  = sum_r[1] + I*sum_i[1];
      V1[4*i+2]  = sum_r[2] + I*sum_i[2];
      V1[4*i+3]  = sum_r[3] + I*sum_i[3];
    }
    else
    {
      V1[4*i  ]  += sum_r[0] + I*sum_i[0];
      V1[4*i+1]  += sum_r[1] + I*sum_i[1];
      V1[4*i+2]  += sum_r[2] + I*sum_i[2];
      V1[4*i+3]  += sum_r[3] + I*sum_i[3];
    }
  }

  if (zeroArray)
    NPME_TransformComplexV1_8x_2_V1_AVX_512 (nCharge2, V2_8x, &V2[0]);
  else
    NPME_TransformUpdateComplexV1_8x_2_V1_AVX_512 (nCharge2, V2_8x, &V2[0]);
}


void NPME_PotHelmholtz_SR_DM_Self_V1_AVX_512 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir, const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotHelmholtz_Self_V1_AVX_512.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  double qCoCrd_8x[5*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  NPME_TransformQcomplexCoord_8x_AVX_512 (nCharge, q, coord, qCoCrd_8x);

  double V2_8x[8*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  {
    const size_t remain    = nCharge%8;
    const size_t nLoop     = (nCharge - remain)/8;
    size_t nLoopwRemainder = nLoop;
    if (remain > 0)
      nLoopwRemainder++;

    memset(V2_8x, 0, 64*nLoopwRemainder*sizeof(double));
  }

  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r1        = &coord[3*i];
    __m512d q1r_Vec         = _mm512_set1_pd(creal( q[i]));
    __m512d q1i_Vec         = _mm512_set1_pd(cimag( q[i]));
    __m512d q1Nr_Vec        = _mm512_set1_pd(creal(-q[i]));

    const __m512d x1Vec     = _mm512_set1_pd(r1[0]);
    const __m512d y1Vec     = _mm512_set1_pd(r1[1]);
    const __m512d z1Vec     = _mm512_set1_pd(r1[2]);

    __m512d V0r_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VXr_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VYr_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VZr_1_Vec       = _mm512_set1_pd(0.0);

    __m512d V0i_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VXi_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VYi_1_Vec       = _mm512_set1_pd(0.0);
    __m512d VZi_1_Vec       = _mm512_set1_pd(0.0);

    const size_t remain2  = (i)%8;
    const size_t nLoop2   = (i-remain2)/8;

    size_t index40j = 0;
    size_t index64j = 0;
    for (size_t j = 0; j < nLoop2; j++)
    {
      const __m512d q2r_Vec = _mm512_load_pd (&qCoCrd_8x[index40j   ]);
      const __m512d q2i_Vec = _mm512_load_pd (&qCoCrd_8x[index40j+ 8]);

      __m512d f0r_Vec, fXr_Vec, fYr_Vec, fZr_Vec;
      __m512d f0i_Vec, fXi_Vec, fYi_Vec, fZi_Vec;
      //calc f0, fX, fY, fZ
      {
        __m512d xVec        = _mm512_load_pd (&qCoCrd_8x[index40j+16]);
        __m512d yVec        = _mm512_load_pd (&qCoCrd_8x[index40j+24]);
        __m512d zVec        = _mm512_load_pd (&qCoCrd_8x[index40j+32]);

        xVec = _mm512_sub_pd (x1Vec, xVec);
        yVec = _mm512_sub_pd (y1Vec, yVec);
        zVec = _mm512_sub_pd (z1Vec, zVec);

        NPME_PotHelmholtz_SR_DM_AVX_512 (k0,
          f0r_Vec, f0i_Vec, fXr_Vec, fXi_Vec,
          fYr_Vec, fYi_Vec, fZr_Vec, fZi_Vec,
          xVec, yVec, zVec, Nder, a, b, Rdir);
      }

      //V   = f*q
      //V_r = f_r*q_r - f_i*q_i
      //V_i = f_r*q_i + f_i*q_r
  
      __m512d V0r_2_Vec = _mm512_load_pd (&V2_8x[index64j   ]);
      __m512d V0i_2_Vec = _mm512_load_pd (&V2_8x[index64j+ 8]);
      __m512d VXr_2_Vec = _mm512_load_pd (&V2_8x[index64j+16]);
      __m512d VXi_2_Vec = _mm512_load_pd (&V2_8x[index64j+24]);
      __m512d VYr_2_Vec = _mm512_load_pd (&V2_8x[index64j+32]);
      __m512d VYi_2_Vec = _mm512_load_pd (&V2_8x[index64j+40]);
      __m512d VZr_2_Vec = _mm512_load_pd (&V2_8x[index64j+48]);
      __m512d VZi_2_Vec = _mm512_load_pd (&V2_8x[index64j+56]);


      //V   = f*q
      //V_r = f_r*q_r - f_i*q_i
      //V_i = f_r*q_i + f_i*q_r
      //V1 contribution
      V0r_1_Vec  = _mm512_fmsub_pd  (f0i_Vec, q2i_Vec, V0r_1_Vec);
      V0r_1_Vec  = _mm512_fmsub_pd  (f0r_Vec, q2r_Vec, V0r_1_Vec);
      VXr_1_Vec  = _mm512_fmsub_pd  (fXi_Vec, q2i_Vec, VXr_1_Vec);
      VXr_1_Vec  = _mm512_fmsub_pd  (fXr_Vec, q2r_Vec, VXr_1_Vec);
      VYr_1_Vec  = _mm512_fmsub_pd  (fYi_Vec, q2i_Vec, VYr_1_Vec);
      VYr_1_Vec  = _mm512_fmsub_pd  (fYr_Vec, q2r_Vec, VYr_1_Vec);
      VZr_1_Vec  = _mm512_fmsub_pd  (fZi_Vec, q2i_Vec, VZr_1_Vec);
      VZr_1_Vec  = _mm512_fmsub_pd  (fZr_Vec, q2r_Vec, VZr_1_Vec);

      V0i_1_Vec  = _mm512_fmadd_pd  (f0r_Vec, q2i_Vec, V0i_1_Vec);
      V0i_1_Vec  = _mm512_fmadd_pd  (f0i_Vec, q2r_Vec, V0i_1_Vec);
      VXi_1_Vec  = _mm512_fmadd_pd  (fXr_Vec, q2i_Vec, VXi_1_Vec);
      VXi_1_Vec  = _mm512_fmadd_pd  (fXi_Vec, q2r_Vec, VXi_1_Vec);
      VYi_1_Vec  = _mm512_fmadd_pd  (fYr_Vec, q2i_Vec, VYi_1_Vec);
      VYi_1_Vec  = _mm512_fmadd_pd  (fYi_Vec, q2r_Vec, VYi_1_Vec);
      VZi_1_Vec  = _mm512_fmadd_pd  (fZr_Vec, q2i_Vec, VZi_1_Vec);
      VZi_1_Vec  = _mm512_fmadd_pd  (fZi_Vec, q2r_Vec, VZi_1_Vec);

      //V2 contribution
      V0r_2_Vec  = _mm512_fmsub_pd  (f0i_Vec, q1i_Vec, V0r_2_Vec);
      V0r_2_Vec  = _mm512_fmsub_pd  (f0r_Vec, q1r_Vec, V0r_2_Vec);
      VXr_2_Vec  = _mm512_fmsub_pd  (fXr_Vec, q1r_Vec, VXr_2_Vec);
      VXr_2_Vec  = _mm512_fmsub_pd  (fXi_Vec, q1i_Vec, VXr_2_Vec);
      VYr_2_Vec  = _mm512_fmsub_pd  (fYr_Vec, q1r_Vec, VYr_2_Vec);
      VYr_2_Vec  = _mm512_fmsub_pd  (fYi_Vec, q1i_Vec, VYr_2_Vec);
      VZr_2_Vec  = _mm512_fmsub_pd  (fZr_Vec, q1r_Vec, VZr_2_Vec);
      VZr_2_Vec  = _mm512_fmsub_pd  (fZi_Vec, q1i_Vec, VZr_2_Vec);

      V0i_2_Vec  = _mm512_fmadd_pd  (f0r_Vec, q1i_Vec, V0i_2_Vec);
      V0i_2_Vec  = _mm512_fmadd_pd  (f0i_Vec, q1r_Vec, V0i_2_Vec);
      VXi_2_Vec  = _mm512_fmsub_pd  (fXr_Vec,  q1i_Vec, VXi_2_Vec);
      VXi_2_Vec  = _mm512_fmsub_pd  (fXi_Vec, q1Nr_Vec, VXi_2_Vec);
      VYi_2_Vec  = _mm512_fmsub_pd  (fYr_Vec,  q1i_Vec, VYi_2_Vec);
      VYi_2_Vec  = _mm512_fmsub_pd  (fYi_Vec, q1Nr_Vec, VYi_2_Vec);
      VZi_2_Vec  = _mm512_fmsub_pd  (fZr_Vec,  q1i_Vec, VZi_2_Vec);
      VZi_2_Vec  = _mm512_fmsub_pd  (fZi_Vec, q1Nr_Vec, VZi_2_Vec);

      _mm512_store_pd (&V2_8x[index64j   ], V0r_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+ 8], V0i_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+16], VXr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+24], VXi_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+32], VYr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+40], VYi_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+48], VZr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+56], VZi_2_Vec);

      index40j += 40;
      index64j += 64;
    }

    if (remain2 > 0)
    {
      const size_t indexStart     = 8*nLoop2;
      const _Complex double *qLoc = &q[indexStart];
      const double *crdLoc        = &coord[3*indexStart];
      const double X              = r1[0]+1.0;
      double x2Array[8]  __attribute__((aligned(64))) = {X,X,X,X,X,X,X,X};
      double y2Array[8]  __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};
      double z2Array[8]  __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};
      double q2rArray[8] __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};
      double q2iArray[8] __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};
      double mskArray[8] __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};

      for (size_t k = 0; k < remain2; k++)
      {
        q2rArray[k] = creal(qLoc[k]);
        q2iArray[k] = cimag(qLoc[k]);
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
        mskArray[k] = 1.0;
      }
      __m512d q2r_Vec   = _mm512_load_pd (q2rArray);
      __m512d q2i_Vec   = _mm512_load_pd (q2iArray);
      __m512d xVec      = _mm512_load_pd (x2Array);
      __m512d yVec      = _mm512_load_pd (y2Array);
      __m512d zVec      = _mm512_load_pd (z2Array);
      __m512d mskVec    = _mm512_load_pd (mskArray);

      //apply mask to q1
      q1r_Vec   = _mm512_mul_pd (mskVec, q1r_Vec);
      q1i_Vec   = _mm512_mul_pd (mskVec, q1i_Vec);
      q1Nr_Vec  = _mm512_mul_pd (mskVec, q1Nr_Vec);

      __m512d V0r_2_Vec = _mm512_load_pd (&V2_8x[index64j   ]);
      __m512d V0i_2_Vec = _mm512_load_pd (&V2_8x[index64j+ 8]);
      __m512d VXr_2_Vec = _mm512_load_pd (&V2_8x[index64j+16]);
      __m512d VXi_2_Vec = _mm512_load_pd (&V2_8x[index64j+24]);
      __m512d VYr_2_Vec = _mm512_load_pd (&V2_8x[index64j+32]);
      __m512d VYi_2_Vec = _mm512_load_pd (&V2_8x[index64j+40]);
      __m512d VZr_2_Vec = _mm512_load_pd (&V2_8x[index64j+48]);
      __m512d VZi_2_Vec = _mm512_load_pd (&V2_8x[index64j+56]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      __m512d f0r_Vec, fXr_Vec, fYr_Vec, fZr_Vec;
      __m512d f0i_Vec, fXi_Vec, fYi_Vec, fZi_Vec;
      //calc f0, fX, fY, fZ
      {
        NPME_PotHelmholtz_SR_DM_AVX_512 (k0,
          f0r_Vec, f0i_Vec, fXr_Vec, fXi_Vec,
          fYr_Vec, fYi_Vec, fZr_Vec, fZi_Vec,
          xVec, yVec, zVec, Nder, a, b, Rdir);
      }
      
      //V1 contribution
      V0r_1_Vec  = _mm512_fmsub_pd  (f0i_Vec, q2i_Vec, V0r_1_Vec);
      V0r_1_Vec  = _mm512_fmsub_pd  (f0r_Vec, q2r_Vec, V0r_1_Vec);
      VXr_1_Vec  = _mm512_fmsub_pd  (fXi_Vec, q2i_Vec, VXr_1_Vec);
      VXr_1_Vec  = _mm512_fmsub_pd  (fXr_Vec, q2r_Vec, VXr_1_Vec);
      VYr_1_Vec  = _mm512_fmsub_pd  (fYi_Vec, q2i_Vec, VYr_1_Vec);
      VYr_1_Vec  = _mm512_fmsub_pd  (fYr_Vec, q2r_Vec, VYr_1_Vec);
      VZr_1_Vec  = _mm512_fmsub_pd  (fZi_Vec, q2i_Vec, VZr_1_Vec);
      VZr_1_Vec  = _mm512_fmsub_pd  (fZr_Vec, q2r_Vec, VZr_1_Vec);

      V0i_1_Vec  = _mm512_fmadd_pd  (f0r_Vec, q2i_Vec, V0i_1_Vec);
      V0i_1_Vec  = _mm512_fmadd_pd  (f0i_Vec, q2r_Vec, V0i_1_Vec);
      VXi_1_Vec  = _mm512_fmadd_pd  (fXr_Vec, q2i_Vec, VXi_1_Vec);
      VXi_1_Vec  = _mm512_fmadd_pd  (fXi_Vec, q2r_Vec, VXi_1_Vec);
      VYi_1_Vec  = _mm512_fmadd_pd  (fYr_Vec, q2i_Vec, VYi_1_Vec);
      VYi_1_Vec  = _mm512_fmadd_pd  (fYi_Vec, q2r_Vec, VYi_1_Vec);
      VZi_1_Vec  = _mm512_fmadd_pd  (fZr_Vec, q2i_Vec, VZi_1_Vec);
      VZi_1_Vec  = _mm512_fmadd_pd  (fZi_Vec, q2r_Vec, VZi_1_Vec);

      //V2 contribution
      V0r_2_Vec  = _mm512_fmsub_pd  (f0i_Vec, q1i_Vec, V0r_2_Vec);
      V0r_2_Vec  = _mm512_fmsub_pd  (f0r_Vec, q1r_Vec, V0r_2_Vec);
      VXr_2_Vec  = _mm512_fmsub_pd  (fXr_Vec, q1r_Vec, VXr_2_Vec);
      VXr_2_Vec  = _mm512_fmsub_pd  (fXi_Vec, q1i_Vec, VXr_2_Vec);
      VYr_2_Vec  = _mm512_fmsub_pd  (fYr_Vec, q1r_Vec, VYr_2_Vec);
      VYr_2_Vec  = _mm512_fmsub_pd  (fYi_Vec, q1i_Vec, VYr_2_Vec);
      VZr_2_Vec  = _mm512_fmsub_pd  (fZr_Vec, q1r_Vec, VZr_2_Vec);
      VZr_2_Vec  = _mm512_fmsub_pd  (fZi_Vec, q1i_Vec, VZr_2_Vec);

      V0i_2_Vec  = _mm512_fmadd_pd  (f0r_Vec, q1i_Vec, V0i_2_Vec);
      V0i_2_Vec  = _mm512_fmadd_pd  (f0i_Vec, q1r_Vec, V0i_2_Vec);
      VXi_2_Vec  = _mm512_fmsub_pd  (fXr_Vec,  q1i_Vec, VXi_2_Vec);
      VXi_2_Vec  = _mm512_fmsub_pd  (fXi_Vec, q1Nr_Vec, VXi_2_Vec);
      VYi_2_Vec  = _mm512_fmsub_pd  (fYr_Vec,  q1i_Vec, VYi_2_Vec);
      VYi_2_Vec  = _mm512_fmsub_pd  (fYi_Vec, q1Nr_Vec, VYi_2_Vec);
      VZi_2_Vec  = _mm512_fmsub_pd  (fZr_Vec,  q1i_Vec, VZi_2_Vec);
      VZi_2_Vec  = _mm512_fmsub_pd  (fZi_Vec, q1Nr_Vec, VZi_2_Vec);

      _mm512_store_pd (&V2_8x[index64j   ], V0r_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+ 8], V0i_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+16], VXr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+24], VXi_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+32], VYr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+40], VYi_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+48], VZr_2_Vec);
      _mm512_store_pd (&V2_8x[index64j+56], VZi_2_Vec);
    }


    //accumulate and store V1 contributions
    double sum_r[4];
    double sum_i[4];
    NPME_mm512_4x8HorizontalSum_pd (sum_r, V0r_1_Vec, VXr_1_Vec, 
                                       VYr_1_Vec, VZr_1_Vec);
    NPME_mm512_4x8HorizontalSum_pd (sum_i, V0i_1_Vec, VXi_1_Vec, 
                                       VYi_1_Vec, VZi_1_Vec);

    if (zeroArray)
    {
      V[4*i  ]  = sum_r[0] + I*sum_i[0];
      V[4*i+1]  = sum_r[1] + I*sum_i[1];
      V[4*i+2]  = sum_r[2] + I*sum_i[2];
      V[4*i+3]  = sum_r[3] + I*sum_i[3];
    }
    else
    {
      V[4*i  ]  += sum_r[0] + I*sum_i[0];
      V[4*i+1]  += sum_r[1] + I*sum_i[1];
      V[4*i+2]  += sum_r[2] + I*sum_i[2];
      V[4*i+3]  += sum_r[3] + I*sum_i[3];
    }
  }

  NPME_TransformUpdateComplexV1_8x_2_V1_AVX_512 (nCharge, V2_8x, V);
}




#endif


void NPME_PotHelmholtz_Pair_V1 (const _Complex double k0,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, int vecOption, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (vecOption == 0)
  {
    NPME_PotHelmholtz_Pair_V1 (k0,
      nCharge1, coord1, q1,
      nCharge2, coord2, q2, V1, V2, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotHelmholtz_Pair_V1_AVX (k0,
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotHelmholtz_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotHelmholtz_Pair_V1_AVX_512 (k0,
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotHelmholtz_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}
void NPME_PotHelmholtz_Self_V1 (const _Complex double k0,
  const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, int vecOption, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (vecOption == 0)
  {
    NPME_PotHelmholtz_Self_V1 (k0, nCharge, coord, q, V, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotHelmholtz_Self_V1_AVX (k0, nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotHelmholtz_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotHelmholtz_Self_V1_AVX_512 (k0, nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotHelmholtz_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}















void NPME_PotHelmholtz_SR_DM_Pair_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, int vecOption, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//        a[Nder+1], b[Nder+1]
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (vecOption == 0)
  {
    NPME_PotHelmholtz_SR_DM_Pair_V1 (k0,
      Nder, a, b, Rdir,
      nCharge1, coord1, q1,
      nCharge2, coord2, q2, V1, V2, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotHelmholtz_SR_DM_Pair_V1_AVX (k0,
        Nder, a, b, Rdir,
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotHelmholtz_SR_DM_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotHelmholtz_SR_DM_Pair_V1_AVX_512 (k0,
        Nder, a, b, Rdir,
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotHelmholtz_SR_DM_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}







void NPME_PotHelmholtz_SR_DM_Self_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir, const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, int vecOption, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//        a[Nder+1], b[Nder+1]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (vecOption == 0)
  {
    NPME_PotHelmholtz_SR_DM_Self_V1 (k0, 
      Nder, a, b, Rdir, nCharge, coord, q, V, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotHelmholtz_SR_DM_Self_V1_AVX (k0, 
        Nder, a, b, Rdir, nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotHelmholtz_SR_DM_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotHelmholtz_SR_DM_Self_V1_AVX_512 (k0, 
        Nder, a, b, Rdir, nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotHelmholtz_SR_DM_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}


void NPME_PotHelmholtz_SR_DM_LargePair_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir,
  const size_t nCharge1, const double *coord1, const _Complex double *charge1,
  const size_t nCharge2, const double *coord2, const _Complex double *charge2,
  _Complex double *V1, _Complex double *V2, int vecOption, 
  size_t blockSize, bool zeroArray)
{
  if (nCharge2 < blockSize)
    return NPME_PotHelmholtz_SR_DM_Pair_V1 (k0, Nder, a, b, Rdir,
      nCharge1, coord1, charge1,
      nCharge2, coord2, charge2,
      V1, V2, vecOption, zeroArray);


  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotHelmholtz_SR_DM_LargePair_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotHelmholtz_SR_DM_LargePair_V1\n";
    sprintf(str, "blockSize = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n",
      blockSize, NPME_Pot_MaxChgBlock_V1);
    std::cout << str;
    exit(0);
  }

  if (zeroArray)
  {
    memset(V1, 0, 4*nCharge1*sizeof(_Complex double));
    memset(V2, 0, 4*nCharge2*sizeof(_Complex double));
  }

  size_t remain1    = nCharge1%blockSize;
  size_t remain2    = nCharge2%blockSize;

  size_t nBlock1    = (nCharge1-remain1)/blockSize;
  size_t nBlock2    = (nCharge2-remain2)/blockSize;

  if (remain1 > 0) nBlock1++;
  if (remain2 > 0) nBlock2++;

  const size_t nPair = nBlock1*nBlock2;
  for (size_t k = 0; k < nPair; k++)
  {
    size_t i1, i2;
    NPME_ind2D_2_n1_n2 (k, nBlock2, i1, i2);

    const size_t start1B  = i1*blockSize;
    const size_t start2B  = i2*blockSize;

    size_t nCharge1B  = blockSize;
    size_t nCharge2B  = blockSize;

    if ( (remain1 > 0) && (i1 == nBlock1 - 1))  nCharge1B = remain1;
    if ( (remain2 > 0) && (i2 == nBlock2 - 1))  nCharge2B = remain2;

    const _Complex double *charge1B = &charge1[start1B];
    const double *coord1B           = &coord1[3*start1B];
    _Complex double *V1_B           = &V1[4*start1B];

    const _Complex double *charge2B = &charge2[start2B];
    const double *coord2B           = &coord2[3*start2B];
    _Complex double *V2_B           = &V2[4*start2B]; 

    bool zeroArrayB = 0;
    NPME_PotHelmholtz_SR_DM_Pair_V1 (k0, Nder, a, b, Rdir,
      nCharge1B, coord1B, charge1B, 
      nCharge2B, coord2B, charge2B, 
      V1_B, V2_B, vecOption, zeroArrayB);
  }
}


void NPME_PotHelmholtz_SR_DM_LargeSelf_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir, const size_t nCharge, const double *coord, 
  const _Complex double *charge, _Complex double *V, int vecOption, 
  size_t blockSize, bool zeroArray)
{
  if (nCharge < blockSize)
    return NPME_PotHelmholtz_SR_DM_Self_V1 (k0, Nder, a, b, Rdir, 
            nCharge, coord, charge, V, vecOption, zeroArray);

  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotHelmholtz_SR_DM_LargeSelf_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotHelmholtz_SR_DM_LargeSelf_V1\n";
    sprintf(str, "blockSize = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n",
      blockSize, NPME_Pot_MaxChgBlock_V1);
    std::cout << str;
    exit(0);
  }

  if (zeroArray)
    memset(V, 0, 4*nCharge*sizeof(_Complex double));


  size_t remain1    = nCharge%blockSize;
  size_t nBlock1    = (nCharge-remain1)/blockSize;

  if (remain1 > 0) nBlock1++;

  size_t nPair = (nBlock1*(nBlock1+1))/2;

  for (size_t l = 0; l < nPair; l++)
  {
    size_t i1, i2;
    NPME_ind2D_symmetric2_index_2_pq (i1, i2, l);

    if (i1 != i2)
    {
      size_t start1B    = i1*blockSize;
      size_t start2B    = i2*blockSize;
      size_t nCharge1B  = blockSize;
      size_t nCharge2B  = blockSize;

      if ( (remain1 > 0) && (i1 == nBlock1 - 1))  nCharge1B = remain1;
      if ( (remain1 > 0) && (i2 == nBlock1 - 1))  nCharge2B = remain1;

      const _Complex double *charge1B = &charge[start1B];
      const double *coord1B           = &coord[3*start1B];
      _Complex double *V1B            = &V[4*start1B];

      const _Complex double *charge2B = &charge[start2B];
      const double *coord2B           = &coord[3*start2B];
      _Complex double *V2B            = &V[4*start2B];

      bool zeroArrayB = 0;
      NPME_PotHelmholtz_SR_DM_Pair_V1 (k0, Nder, a, b, Rdir,
        nCharge1B, coord1B, charge1B, 
        nCharge2B, coord2B, charge2B, 
        V1B, V2B, vecOption, zeroArrayB);
    }
    else
    {
      size_t start1B    = i1*blockSize;
      size_t nCharge1B  = blockSize;
      if ( (remain1 > 0) && (i1 == nBlock1 - 1))  nCharge1B = remain1;

      const _Complex double *charge1B = &charge[start1B];
      const double *coord1B           = &coord[3*start1B];
      _Complex double *V1B            = &V[4*start1B];

      bool zeroArrayB = 0;
      NPME_PotHelmholtz_SR_DM_Self_V1 (k0, Nder, a, b, Rdir, 
            nCharge1B, coord1B, charge1B, V1B, vecOption, zeroArrayB);
    }
  }
}

}//end namespace NPME_Library



