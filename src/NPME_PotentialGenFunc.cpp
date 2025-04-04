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
#include "NPME_PotentialGenFunc.h"
#include "NPME_KernelFunction.h"
#include "NPME_SupportFunctions.h"
#include "NPME_PartitionBox.h"
#include "NPME_PartitionEmbeddedBox.h"
#include "NPME_PotentialSupportFunctions.h"
#include "NPME_ExtLibrary.h"
#include "NPME_AlignedArray.h"


namespace NPME_Library
{
void NPME_PotGenFunc_AddSelfTerm_V (const NPME_Library::NPME_KfuncReal& funcLR, 
  const size_t nCharge, const double *Q, double *V)
//input:  Q[nCharge]
//output: V[nCharge][4]
{
  double fself;

  //Vself = funcLR(0,0,0)
  {
    double x[1] = {0};
    double y[1] = {0};
    double z[1] = {0};
    funcLR.Calc (1, x, y, z);
    fself = x[0];
  }
  for (size_t i = 0; i < nCharge; i++)
    V[4*i] += fself*Q[i];
}
void NPME_PotGenFunc_AddSelfTerm_V (
  const NPME_Library::NPME_KfuncComplex& funcLR, 
  const size_t nCharge, const _Complex double *Q, _Complex double *V)
//input:  Q[nCharge]
//output: V[nCharge][4]
{
  _Complex double fself;

  //Vself = funcLR(0,0,0)
  {
    double x_f0_r[1] = {0};
    double f0_i[1]   = {0};
    double y[1]      = {0};
    double z[1]      = {0};
    funcLR.Calc (1, x_f0_r, f0_i, y, z);
    fself = x_f0_r[0] + I*f0_i[0];

  }
  for (size_t i = 0; i < nCharge; i++)
    V[4*i] += fself*Q[i];
}








void NPME_PotGenFunc_ClusterElement_V1 (
  const NPME_Library::NPME_KfuncReal& func, 
  const NPME_Library::NPME_ClusterPair& cluster,
  const double *coord, const double *charge,
  double *V1, double *V2, int vecOption, size_t blockSize, bool zeroArray)
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
    memset(V1, 0, 4*nChargeA1*sizeof(double));
    memset(V2, 0, 4*nChargeA2*sizeof(double));
  }

  for (size_t k = 0; k < cluster.pairB.size(); k++)
  {
    const size_t nChargeB1    = cluster.pairB[k].nPointPerCell1;
    const size_t nChargeB2    = cluster.pairB[k].nPointPerCell2;
    const size_t startIndB1   = cluster.pairB[k].startPointIndex1;
    const size_t startIndB2   = cluster.pairB[k].startPointIndex2;
    const size_t locStartInd1 = startIndB1 - startIndA1;
    const size_t locStartInd2 = startIndB2 - startIndA2;

    const double *coordB1     = &coord[3*startIndB1];
    const double *coordB2     = &coord[3*startIndB2];
    const double *chargeB1    = &charge[startIndB1];
    const double *chargeB2    = &charge[startIndB2];
    double *VB1               = &V1[4*locStartInd1];
    double *VB2               = &V2[4*locStartInd2];

    bool zeroArrayB = 0;
    if (cluster.pairB[k].cellIndex1 != cluster.pairB[k].cellIndex2)
    {
      NPME_PotGenFunc_LargePair_V1 (func,
        nChargeB1, coordB1, chargeB1, 
        nChargeB2, coordB2, chargeB2, 
        VB1, VB2, vecOption, blockSize, zeroArrayB);
    }
    else
    {
      NPME_PotGenFunc_LargeSelf_V1 (func,
        nChargeB1, coordB1, chargeB1, VB1, 
        vecOption, blockSize, zeroArrayB);
    }
  }
}


void NPME_PotGenFunc_DirectSum_V1 (const NPME_Library::NPME_KfuncReal& func, 
  const size_t nCharge, const double *coord, const double *charge, 
  double *V, const int nProc, const int vecOption,
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster, 
  const size_t blockSize)
{
  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotGenFunc_DirectSum_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotGenFunc_DirectSum_V1\n";
    sprintf(str, "blockSize = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n",
      blockSize, NPME_Pot_MaxChgBlock_V1);
    std::cout << str;
    exit(0);
  }

  memset(V, 0, 4*nCharge*sizeof(double));

  const size_t nCluster = cluster.size();

  size_t k;
  #pragma omp parallel shared(V, charge, coord, cluster, func) private(k) default(none) num_threads(nProc)
  {
    #pragma omp for schedule(dynamic)
    for (k = 0; k < nCluster; k++)
    {
      bool zeroArray = 1;
      const size_t nChargeA1  = cluster[k].nPointPerCluster1;
      const size_t nChargeA2  = cluster[k].nPointPerCluster2;
      const size_t startIndA1 = cluster[k].pointStartA1;
      const size_t startIndA2 = cluster[k].pointStartA2;

      NPME_AlignedArrayDouble Vmem_1 (4*nChargeA1, 64);
      NPME_AlignedArrayDouble Vmem_2 (4*nChargeA2, 64);

      double *VA1             = &V[4*startIndA1];
      double *VA2             = &V[4*startIndA2];
      double *V1loc_1         = Vmem_1.GetPtr();
      double *V1loc_2         = Vmem_2.GetPtr();


      NPME_PotGenFunc_ClusterElement_V1 (func,
            cluster[k], coord, charge, V1loc_1, V1loc_2, 
            vecOption, blockSize, zeroArray);

      #pragma omp critical (update_NPME_PotGenFunc_DirectSum_V1_RealCluster)
      {
        for (size_t n = 0; n < 4*nChargeA1; n++)   VA1[n] += V1loc_1[n];
        for (size_t n = 0; n < 4*nChargeA2; n++)   VA2[n] += V1loc_2[n];
      }
    }
  }
}



void NPME_PotGenFunc_ClusterElement_V1 (
  const NPME_Library::NPME_KfuncComplex& func, 
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
      NPME_PotGenFunc_LargePair_V1 (func,
        nChargeB1, coordB1, chargeB1, 
        nChargeB2, coordB2, chargeB2, 
        VB1, VB2, vecOption, blockSize, zeroArrayB);
    }
    else
    {
      NPME_PotGenFunc_LargeSelf_V1 (func,
        nChargeB1, coordB1, chargeB1, VB1, 
        vecOption, blockSize, zeroArrayB);
    }
  }
}


void NPME_PotGenFunc_DirectSum_V1 (const NPME_Library::NPME_KfuncComplex& func, 
  const size_t nCharge, const double *coord, const _Complex double *charge, 
  _Complex double *V, const int nProc, const int vecOption,
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster, 
  const size_t blockSize)
{
  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotGenFunc_DirectSum_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotGenFunc_DirectSum_V1\n";
    sprintf(str, "blockSize = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n",
      blockSize, NPME_Pot_MaxChgBlock_V1);
    std::cout << str;
    exit(0);
  }

  memset(V, 0, 4*nCharge*sizeof(_Complex double));

  const size_t nCluster = cluster.size();

  size_t k;
  #pragma omp parallel shared(V, charge, coord, cluster, func) private(k) default(none) num_threads(nProc)
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


      NPME_PotGenFunc_ClusterElement_V1 (func,
            cluster[k], coord, charge, V1loc_1, V1loc_2, 
            vecOption, blockSize, zeroArray);

      #pragma omp critical (update_NPME_PotGenFunc_DirectSum_V1_ComplexCluster)
      {
        for (size_t n = 0; n < 4*nChargeA1; n++)   VA1[n] += V1loc_1[n];
        for (size_t n = 0; n < 4*nChargeA2; n++)   VA2[n] += V1loc_2[n];
      }
    }
  }
}

void NPME_PotGenFunc_MacroSelf_V1 (const NPME_Library::NPME_KfuncReal& func, 
  const size_t nCharge, const double *coord, const double *Q1, 
  double *V1, const int nProc, const int vecOption, 
  const size_t blockSize)
{
  if (blockSize%8 != 0)
  {
    printf("Error in NPME_PotGenFunc_MacroSelf_V1.\n");
    printf("blockSize = %lu is not a multiple of 8\n", blockSize);
    exit(0);
  }

  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_MacroSelf_V1.\n");
    printf("blockSize = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      blockSize, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  const size_t remain = nCharge%blockSize;
  size_t nBlock       = (nCharge-remain)/blockSize;
  if (remain > 0)
    nBlock++;

  memset(V1, 0, 4*nCharge*sizeof(double));



  const size_t nPair = (nBlock*(nBlock+1))/2;

  size_t k;
  #pragma omp parallel shared(V1, Q1, coord, nBlock, func) private(k) default(none) num_threads(nProc)
  {
    double V1loc_1 [4*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
    double V1loc_2 [4*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));

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

        const size_t index_i    = i*blockSize;
        const double *Q1_1      = &Q1[index_i];
        const double *coord1    = &coord[3*index_i];
        double *V1_1            = &V1[4*index_i];

        const size_t index_j    = j*blockSize;
        const double *Q1_2      = &Q1[index_j];
        const double *coord2    = &coord[3*index_j];
        double *V1_2            = &V1[4*index_j];

        NPME_PotGenFunc_Pair_V1 (func, 
          nCharge1, coord1, Q1_1, 
          nCharge2, coord2, Q1_2, 
          V1loc_1, V1loc_2, vecOption);

        #pragma omp critical (update_NPME_PotGenFunc_MacroSelf_V1_Real)
        {
          for (size_t n = 0; n < 4*nCharge1; n++)   V1_1[n] += V1loc_1[n];
          for (size_t n = 0; n < 4*nCharge2; n++)   V1_2[n] += V1loc_2[n];
        }
      }
      else
      {
        size_t nCharge1 = blockSize;

        if ( (remain > 0) && (i == nBlock - 1) )  nCharge1 = remain;

        const size_t index_i    = i*blockSize;
        const double *Q1_1      = &Q1[index_i];
        const double *coord1    = &coord[3*index_i];
        double *V1_1            = &V1[4*index_i];

        NPME_PotGenFunc_Self_V1 (func, nCharge1, 
          coord1, Q1_1, V1loc_1, vecOption);

        #pragma omp critical (update_NPME_PotGenFunc_MacroSelf_V1_Real)
        {
          for (size_t n = 0; n < 4*nCharge1; n++)   V1_1[n] += V1loc_1[n];
        }
      }
    }
  }
}

void NPME_PotGenFunc_MacroSelf_V1 (const NPME_Library::NPME_KfuncComplex& func, 
  const size_t nCharge, const double *coord, const _Complex double *Q1, 
  _Complex double *V1, const int nProc, const int vecOption, 
  const size_t blockSize)
{
  if (blockSize%8 != 0)
  {
    printf("Error in NPME_PotGenFunc_MacroSelf_V1.\n");
    printf("blockSize = %lu is not a multiple of 8\n", blockSize);
    exit(0);
  }

  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_MacroSelf_V1.\n");
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
  #pragma omp parallel shared(V1, Q1, coord, nBlock, func) private(k) default(none) num_threads(nProc)
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

        NPME_PotGenFunc_Pair_V1 (func, 
          nCharge1, coord1, Q1_1, 
          nCharge2, coord2, Q1_2, 
          V1loc_1, V1loc_2, vecOption);

        #pragma omp critical (update_NPME_PotGenFunc_MacroSelf_V1_Complex)
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

        NPME_PotGenFunc_Self_V1 (func, nCharge1, 
          coord1, Q1_1, V1loc_1, vecOption);

        #pragma omp critical (update_NPME_PotGenFunc_MacroSelf_V1_Complex)
        {
          for (size_t n = 0; n < 4*nCharge1; n++)   V1_1[n] += V1loc_1[n];
        }
      }
    }
  }
}





//******************************************************************************
//******************************************************************************
//******************************************************************************
//************************Real Low Level Functions******************************
//******************************************************************************
//******************************************************************************
//******************************************************************************

void NPME_PotGenFunc_Pair_V1 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        func(x,y,z)
//output: V1[nCharge1][4]
//        V2[nCharge2][4]
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_Pair_V1.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  if (zeroArray)
  {
    memset(V1, 0, 4*nCharge1*sizeof(double));
    memset(V2, 0, 4*nCharge2*sizeof(double));
  }

  double f0[NPME_Pot_MaxChgBlock_V1];
  double fX[NPME_Pot_MaxChgBlock_V1];
  double fY[NPME_Pot_MaxChgBlock_V1];
  double fZ[NPME_Pot_MaxChgBlock_V1];

  for (size_t i = 0; i < nCharge1; i++)
  {
    const double x1 = coord1[3*i  ];
    const double y1 = coord1[3*i+1];
    const double z1 = coord1[3*i+2];

    size_t count2 = 0;
    for (size_t j = 0; j < nCharge2; j++)
    {
      fX[j] = x1 - coord2[count2  ];
      fY[j] = y1 - coord2[count2+1];
      fZ[j] = z1 - coord2[count2+2];
      count2 += 3;
    }

    func.Calc(nCharge2, f0, fX, fY, fZ);

    for (size_t j = 0; j < nCharge2; j++)
    {
      V1[4*i  ] += f0[j]*q2[j];
      V1[4*i+1] += fX[j]*q2[j];
      V1[4*i+2] += fY[j]*q2[j];
      V1[4*i+3] += fZ[j]*q2[j];

      V2[4*j  ] += f0[j]*q1[i];
      V2[4*j+1] -= fX[j]*q1[i];
      V2[4*j+2] -= fY[j]*q1[i];
      V2[4*j+3] -= fZ[j]*q1[i];
    }
  }
}

void NPME_PotGenFunc_Self_V1 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge, const double *coord, const double *q, double *V, 
  bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[nCharge][4]
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_Self_V1.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  if (zeroArray)
    memset(V, 0, 4*nCharge*sizeof(double));


  double f0[NPME_Pot_MaxChgBlock_V1];
  double fX[NPME_Pot_MaxChgBlock_V1];
  double fY[NPME_Pot_MaxChgBlock_V1];
  double fZ[NPME_Pot_MaxChgBlock_V1];

  for (size_t i = 0; i < nCharge; i++)
  {
    const double x1 = coord[3*i  ];
    const double y1 = coord[3*i+1];
    const double z1 = coord[3*i+2];

    size_t count2 = 0;
    for (size_t j = 0; j < i; j++)
    {
      fX[j] = x1 - coord[count2  ];
      fY[j] = y1 - coord[count2+1];
      fZ[j] = z1 - coord[count2+2];
      count2 += 3;
    }

    func.Calc(i, f0, fX, fY, fZ);

    for (size_t j = 0; j < i; j++)
    {
      V[4*i  ] += f0[j]*q[j];
      V[4*i+1] += fX[j]*q[j];
      V[4*i+2] += fY[j]*q[j];
      V[4*i+3] += fZ[j]*q[j];

      V[4*j  ] += f0[j]*q[i];
      V[4*j+1] -= fX[j]*q[i];
      V[4*j+2] -= fY[j]*q[i];
      V[4*j+3] -= fZ[j]*q[i];
    }
  }
}





#if NPME_USE_AVX


void NPME_PotGenFunc_Pair_V1_AVX (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        nCharge2 must be a multiple of 4 and <= NPME_Pot_MaxChgBlock_V1
//output: V1[nCharge1][4]
//        V2[nCharge2][4]
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_Pair_V1_AVX.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  const size_t remain2    = nCharge2%4;
  const size_t nLoop2     = (nCharge2 - remain2)/4;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  const size_t nCharge2Loop = 4*nLoop2wRemainder;


  double coord2_4x[3*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformCoord_4x_AVX (nCharge2, coord2, coord2_4x);

  double q2_4x[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformQreal_4x_AVX (nCharge2, q2, q2_4x);

  //tmp aligned arrays for potential 2
  double V2_4x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_4x, 0, 16*nLoop2wRemainder*sizeof(double));

  double f0[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fX[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fY[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fZ[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));

  for (size_t i = 0; i < nCharge1; i++)
  {
    const double *r1      = &coord1[3*i];
    const __m256d q1Vec   = _mm256_set1_pd( q1[i]);
    const __m256d q1NVec  = _mm256_set1_pd(-q1[i]);
    const __m256d x1Vec   = _mm256_set1_pd(r1[0]);
    const __m256d y1Vec   = _mm256_set1_pd(r1[1]);
    const __m256d z1Vec   = _mm256_set1_pd(r1[2]);

    __m256d V0_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VX_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VY_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VZ_1_Vec      = _mm256_set1_pd(0.0);

    size_t index4j, index12j, index16j;

    index4j  = 0;
    index12j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      __m256d xVec  = _mm256_load_pd (&coord2_4x[index12j  ]);
      __m256d yVec  = _mm256_load_pd (&coord2_4x[index12j+4]);
      __m256d zVec  = _mm256_load_pd (&coord2_4x[index12j+8]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      _mm256_store_pd (&fX[index4j], xVec);
      _mm256_store_pd (&fY[index4j], yVec);
      _mm256_store_pd (&fZ[index4j], zVec);

      index4j  += 4;
      index12j += 12;
    }

    func.CalcAVX (nCharge2Loop, f0, fX, fY, fZ);

    index4j  = 0;
    index16j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      const __m256d q2Vec  = _mm256_load_pd  (&q2_4x[index4j]);
      const __m256d f0_Vec = _mm256_load_pd  (&f0[index4j]);
      const __m256d fX_Vec = _mm256_load_pd  (&fX[index4j]);
      const __m256d fY_Vec = _mm256_load_pd  (&fY[index4j]);
      const __m256d fZ_Vec = _mm256_load_pd  (&fZ[index4j]);

      __m256d V0_2_Vec  = _mm256_load_pd (&V2_4x[index16j   ]);
      __m256d VX_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 4]);
      __m256d VY_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 8]);
      __m256d VZ_2_Vec  = _mm256_load_pd (&V2_4x[index16j+12]);

      #if NPME_USE_AVX_FMA
      {
        //V1 contribution
        V0_1_Vec  = _mm256_fmadd_pd  (f0_Vec, q2Vec, V0_1_Vec);
        VX_1_Vec  = _mm256_fmadd_pd  (fX_Vec, q2Vec, VX_1_Vec);
        VY_1_Vec  = _mm256_fmadd_pd  (fY_Vec, q2Vec, VY_1_Vec);
        VZ_1_Vec  = _mm256_fmadd_pd  (fZ_Vec, q2Vec, VZ_1_Vec);

        //V2 contribution
        V0_2_Vec  = _mm256_fmadd_pd  (f0_Vec, q1Vec,  V0_2_Vec);
        VX_2_Vec  = _mm256_fmadd_pd  (fX_Vec, q1NVec, VX_2_Vec);
        VY_2_Vec  = _mm256_fmadd_pd  (fY_Vec, q1NVec, VY_2_Vec);
        VZ_2_Vec  = _mm256_fmadd_pd  (fZ_Vec, q1NVec, VZ_2_Vec);
      }
      #else
      {
        //V1 contribution
        V0_1_Vec  = _mm256_add_pd  (_mm256_mul_pd (f0_Vec, q2Vec), V0_1_Vec);
        VX_1_Vec  = _mm256_add_pd  (_mm256_mul_pd (fX_Vec, q2Vec), VX_1_Vec);
        VY_1_Vec  = _mm256_add_pd  (_mm256_mul_pd (fY_Vec, q2Vec), VY_1_Vec);
        VZ_1_Vec  = _mm256_add_pd  (_mm256_mul_pd (fZ_Vec, q2Vec), VZ_1_Vec);

        //V2 contribution
        V0_2_Vec  = _mm256_add_pd  (_mm256_mul_pd (f0_Vec, q1Vec),  V0_2_Vec);
        VX_2_Vec  = _mm256_add_pd  (_mm256_mul_pd (fX_Vec, q1NVec), VX_2_Vec);
        VY_2_Vec  = _mm256_add_pd  (_mm256_mul_pd (fY_Vec, q1NVec), VY_2_Vec);
        VZ_2_Vec  = _mm256_add_pd  (_mm256_mul_pd (fZ_Vec, q1NVec), VZ_2_Vec);
      }
      #endif

      _mm256_store_pd (&V2_4x[index16j   ], V0_2_Vec);
      _mm256_store_pd (&V2_4x[index16j+ 4], VX_2_Vec);
      _mm256_store_pd (&V2_4x[index16j+ 8], VY_2_Vec);
      _mm256_store_pd (&V2_4x[index16j+12], VZ_2_Vec);

      index4j  += 4;
      index16j += 16;
    }

    //accumulate and store V1 contributions
    double sum[4];
    NPME_mm256_4x4HorizontalSum_pd (sum, V0_1_Vec, VX_1_Vec, 
                                         VY_1_Vec, VZ_1_Vec);

    if (zeroArray)
    {
      V1[4*i  ]  = sum[0];
      V1[4*i+1]  = sum[1];
      V1[4*i+2]  = sum[2];
      V1[4*i+3]  = sum[3];
    }
    else
    {
      V1[4*i  ]  += sum[0];
      V1[4*i+1]  += sum[1];
      V1[4*i+2]  += sum[2];
      V1[4*i+3]  += sum[3];
    }
  }

  if (zeroArray)
    NPME_TransformRealV1_4x_2_V1_AVX (nCharge2, V2_4x, V2);
  else
    NPME_TransformUpdateRealV1_4x_2_V1_AVX (nCharge2, V2_4x, V2);
}




void NPME_PotGenFunc_Self_V1_AVX (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge, const double *coord, const double *q, double *V,
  bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[nCharge][4]
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_Self_V1_AVX.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  double coord_4x[3*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformCoord_4x_AVX (nCharge, coord, coord_4x);

  double q_4x[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformQreal_4x_AVX (nCharge, q, q_4x);

  //tmp aligned arrays for potential 2
  double V2_4x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  {
    const size_t remain    = nCharge%4;
    const size_t nLoop     = (nCharge - remain)/4;
    size_t nLoopwRemainder = nLoop;
    if (remain > 0)
      nLoopwRemainder++;
    memset(V2_4x, 0, 16*nLoopwRemainder*sizeof(double));
  }

  double f0[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fX[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fY[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fZ[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));

  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r1      = &coord[3*i];
    __m256d q1Vec         = _mm256_set1_pd( q[i]);
    __m256d q1NVec        = _mm256_set1_pd(-q[i]);
    const __m256d x1Vec   = _mm256_set1_pd(r1[0]);
    const __m256d y1Vec   = _mm256_set1_pd(r1[1]);
    const __m256d z1Vec   = _mm256_set1_pd(r1[2]);

    __m256d V0_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VX_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VY_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VZ_1_Vec      = _mm256_set1_pd(0.0);

    const size_t remainInner   = (i)%4;
    const size_t nLoopInner    = (i-remainInner)/4;
    size_t nLoopInnerwRemainder = nLoopInner;
    if (remainInner > 0)
      nLoopInnerwRemainder++;

    size_t index4j, index12j, index16j;

    index4j  = 0;
    index12j = 0;
    for (size_t j = 0; j < nLoopInner; j++)
    {
      __m256d xVec  = _mm256_load_pd (&coord_4x[index12j  ]);
      __m256d yVec  = _mm256_load_pd (&coord_4x[index12j+4]);
      __m256d zVec  = _mm256_load_pd (&coord_4x[index12j+8]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      _mm256_store_pd (&fX[index4j], xVec);
      _mm256_store_pd (&fY[index4j], yVec);
      _mm256_store_pd (&fZ[index4j], zVec);

      index4j  += 4;
      index12j += 12;
    }

    if (remainInner > 0)
    {
      const size_t indexStart = 4*nLoopInner;
      const double *crdLoc    = &coord[3*indexStart];
      const double X          = r1[0]+1.0;
      double x2Array[4]  __attribute__((aligned(64))) = {X, X, X, X};
      double y2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double z2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};

      for (size_t k = 0; k < remainInner; k++)
      {
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
      }

      __m256d xVec      = _mm256_load_pd (x2Array);
      __m256d yVec      = _mm256_load_pd (y2Array);
      __m256d zVec      = _mm256_load_pd (z2Array);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      _mm256_store_pd (&fX[index4j], xVec);
      _mm256_store_pd (&fY[index4j], yVec);
      _mm256_store_pd (&fZ[index4j], zVec);
    }

    func.CalcAVX (4*nLoopInnerwRemainder, f0, fX, fY, fZ);

    index4j  = 0;
    index16j = 0;
    for (size_t j = 0; j < nLoopInner; j++)
    {
      const __m256d q2Vec  = _mm256_load_pd  (&q_4x[index4j]);
      const __m256d f0_Vec = _mm256_load_pd  (&f0[index4j]);
      const __m256d fX_Vec = _mm256_load_pd  (&fX[index4j]);
      const __m256d fY_Vec = _mm256_load_pd  (&fY[index4j]);
      const __m256d fZ_Vec = _mm256_load_pd  (&fZ[index4j]);

      __m256d V0_2_Vec  = _mm256_load_pd (&V2_4x[index16j   ]);
      __m256d VX_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 4]);
      __m256d VY_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 8]);
      __m256d VZ_2_Vec  = _mm256_load_pd (&V2_4x[index16j+12]);

      #if NPME_USE_AVX_FMA
      {
        //V1 contribution
        V0_1_Vec  = _mm256_fmadd_pd  (f0_Vec, q2Vec, V0_1_Vec);
        VX_1_Vec  = _mm256_fmadd_pd  (fX_Vec, q2Vec, VX_1_Vec);
        VY_1_Vec  = _mm256_fmadd_pd  (fY_Vec, q2Vec, VY_1_Vec);
        VZ_1_Vec  = _mm256_fmadd_pd  (fZ_Vec, q2Vec, VZ_1_Vec);

        //V2 contribution
        V0_2_Vec  = _mm256_fmadd_pd  (f0_Vec, q1Vec,  V0_2_Vec);
        VX_2_Vec  = _mm256_fmadd_pd  (fX_Vec, q1NVec, VX_2_Vec);
        VY_2_Vec  = _mm256_fmadd_pd  (fY_Vec, q1NVec, VY_2_Vec);
        VZ_2_Vec  = _mm256_fmadd_pd  (fZ_Vec, q1NVec, VZ_2_Vec);
      }
      #else
      {
        //V1 contribution
        V0_1_Vec  = _mm256_add_pd  (_mm256_mul_pd (f0_Vec, q2Vec), V0_1_Vec);
        VX_1_Vec  = _mm256_add_pd  (_mm256_mul_pd (fX_Vec, q2Vec), VX_1_Vec);
        VY_1_Vec  = _mm256_add_pd  (_mm256_mul_pd (fY_Vec, q2Vec), VY_1_Vec);
        VZ_1_Vec  = _mm256_add_pd  (_mm256_mul_pd (fZ_Vec, q2Vec), VZ_1_Vec);

        //V2 contribution
        V0_2_Vec  = _mm256_add_pd  (_mm256_mul_pd (f0_Vec, q1Vec),  V0_2_Vec);
        VX_2_Vec  = _mm256_add_pd  (_mm256_mul_pd (fX_Vec, q1NVec), VX_2_Vec);
        VY_2_Vec  = _mm256_add_pd  (_mm256_mul_pd (fY_Vec, q1NVec), VY_2_Vec);
        VZ_2_Vec  = _mm256_add_pd  (_mm256_mul_pd (fZ_Vec, q1NVec), VZ_2_Vec);
      }
      #endif

      _mm256_store_pd (&V2_4x[index16j   ], V0_2_Vec);
      _mm256_store_pd (&V2_4x[index16j+ 4], VX_2_Vec);
      _mm256_store_pd (&V2_4x[index16j+ 8], VY_2_Vec);
      _mm256_store_pd (&V2_4x[index16j+12], VZ_2_Vec);

      index4j  += 4;
      index16j += 16;
    }
    if (remainInner > 0)
    {
      const size_t indexStart = 4*nLoopInner;
      const double *qLoc      = &q[indexStart];
      double q2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double mskArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};

      for (size_t k = 0; k < remainInner; k++)
      {
        q2Array[k]  = qLoc[k];
        mskArray[k] = 1.0;
      }
      __m256d q2Vec     = _mm256_load_pd (q2Array);
      __m256d mskVec    = _mm256_load_pd (mskArray);

      __m256d f0_Vec    = _mm256_load_pd  (&f0[index4j]);
      __m256d fX_Vec    = _mm256_load_pd  (&fX[index4j]);
      __m256d fY_Vec    = _mm256_load_pd  (&fY[index4j]);
      __m256d fZ_Vec    = _mm256_load_pd  (&fZ[index4j]);

      //apply mask
      q1Vec             = _mm256_mul_pd  (mskVec, q1Vec);
      q1NVec            = _mm256_mul_pd  (mskVec, q1NVec);

      __m256d V0_2_Vec  = _mm256_load_pd (&V2_4x[index16j   ]);
      __m256d VX_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 4]);
      __m256d VY_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 8]);
      __m256d VZ_2_Vec  = _mm256_load_pd (&V2_4x[index16j+12]);

      #if NPME_USE_AVX_FMA
      {
        //V1 contribution
        V0_1_Vec  = _mm256_fmadd_pd  (f0_Vec, q2Vec, V0_1_Vec);
        VX_1_Vec  = _mm256_fmadd_pd  (fX_Vec, q2Vec, VX_1_Vec);
        VY_1_Vec  = _mm256_fmadd_pd  (fY_Vec, q2Vec, VY_1_Vec);
        VZ_1_Vec  = _mm256_fmadd_pd  (fZ_Vec, q2Vec, VZ_1_Vec);

        //V2 contribution
        V0_2_Vec  = _mm256_fmadd_pd  (f0_Vec, q1Vec,  V0_2_Vec);
        VX_2_Vec  = _mm256_fmadd_pd  (fX_Vec, q1NVec, VX_2_Vec);
        VY_2_Vec  = _mm256_fmadd_pd  (fY_Vec, q1NVec, VY_2_Vec);
        VZ_2_Vec  = _mm256_fmadd_pd  (fZ_Vec, q1NVec, VZ_2_Vec);
      }
      #else
      {
        //V1 contribution
        V0_1_Vec  = _mm256_add_pd  (_mm256_mul_pd (f0_Vec, q2Vec), V0_1_Vec);
        VX_1_Vec  = _mm256_add_pd  (_mm256_mul_pd (fX_Vec, q2Vec), VX_1_Vec);
        VY_1_Vec  = _mm256_add_pd  (_mm256_mul_pd (fY_Vec, q2Vec), VY_1_Vec);
        VZ_1_Vec  = _mm256_add_pd  (_mm256_mul_pd (fZ_Vec, q2Vec), VZ_1_Vec);

        //V2 contribution
        V0_2_Vec  = _mm256_add_pd  (_mm256_mul_pd (f0_Vec, q1Vec),  V0_2_Vec);
        VX_2_Vec  = _mm256_add_pd  (_mm256_mul_pd (fX_Vec, q1NVec), VX_2_Vec);
        VY_2_Vec  = _mm256_add_pd  (_mm256_mul_pd (fY_Vec, q1NVec), VY_2_Vec);
        VZ_2_Vec  = _mm256_add_pd  (_mm256_mul_pd (fZ_Vec, q1NVec), VZ_2_Vec);
      }
      #endif

      _mm256_store_pd (&V2_4x[index16j   ], V0_2_Vec);
      _mm256_store_pd (&V2_4x[index16j+ 4], VX_2_Vec);
      _mm256_store_pd (&V2_4x[index16j+ 8], VY_2_Vec);
      _mm256_store_pd (&V2_4x[index16j+12], VZ_2_Vec);
    }


    //accumulate and store V1 contributions
    double sum[4];
    NPME_mm256_4x4HorizontalSum_pd (sum, V0_1_Vec, VX_1_Vec, 
                                         VY_1_Vec, VZ_1_Vec);

    if (zeroArray)
    {
      V[4*i  ]  = sum[0];
      V[4*i+1]  = sum[1];
      V[4*i+2]  = sum[2];
      V[4*i+3]  = sum[3];
    }
    else
    {
      V[4*i  ]  += sum[0];
      V[4*i+1]  += sum[1];
      V[4*i+2]  += sum[2];
      V[4*i+3]  += sum[3];
    }
  }

  NPME_TransformUpdateRealV1_4x_2_V1_AVX (nCharge, V2_4x, V);
}

#endif

#if NPME_USE_AVX_512


void NPME_PotGenFunc_Pair_V1_AVX_512 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        nCharge2 must be a multiple of 4 and <= NPME_Pot_MaxChgBlock_V1
//output: V1[nCharge1][4]
//        V2[nCharge2][4]
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_Pair_V1_AVX_512.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  const size_t remain2    = nCharge2%8;
  const size_t nLoop2     = (nCharge2 - remain2)/8;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  const size_t nCharge2Loop = 8*nLoop2wRemainder;


  double coord2_8x[3*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformCoord_8x_AVX_512 (nCharge2, coord2, coord2_8x);

  double q2_8x[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformQreal_8x_AVX_512 (nCharge2, q2, q2_8x);

  //tmp aligned arrays for potential 2
  double V2_8x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_8x, 0, 32*nLoop2wRemainder*sizeof(double));

  double f0[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fX[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fY[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fZ[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));

  for (size_t i = 0; i < nCharge1; i++)
  {
    const double *r1      = &coord1[3*i];
    const __m512d q1Vec   = _mm512_set1_pd( q1[i]);
    const __m512d q1NVec  = _mm512_set1_pd(-q1[i]);
    const __m512d x1Vec   = _mm512_set1_pd(r1[0]);
    const __m512d y1Vec   = _mm512_set1_pd(r1[1]);
    const __m512d z1Vec   = _mm512_set1_pd(r1[2]);

    __m512d V0_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VX_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VY_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VZ_1_Vec      = _mm512_set1_pd(0.0);

    size_t index8j, index24j, index32j;

    index8j  = 0;
    index24j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      __m512d xVec  = _mm512_load_pd (&coord2_8x[index24j   ]);
      __m512d yVec  = _mm512_load_pd (&coord2_8x[index24j+ 8]);
      __m512d zVec  = _mm512_load_pd (&coord2_8x[index24j+16]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      _mm512_store_pd (&fX[index8j], xVec);
      _mm512_store_pd (&fY[index8j], yVec);
      _mm512_store_pd (&fZ[index8j], zVec);

      index8j  += 8;
      index24j += 24;
    }

    func.CalcAVX_512 (nCharge2Loop, f0, fX, fY, fZ);

    index8j  = 0;
    index32j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      const __m512d q2Vec  = _mm512_load_pd  (&q2_8x[index8j]);
      const __m512d f0_Vec = _mm512_load_pd  (&f0[index8j]);
      const __m512d fX_Vec = _mm512_load_pd  (&fX[index8j]);
      const __m512d fY_Vec = _mm512_load_pd  (&fY[index8j]);
      const __m512d fZ_Vec = _mm512_load_pd  (&fZ[index8j]);

      __m512d V0_2_Vec  = _mm512_load_pd (&V2_8x[index32j   ]);
      __m512d VX_2_Vec  = _mm512_load_pd (&V2_8x[index32j+ 8]);
      __m512d VY_2_Vec  = _mm512_load_pd (&V2_8x[index32j+16]);
      __m512d VZ_2_Vec  = _mm512_load_pd (&V2_8x[index32j+24]);

      //V1 contribution
      V0_1_Vec  = _mm512_fmadd_pd  (f0_Vec, q2Vec, V0_1_Vec);
      VX_1_Vec  = _mm512_fmadd_pd  (fX_Vec, q2Vec, VX_1_Vec);
      VY_1_Vec  = _mm512_fmadd_pd  (fY_Vec, q2Vec, VY_1_Vec);
      VZ_1_Vec  = _mm512_fmadd_pd  (fZ_Vec, q2Vec, VZ_1_Vec);

      //V2 contribution
      V0_2_Vec  = _mm512_fmadd_pd  (f0_Vec, q1Vec,  V0_2_Vec);
      VX_2_Vec  = _mm512_fmadd_pd  (fX_Vec, q1NVec, VX_2_Vec);
      VY_2_Vec  = _mm512_fmadd_pd  (fY_Vec, q1NVec, VY_2_Vec);
      VZ_2_Vec  = _mm512_fmadd_pd  (fZ_Vec, q1NVec, VZ_2_Vec);

      _mm512_store_pd (&V2_8x[index32j   ], V0_2_Vec);
      _mm512_store_pd (&V2_8x[index32j+ 8], VX_2_Vec);
      _mm512_store_pd (&V2_8x[index32j+16], VY_2_Vec);
      _mm512_store_pd (&V2_8x[index32j+24], VZ_2_Vec);

      index8j  += 8;
      index32j += 32;
    }

    //accumulate and store V1 contributions
    double sum[4];
    NPME_mm512_4x8HorizontalSum_pd (sum, V0_1_Vec, VX_1_Vec, 
                                         VY_1_Vec, VZ_1_Vec);

    if (zeroArray)
    {
      V1[4*i  ]  = sum[0];
      V1[4*i+1]  = sum[1];
      V1[4*i+2]  = sum[2];
      V1[4*i+3]  = sum[3];
    }
    else
    {
      V1[4*i  ]  += sum[0];
      V1[4*i+1]  += sum[1];
      V1[4*i+2]  += sum[2];
      V1[4*i+3]  += sum[3];
    }
  }

  if (zeroArray)
    NPME_TransformRealV1_8x_2_V1_AVX_512 (nCharge2, V2_8x, V2);
  else
    NPME_TransformUpdateRealV1_8x_2_V1_AVX_512 (nCharge2, V2_8x, V2);
}





void NPME_PotGenFunc_Self_V1_AVX_512 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge, const double *coord, const double *q, double *V, 
  bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[nCharge][4]
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_Self_V1_AVX_512.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  double coord_8x[3*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformCoord_8x_AVX_512 (nCharge, coord, coord_8x);

  double q_8x[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformQreal_8x_AVX_512 (nCharge, q, q_8x);

  //tmp aligned arrays for potential 2
  double V2_8x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  {
    const size_t remain    = nCharge%8;
    const size_t nLoop     = (nCharge - remain)/8;
    size_t nLoopwRemainder = nLoop;
    if (remain > 0)
      nLoopwRemainder++;
    memset(V2_8x, 0, 32*nLoopwRemainder*sizeof(double));
  }

  double f0[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fX[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fY[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fZ[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));

  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r1      = &coord[3*i];
    __m512d q1Vec         = _mm512_set1_pd( q[i]);
    __m512d q1NVec        = _mm512_set1_pd(-q[i]);
    const __m512d x1Vec   = _mm512_set1_pd(r1[0]);
    const __m512d y1Vec   = _mm512_set1_pd(r1[1]);
    const __m512d z1Vec   = _mm512_set1_pd(r1[2]);

    __m512d V0_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VX_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VY_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VZ_1_Vec      = _mm512_set1_pd(0.0);

    const size_t remainInner   = (i)%8;
    const size_t nLoopInner    = (i-remainInner)/8;
    size_t nLoopInnerwRemainder = nLoopInner;
    if (remainInner > 0)
      nLoopInnerwRemainder++;

    size_t index8j, index24j, index32j;

    index8j  = 0;
    index24j = 0;
    for (size_t j = 0; j < nLoopInner; j++)
    {
      __m512d xVec  = _mm512_load_pd (&coord_8x[index24j  ]);
      __m512d yVec  = _mm512_load_pd (&coord_8x[index24j+8]);
      __m512d zVec  = _mm512_load_pd (&coord_8x[index24j+16]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      _mm512_store_pd (&fX[index8j], xVec);
      _mm512_store_pd (&fY[index8j], yVec);
      _mm512_store_pd (&fZ[index8j], zVec);

      index8j  += 8;
      index24j += 24;
    }

    if (remainInner > 0)
    {
      const size_t indexStart = 8*nLoopInner;
      const double *crdLoc    = &coord[3*indexStart];
      const double X          = r1[0]+1.0;
      double x2Array[8] __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};
      double y2Array[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
      double z2Array[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};

      for (size_t k = 0; k < remainInner; k++)
      {
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
      }

      __m512d xVec      = _mm512_load_pd (x2Array);
      __m512d yVec      = _mm512_load_pd (y2Array);
      __m512d zVec      = _mm512_load_pd (z2Array);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      _mm512_store_pd (&fX[index8j], xVec);
      _mm512_store_pd (&fY[index8j], yVec);
      _mm512_store_pd (&fZ[index8j], zVec);
    }

    func.CalcAVX_512 (8*nLoopInnerwRemainder, f0, fX, fY, fZ);

    index8j  = 0;
    index32j = 0;
    for (size_t j = 0; j < nLoopInner; j++)
    {
      const __m512d q2Vec  = _mm512_loadu_pd (&q[index8j]);
      const __m512d f0_Vec = _mm512_load_pd  (&f0[index8j]);
      const __m512d fX_Vec = _mm512_load_pd  (&fX[index8j]);
      const __m512d fY_Vec = _mm512_load_pd  (&fY[index8j]);
      const __m512d fZ_Vec = _mm512_load_pd  (&fZ[index8j]);

      __m512d V0_2_Vec  = _mm512_load_pd (&V2_8x[index32j   ]);
      __m512d VX_2_Vec  = _mm512_load_pd (&V2_8x[index32j+ 8]);
      __m512d VY_2_Vec  = _mm512_load_pd (&V2_8x[index32j+16]);
      __m512d VZ_2_Vec  = _mm512_load_pd (&V2_8x[index32j+24]);

      //V1 contribution
      V0_1_Vec  = _mm512_fmadd_pd  (f0_Vec, q2Vec, V0_1_Vec);
      VX_1_Vec  = _mm512_fmadd_pd  (fX_Vec, q2Vec, VX_1_Vec);
      VY_1_Vec  = _mm512_fmadd_pd  (fY_Vec, q2Vec, VY_1_Vec);
      VZ_1_Vec  = _mm512_fmadd_pd  (fZ_Vec, q2Vec, VZ_1_Vec);

      //V2 contribution
      V0_2_Vec  = _mm512_fmadd_pd  (f0_Vec, q1Vec,  V0_2_Vec);
      VX_2_Vec  = _mm512_fmadd_pd  (fX_Vec, q1NVec, VX_2_Vec);
      VY_2_Vec  = _mm512_fmadd_pd  (fY_Vec, q1NVec, VY_2_Vec);
      VZ_2_Vec  = _mm512_fmadd_pd  (fZ_Vec, q1NVec, VZ_2_Vec);

      _mm512_store_pd (&V2_8x[index32j   ], V0_2_Vec);
      _mm512_store_pd (&V2_8x[index32j+ 8], VX_2_Vec);
      _mm512_store_pd (&V2_8x[index32j+16], VY_2_Vec);
      _mm512_store_pd (&V2_8x[index32j+24], VZ_2_Vec);

      index8j  += 8;
      index32j += 32;
    }
    if (remainInner > 0)
    {
      const size_t indexStart = 8*nLoopInner;
      const double *qLoc      = &q[indexStart];
      double q2Array[8]  __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};
      double mskArray[8] __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};

      for (size_t k = 0; k < remainInner; k++)
      {
        q2Array[k]  = qLoc[k];
        mskArray[k] = 1.0;
      }
      __m512d q2Vec     = _mm512_load_pd (q2Array);
      __m512d mskVec    = _mm512_load_pd (mskArray);

      __m512d f0_Vec    = _mm512_load_pd  (&f0[index8j]);
      __m512d fX_Vec    = _mm512_load_pd  (&fX[index8j]);
      __m512d fY_Vec    = _mm512_load_pd  (&fY[index8j]);
      __m512d fZ_Vec    = _mm512_load_pd  (&fZ[index8j]);

      //apply mask
      q1Vec             = _mm512_mul_pd  (mskVec, q1Vec);
      q1NVec            = _mm512_mul_pd  (mskVec, q1NVec);

      __m512d V0_2_Vec  = _mm512_load_pd (&V2_8x[index32j   ]);
      __m512d VX_2_Vec  = _mm512_load_pd (&V2_8x[index32j+ 8]);
      __m512d VY_2_Vec  = _mm512_load_pd (&V2_8x[index32j+16]);
      __m512d VZ_2_Vec  = _mm512_load_pd (&V2_8x[index32j+24]);

      //V1 contribution
      V0_1_Vec  = _mm512_fmadd_pd  (f0_Vec, q2Vec, V0_1_Vec);
      VX_1_Vec  = _mm512_fmadd_pd  (fX_Vec, q2Vec, VX_1_Vec);
      VY_1_Vec  = _mm512_fmadd_pd  (fY_Vec, q2Vec, VY_1_Vec);
      VZ_1_Vec  = _mm512_fmadd_pd  (fZ_Vec, q2Vec, VZ_1_Vec);

      //V2 contribution
      V0_2_Vec  = _mm512_fmadd_pd  (f0_Vec, q1Vec,  V0_2_Vec);
      VX_2_Vec  = _mm512_fmadd_pd  (fX_Vec, q1NVec, VX_2_Vec);
      VY_2_Vec  = _mm512_fmadd_pd  (fY_Vec, q1NVec, VY_2_Vec);
      VZ_2_Vec  = _mm512_fmadd_pd  (fZ_Vec, q1NVec, VZ_2_Vec);

      _mm512_store_pd (&V2_8x[index32j   ], V0_2_Vec);
      _mm512_store_pd (&V2_8x[index32j+ 8], VX_2_Vec);
      _mm512_store_pd (&V2_8x[index32j+16], VY_2_Vec);
      _mm512_store_pd (&V2_8x[index32j+24], VZ_2_Vec);
    }


    //accumulate and store V1 contributions
    double sum[4];
    NPME_mm512_4x8HorizontalSum_pd (sum, V0_1_Vec, VX_1_Vec, 
                                         VY_1_Vec, VZ_1_Vec);

    if (zeroArray)
    {
      V[4*i  ]  = sum[0];
      V[4*i+1]  = sum[1];
      V[4*i+2]  = sum[2];
      V[4*i+3]  = sum[3];
    }
    else
    {
      V[4*i  ]  += sum[0];
      V[4*i+1]  += sum[1];
      V[4*i+2]  += sum[2];
      V[4*i+3]  += sum[3];
    }
  }

  NPME_TransformUpdateRealV1_8x_2_V1_AVX_512 (nCharge, V2_8x, V);
}

#endif

void NPME_PotGenFunc_Pair_V1 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, int vecOption, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[nCharge1][4]
//        V2[nCharge2][4]
{
  if (vecOption == 0)
  {
    NPME_PotGenFunc_Pair_V1 (func,
      nCharge1, coord1, q1,
      nCharge2, coord2, q2, V1, V2, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotGenFunc_Pair_V1_AVX (func,
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotGenFunc_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotGenFunc_Pair_V1_AVX_512 (func,
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotGenFunc_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}



void NPME_PotGenFunc_Self_V1 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge, const double *coord, const double *q, 
  double *V, int vecOption, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V[nCharge][4]
{
  if (vecOption == 0)
  {
    NPME_PotGenFunc_Self_V1 (func,
      nCharge, coord, q, V, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotGenFunc_Self_V1_AVX (func,
        nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotGenFunc_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotGenFunc_Self_V1_AVX_512 (func,
        nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotGenFunc_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}


void NPME_PotGenFunc_LargePair_V1 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge1, const double *coord1, const double *charge1,
  const size_t nCharge2, const double *coord2, const double *charge2,
  double *V1, double *V2, int vecOption, size_t blockSize, bool zeroArray)
{
  if (nCharge2 < blockSize)
    return NPME_PotGenFunc_Pair_V1 (func,
              nCharge1, coord1, charge1,
              nCharge2, coord2, charge2, 
              V1, V2, vecOption, zeroArray);

  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotGenFunc_LargePair_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotGenFunc_LargePair_V1\n";
    sprintf(str, "blockSize = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n",
      blockSize, NPME_Pot_MaxChgBlock_V1);
    std::cout << str;
    exit(0);
  }

  if (zeroArray)
  {
    memset(V1, 0, 4*nCharge1*sizeof(double));
    memset(V2, 0, 4*nCharge2*sizeof(double));
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

    const double *charge1B  = &charge1[start1B];
    const double *coord1B   = &coord1[3*start1B];
    double *V1_B            = &V1[4*start1B];

    const double *charge2B  = &charge2[start2B];
    const double *coord2B   = &coord2[3*start2B];
    double *V2_B            = &V2[4*start2B]; 

    bool zeroArrayB = 0;
    NPME_PotGenFunc_Pair_V1 (func,
      nCharge1B, coord1B, charge1B, 
      nCharge2B, coord2B, charge2B, 
      V1_B, V2_B, vecOption, zeroArrayB);
  }
}


void NPME_PotGenFunc_LargeSelf_V1 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge, const double *coord, const double *charge, 
  double *V, int vecOption, size_t blockSize, bool zeroArray)
{
  if (nCharge < blockSize)
    return NPME_PotGenFunc_Self_V1 (func,
        nCharge, coord, charge, V, vecOption, zeroArray);


  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotGenFunc_LargeSelf_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotGenFunc_LargeSelf_V1\n";
    sprintf(str, "blockSize = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n",
      blockSize, NPME_Pot_MaxChgBlock_V1);
    std::cout << str;
    exit(0);
  }

  if (zeroArray)
    memset(V, 0, 4*nCharge*sizeof(double));


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

      const double *charge1B  = &charge[start1B];
      const double *coord1B   = &coord[3*start1B];
      double *V1B             = &V[4*start1B];

      const double *charge2B  = &charge[start2B];
      const double *coord2B   = &coord[3*start2B];
      double *V2B             = &V[4*start2B];

      bool zeroArray2 = 0;
      NPME_PotGenFunc_Pair_V1 (func,
        nCharge1B, coord1B, charge1B, 
        nCharge2B, coord2B, charge2B, 
        V1B, V2B, vecOption, zeroArray2);
    }
    else
    {
      size_t start1B    = i1*blockSize;
      size_t nCharge1B  = blockSize;
      if ( (remain1 > 0) && (i1 == nBlock1 - 1))  nCharge1B = remain1;

      const double *charge1B  = &charge[start1B];
      const double *coord1B   = &coord[3*start1B];
      double *V1B             = &V[4*start1B];

      bool zeroArrayB = 0;
      NPME_PotGenFunc_Self_V1 (func,
        nCharge1B, coord1B, charge1B, V1B, vecOption, zeroArrayB);
    }
  }
}

//******************************************************************************
//******************************************************************************
//******************************************************************************
//*********************Complex Low Level Functions******************************
//******************************************************************************
//******************************************************************************
//******************************************************************************

void NPME_PotGenFunc_Pair_V1 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//output: V1[nCharge1][4]
//        V2[nCharge2][4]
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_Pair_V1.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  if (zeroArray)
  {
    memset(V1, 0, 4*nCharge1*sizeof(_Complex double));
    memset(V2, 0, 4*nCharge2*sizeof(_Complex double));
  }

  double f0_r[NPME_Pot_MaxChgBlock_V1];
  double fX_r[NPME_Pot_MaxChgBlock_V1];
  double fY_r[NPME_Pot_MaxChgBlock_V1];
  double fZ_r[NPME_Pot_MaxChgBlock_V1];

  double f0_i[NPME_Pot_MaxChgBlock_V1];
  double fX_i[NPME_Pot_MaxChgBlock_V1];
  double fY_i[NPME_Pot_MaxChgBlock_V1];
  double fZ_i[NPME_Pot_MaxChgBlock_V1];

  for (size_t i = 0; i < nCharge1; i++)
  {
    const double x1 = coord1[3*i  ];
    const double y1 = coord1[3*i+1];
    const double z1 = coord1[3*i+2];

    for (size_t j = 0; j < nCharge2; j++)
    {
      fX_r[j] = x1 - coord2[3*j  ];
      fY_r[j] = y1 - coord2[3*j+1];
      fZ_r[j] = z1 - coord2[3*j+2];
    }

    func.Calc (nCharge2, f0_r,f0_i, 
      fX_r, fX_i,
      fY_r, fY_i,
      fZ_r, fZ_i);

    for (size_t j = 0; j < nCharge2; j++)
    {
      V1[4*i  ] += (f0_r[j] + I*f0_i[j])*q2[j];
      V1[4*i+1] += (fX_r[j] + I*fX_i[j])*q2[j];
      V1[4*i+2] += (fY_r[j] + I*fY_i[j])*q2[j];
      V1[4*i+3] += (fZ_r[j] + I*fZ_i[j])*q2[j];

      V2[4*j  ] += (f0_r[j] + I*f0_i[j])*q1[i];
      V2[4*j+1] -= (fX_r[j] + I*fX_i[j])*q1[i];
      V2[4*j+2] -= (fY_r[j] + I*fY_i[j])*q1[i];
      V2[4*j+3] -= (fZ_r[j] + I*fZ_i[j])*q1[i];
    }
  }
}


void NPME_PotGenFunc_Self_V1 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[nCharge][4]
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_Self_V1.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  if (zeroArray)
    memset(V, 0, 4*nCharge*sizeof(_Complex double));

  double f0_r[NPME_Pot_MaxChgBlock_V1];
  double fX_r[NPME_Pot_MaxChgBlock_V1];
  double fY_r[NPME_Pot_MaxChgBlock_V1];
  double fZ_r[NPME_Pot_MaxChgBlock_V1];

  double f0_i[NPME_Pot_MaxChgBlock_V1];
  double fX_i[NPME_Pot_MaxChgBlock_V1];
  double fY_i[NPME_Pot_MaxChgBlock_V1];
  double fZ_i[NPME_Pot_MaxChgBlock_V1];

  for (size_t i = 0; i < nCharge; i++)
  {
    const double x1 = coord[3*i  ];
    const double y1 = coord[3*i+1];
    const double z1 = coord[3*i+2];

    for (size_t j = 0; j < i; j++)
    {
      fX_r[j] = x1 - coord[3*j  ];
      fY_r[j] = y1 - coord[3*j+1];
      fZ_r[j] = z1 - coord[3*j+2];
    }

    func.Calc (i, f0_r,f0_i, 
      fX_r, fX_i,
      fY_r, fY_i,
      fZ_r, fZ_i);

    for (size_t j = 0; j < i; j++)
    {
      V[4*i  ] += (f0_r[j] + I*f0_i[j])*q[j];
      V[4*i+1] += (fX_r[j] + I*fX_i[j])*q[j];
      V[4*i+2] += (fY_r[j] + I*fY_i[j])*q[j];
      V[4*i+3] += (fZ_r[j] + I*fZ_i[j])*q[j];

      V[4*j  ] += (f0_r[j] + I*f0_i[j])*q[i];
      V[4*j+1] -= (fX_r[j] + I*fX_i[j])*q[i];
      V[4*j+2] -= (fY_r[j] + I*fY_i[j])*q[i];
      V[4*j+3] -= (fZ_r[j] + I*fZ_i[j])*q[i];
    }
  }
}







#if NPME_USE_AVX
void NPME_PotGenFunc_Pair_V1_AVX (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//output: V1[nCharge1][4]
//        V2[nCharge2][4]
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_Pair_V1_AVX.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain2    = nCharge2%4;
  const size_t nLoop2     = (nCharge2 - remain2)/4;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  const size_t nCharge2Loop = 4*nLoop2wRemainder;


  double coord2_4x[3*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformCoord_4x_AVX (nCharge2, coord2, coord2_4x);

  double q2_4x[2*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformQcomplex_4x_AVX (nCharge2, q2, q2_4x);

  //tmp aligned arrays for potential 2
  double V2_4x[8*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_4x, 0, 32*nLoop2wRemainder*sizeof(double));


  double f0_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fX_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fY_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fZ_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));

  double f0_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fX_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fY_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fZ_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));


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

    size_t index4j, index8j, index12j, index32j;

    index4j  = 0;
    index12j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      __m256d xVec  = _mm256_load_pd (&coord2_4x[index12j  ]);
      __m256d yVec  = _mm256_load_pd (&coord2_4x[index12j+4]);
      __m256d zVec  = _mm256_load_pd (&coord2_4x[index12j+8]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      _mm256_store_pd (&fX_r[index4j], xVec);
      _mm256_store_pd (&fY_r[index4j], yVec);
      _mm256_store_pd (&fZ_r[index4j], zVec);

      index4j  += 4;
      index12j += 12;
    }

    func.CalcAVX (nCharge2Loop, f0_r, f0_i,
      fX_r, fX_i,
      fY_r, fY_i,
      fZ_r, fZ_i);

    index4j  = 0;
    index8j  = 0;
    index32j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      const __m256d q2r_Vec = _mm256_load_pd (&q2_4x[index8j]);
      const __m256d q2i_Vec = _mm256_load_pd (&q2_4x[index8j+4]);

      const __m256d f0r_Vec = _mm256_load_pd  (&f0_r[index4j]);
      const __m256d fXr_Vec = _mm256_load_pd  (&fX_r[index4j]);
      const __m256d fYr_Vec = _mm256_load_pd  (&fY_r[index4j]);
      const __m256d fZr_Vec = _mm256_load_pd  (&fZ_r[index4j]);

      const __m256d f0i_Vec = _mm256_load_pd  (&f0_i[index4j]);
      const __m256d fXi_Vec = _mm256_load_pd  (&fX_i[index4j]);
      const __m256d fYi_Vec = _mm256_load_pd  (&fY_i[index4j]);
      const __m256d fZi_Vec = _mm256_load_pd  (&fZ_i[index4j]);

      __m256d V0r_2_Vec  = _mm256_load_pd (&V2_4x[index32j   ]);
      __m256d V0i_2_Vec  = _mm256_load_pd (&V2_4x[index32j+ 4]);
      __m256d VXr_2_Vec  = _mm256_load_pd (&V2_4x[index32j+ 8]);
      __m256d VXi_2_Vec  = _mm256_load_pd (&V2_4x[index32j+12]);
      __m256d VYr_2_Vec  = _mm256_load_pd (&V2_4x[index32j+16]);
      __m256d VYi_2_Vec  = _mm256_load_pd (&V2_4x[index32j+20]);
      __m256d VZr_2_Vec  = _mm256_load_pd (&V2_4x[index32j+24]);
      __m256d VZi_2_Vec  = _mm256_load_pd (&V2_4x[index32j+28]);


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

      index4j  += 4;
      index8j  += 8;
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
    NPME_TransformComplexV1_4x_2_V1_AVX (nCharge2, V2_4x, V2);
  else
    NPME_TransformUpdateComplexV1_4x_2_V1_AVX (nCharge2, V2_4x, V2);
}

void NPME_PotGenFunc_Self_V1_AVX (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[nCharge][4]
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_Self_V1_AVX.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  double coord_4x[3*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformCoord_4x_AVX (nCharge, coord, coord_4x);

  double q_4x[2*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformQcomplex_4x_AVX (nCharge, q, q_4x);

  //tmp aligned arrays for potential 2
  double V2_4x[8*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  {
    const size_t remain    = nCharge%4;
    const size_t nLoop     = (nCharge - remain)/4;
    size_t nLoopwRemainder = nLoop;
    if (remain > 0)
      nLoopwRemainder++;
    memset(V2_4x, 0, 32*nLoopwRemainder*sizeof(double));
  }

  double f0_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fX_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fY_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fZ_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));

  double f0_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fX_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fY_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fZ_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));

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


    const size_t remainInner   = (i)%4;
    const size_t nLoopInner    = (i-remainInner)/4;
    size_t nLoopInnerwRemainder = nLoopInner;
    if (remainInner > 0)
      nLoopInnerwRemainder++;

    size_t index4j, index8j, index12j, index32j;

    index4j  = 0;
    index12j = 0;
    for (size_t j = 0; j < nLoopInner; j++)
    {
      __m256d xVec  = _mm256_load_pd (&coord_4x[index12j  ]);
      __m256d yVec  = _mm256_load_pd (&coord_4x[index12j+4]);
      __m256d zVec  = _mm256_load_pd (&coord_4x[index12j+8]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      _mm256_store_pd (&fX_r[index4j], xVec);
      _mm256_store_pd (&fY_r[index4j], yVec);
      _mm256_store_pd (&fZ_r[index4j], zVec);

      index4j  += 4;
      index12j += 12;
    }

    if (remainInner > 0)
    {
      const size_t indexStart = 4*nLoopInner;
      const double *crdLoc    = &coord[3*indexStart];
      const double X          = r1[0]+1.0;
      double x2Array[4]  __attribute__((aligned(64))) = {X, X, X, X};
      double y2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double z2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};


      for (size_t k = 0; k < remainInner; k++)
      {
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
      }

      __m256d xVec      = _mm256_load_pd (x2Array);
      __m256d yVec      = _mm256_load_pd (y2Array);
      __m256d zVec      = _mm256_load_pd (z2Array);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      _mm256_store_pd (&fX_r[index4j], xVec);
      _mm256_store_pd (&fY_r[index4j], yVec);
      _mm256_store_pd (&fZ_r[index4j], zVec);
    }

    func.CalcAVX (4*nLoopInnerwRemainder, f0_r, f0_i,
      fX_r, fX_i,
      fY_r, fY_i,
      fZ_r, fZ_i);


    index4j  = 0;
    index8j  = 0;
    index32j = 0;
    for (size_t j = 0; j < nLoopInner; j++)
    {
      const __m256d q2r_Vec = _mm256_load_pd (&q_4x[index8j  ]);
      const __m256d q2i_Vec = _mm256_load_pd (&q_4x[index8j+4]);

      const __m256d f0r_Vec = _mm256_load_pd  (&f0_r[index4j]);
      const __m256d fXr_Vec = _mm256_load_pd  (&fX_r[index4j]);
      const __m256d fYr_Vec = _mm256_load_pd  (&fY_r[index4j]);
      const __m256d fZr_Vec = _mm256_load_pd  (&fZ_r[index4j]);

      const __m256d f0i_Vec = _mm256_load_pd  (&f0_i[index4j]);
      const __m256d fXi_Vec = _mm256_load_pd  (&fX_i[index4j]);
      const __m256d fYi_Vec = _mm256_load_pd  (&fY_i[index4j]);
      const __m256d fZi_Vec = _mm256_load_pd  (&fZ_i[index4j]);

      __m256d V0r_2_Vec     = _mm256_load_pd (&V2_4x[index32j   ]);
      __m256d V0i_2_Vec     = _mm256_load_pd (&V2_4x[index32j+ 4]);
      __m256d VXr_2_Vec     = _mm256_load_pd (&V2_4x[index32j+ 8]);
      __m256d VXi_2_Vec     = _mm256_load_pd (&V2_4x[index32j+12]);
      __m256d VYr_2_Vec     = _mm256_load_pd (&V2_4x[index32j+16]);
      __m256d VYi_2_Vec     = _mm256_load_pd (&V2_4x[index32j+20]);
      __m256d VZr_2_Vec     = _mm256_load_pd (&V2_4x[index32j+24]);
      __m256d VZi_2_Vec     = _mm256_load_pd (&V2_4x[index32j+28]);

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


      index4j  += 4;
      index8j  += 8;
      index32j += 32;
    }
    if (remainInner > 0)
    {
      const size_t indexStart     = 4*nLoopInner;
      const _Complex double *qLoc = &q[indexStart];
      double q2rArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};
      double q2iArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};
      double mskArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};

      for (size_t k = 0; k < remainInner; k++)
      {
        q2rArray[k] = creal(qLoc[k]);
        q2iArray[k] = cimag(qLoc[k]);
        mskArray[k] = 1.0;
      }
      const __m256d q2r_Vec = _mm256_load_pd (q2rArray);
      const __m256d q2i_Vec = _mm256_load_pd (q2iArray);
      const __m256d mskVec  = _mm256_load_pd (mskArray);

      __m256d f0r_Vec       = _mm256_load_pd  (&f0_r[index4j]);
      __m256d fXr_Vec       = _mm256_load_pd  (&fX_r[index4j]);
      __m256d fYr_Vec       = _mm256_load_pd  (&fY_r[index4j]);
      __m256d fZr_Vec       = _mm256_load_pd  (&fZ_r[index4j]);

      __m256d f0i_Vec       = _mm256_load_pd  (&f0_i[index4j]);
      __m256d fXi_Vec       = _mm256_load_pd  (&fX_i[index4j]);
      __m256d fYi_Vec       = _mm256_load_pd  (&fY_i[index4j]);
      __m256d fZi_Vec       = _mm256_load_pd  (&fZ_i[index4j]);

      //apply mask to q1
      q1r_Vec               = _mm256_mul_pd  (mskVec, q1r_Vec);
      q1i_Vec               = _mm256_mul_pd  (mskVec, q1i_Vec);
      q1Nr_Vec              = _mm256_mul_pd  (mskVec, q1Nr_Vec);

      __m256d V0r_2_Vec     = _mm256_load_pd (&V2_4x[index32j   ]);
      __m256d V0i_2_Vec     = _mm256_load_pd (&V2_4x[index32j+ 4]);
      __m256d VXr_2_Vec     = _mm256_load_pd (&V2_4x[index32j+ 8]);
      __m256d VXi_2_Vec     = _mm256_load_pd (&V2_4x[index32j+12]);
      __m256d VYr_2_Vec     = _mm256_load_pd (&V2_4x[index32j+16]);
      __m256d VYi_2_Vec     = _mm256_load_pd (&V2_4x[index32j+20]);
      __m256d VZr_2_Vec     = _mm256_load_pd (&V2_4x[index32j+24]);
      __m256d VZi_2_Vec     = _mm256_load_pd (&V2_4x[index32j+28]);

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

void NPME_PotGenFunc_Pair_V1_AVX_512 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//output: V1[nCharge1][4]
//        V2[nCharge2][4]
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_Pair_V1_AVX_512.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain2    = nCharge2%8;
  const size_t nLoop2     = (nCharge2 - remain2)/8;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  const size_t nCharge2Loop = 8*nLoop2wRemainder;


  double coord2_8x[3*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformCoord_8x_AVX_512 (nCharge2, coord2, coord2_8x);

  double q2_8x[2*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformQcomplex_8x_AVX_512 (nCharge2, q2, q2_8x);

  //tmp aligned arrays for potential 2
  double V2_8x[8*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_8x, 0, 64*nLoop2wRemainder*sizeof(double));


  double f0_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fX_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fY_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fZ_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));

  double f0_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fX_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fY_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fZ_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));


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

    size_t index8j, index16j, index24j, index64j;

    index8j  = 0;
    index24j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      __m512d xVec  = _mm512_load_pd (&coord2_8x[index24j   ]);
      __m512d yVec  = _mm512_load_pd (&coord2_8x[index24j+ 8]);
      __m512d zVec  = _mm512_load_pd (&coord2_8x[index24j+16]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      _mm512_store_pd (&fX_r[index8j], xVec);
      _mm512_store_pd (&fY_r[index8j], yVec);
      _mm512_store_pd (&fZ_r[index8j], zVec);

      index8j  += 8;
      index24j += 24;
    }

    func.CalcAVX_512 (nCharge2Loop, f0_r, f0_i,
      fX_r, fX_i,
      fY_r, fY_i,
      fZ_r, fZ_i);

    index8j  = 0;
    index16j = 0;
    index64j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      const __m512d q2r_Vec = _mm512_load_pd (&q2_8x[index16j]);
      const __m512d q2i_Vec = _mm512_load_pd (&q2_8x[index16j+8]);

      const __m512d f0r_Vec = _mm512_load_pd  (&f0_r[index8j]);
      const __m512d fXr_Vec = _mm512_load_pd  (&fX_r[index8j]);
      const __m512d fYr_Vec = _mm512_load_pd  (&fY_r[index8j]);
      const __m512d fZr_Vec = _mm512_load_pd  (&fZ_r[index8j]);

      const __m512d f0i_Vec = _mm512_load_pd  (&f0_i[index8j]);
      const __m512d fXi_Vec = _mm512_load_pd  (&fX_i[index8j]);
      const __m512d fYi_Vec = _mm512_load_pd  (&fY_i[index8j]);
      const __m512d fZi_Vec = _mm512_load_pd  (&fZ_i[index8j]);

      __m512d V0r_2_Vec  = _mm512_load_pd (&V2_8x[index64j   ]);
      __m512d V0i_2_Vec  = _mm512_load_pd (&V2_8x[index64j+ 8]);
      __m512d VXr_2_Vec  = _mm512_load_pd (&V2_8x[index64j+16]);
      __m512d VXi_2_Vec  = _mm512_load_pd (&V2_8x[index64j+24]);
      __m512d VYr_2_Vec  = _mm512_load_pd (&V2_8x[index64j+32]);
      __m512d VYi_2_Vec  = _mm512_load_pd (&V2_8x[index64j+40]);
      __m512d VZr_2_Vec  = _mm512_load_pd (&V2_8x[index64j+48]);
      __m512d VZi_2_Vec  = _mm512_load_pd (&V2_8x[index64j+56]);


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

      index8j  += 8;
      index16j += 16;
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
    NPME_TransformComplexV1_8x_2_V1_AVX_512 (nCharge2, V2_8x, V2);
  else
    NPME_TransformUpdateComplexV1_8x_2_V1_AVX_512 (nCharge2, V2_8x, V2);
}

void NPME_PotGenFunc_Self_V1_AVX_512 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[nCharge][4]
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotGenFunc_Self_V1_AVX_512.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }

  double coord_8x[3*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformCoord_8x_AVX_512 (nCharge, coord, coord_8x);

  double q_8x[2*NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  NPME_TransformQcomplex_8x_AVX_512 (nCharge, q, q_8x);

  //tmp aligned arrays for potential 2
  double V2_8x[8*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  {
    const size_t remain    = nCharge%8;
    const size_t nLoop     = (nCharge - remain)/8;
    size_t nLoopwRemainder = nLoop;
    if (remain > 0)
      nLoopwRemainder++;
    memset(V2_8x, 0, 64*nLoopwRemainder*sizeof(double));
  }

  double f0_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fX_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fY_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fZ_r[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));

  double f0_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fX_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fY_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));
  double fZ_i[NPME_Pot_MaxChgBlock_V1]  __attribute__((aligned(64)));

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


    const size_t remainInner   = (i)%8;
    const size_t nLoopInner    = (i-remainInner)/8;
    size_t nLoopInnerwRemainder = nLoopInner;
    if (remainInner > 0)
      nLoopInnerwRemainder++;

    size_t index8j, index16j, index24j, index64j;

    index8j  = 0;
    index24j = 0;
    for (size_t j = 0; j < nLoopInner; j++)
    {
      __m512d xVec  = _mm512_load_pd (&coord_8x[index24j   ]);
      __m512d yVec  = _mm512_load_pd (&coord_8x[index24j+ 8]);
      __m512d zVec  = _mm512_load_pd (&coord_8x[index24j+16]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      _mm512_store_pd (&fX_r[index8j], xVec);
      _mm512_store_pd (&fY_r[index8j], yVec);
      _mm512_store_pd (&fZ_r[index8j], zVec);

      index8j  += 8;
      index24j += 24;
    }

    if (remainInner > 0)
    {
      const size_t indexStart = 8*nLoopInner;
      const double *crdLoc    = &coord[3*indexStart];
      const double X          = r1[0]+1.0;
      double x2Array[8]  __attribute__((aligned(64))) = {X,X,X,X,X,X,X,X};
      double y2Array[8]  __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};
      double z2Array[8]  __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};

      for (size_t k = 0; k < remainInner; k++)
      {
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
      }

      __m512d xVec      = _mm512_load_pd (x2Array);
      __m512d yVec      = _mm512_load_pd (y2Array);
      __m512d zVec      = _mm512_load_pd (z2Array);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      _mm512_store_pd (&fX_r[index8j], xVec);
      _mm512_store_pd (&fY_r[index8j], yVec);
      _mm512_store_pd (&fZ_r[index8j], zVec);
    }

    func.CalcAVX_512 (8*nLoopInnerwRemainder, f0_r, f0_i,
      fX_r, fX_i,
      fY_r, fY_i,
      fZ_r, fZ_i);


    index8j  = 0;
    index16j = 0;
    index64j = 0;
    for (size_t j = 0; j < nLoopInner; j++)
    {
      const __m512d q2r_Vec = _mm512_load_pd (&q_8x[index16j  ]);
      const __m512d q2i_Vec = _mm512_load_pd (&q_8x[index16j+8]);

      const __m512d f0r_Vec = _mm512_load_pd  (&f0_r[index8j]);
      const __m512d fXr_Vec = _mm512_load_pd  (&fX_r[index8j]);
      const __m512d fYr_Vec = _mm512_load_pd  (&fY_r[index8j]);
      const __m512d fZr_Vec = _mm512_load_pd  (&fZ_r[index8j]);

      const __m512d f0i_Vec = _mm512_load_pd  (&f0_i[index8j]);
      const __m512d fXi_Vec = _mm512_load_pd  (&fX_i[index8j]);
      const __m512d fYi_Vec = _mm512_load_pd  (&fY_i[index8j]);
      const __m512d fZi_Vec = _mm512_load_pd  (&fZ_i[index8j]);

      __m512d V0r_2_Vec     = _mm512_load_pd (&V2_8x[index64j   ]);
      __m512d V0i_2_Vec     = _mm512_load_pd (&V2_8x[index64j+ 8]);
      __m512d VXr_2_Vec     = _mm512_load_pd (&V2_8x[index64j+16]);
      __m512d VXi_2_Vec     = _mm512_load_pd (&V2_8x[index64j+24]);
      __m512d VYr_2_Vec     = _mm512_load_pd (&V2_8x[index64j+32]);
      __m512d VYi_2_Vec     = _mm512_load_pd (&V2_8x[index64j+40]);
      __m512d VZr_2_Vec     = _mm512_load_pd (&V2_8x[index64j+48]);
      __m512d VZi_2_Vec     = _mm512_load_pd (&V2_8x[index64j+56]);

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


      index8j  += 8;
      index16j += 16;
      index64j += 64;
    }
    if (remainInner > 0)
    {
      const size_t indexStart     = 8*nLoopInner;
      const _Complex double *qLoc = &q[indexStart];
      double q2rArray[8] __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};
      double q2iArray[8] __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};
      double mskArray[8] __attribute__((aligned(64))) = {0,0,0,0,0,0,0,0};

      for (size_t k = 0; k < remainInner; k++)
      {
        q2rArray[k] = creal(qLoc[k]);
        q2iArray[k] = cimag(qLoc[k]);
        mskArray[k] = 1.0;
      }
      const __m512d q2r_Vec = _mm512_load_pd (q2rArray);
      const __m512d q2i_Vec = _mm512_load_pd (q2iArray);
      const __m512d mskVec  = _mm512_load_pd (mskArray);

      __m512d f0r_Vec       = _mm512_load_pd  (&f0_r[index8j]);
      __m512d fXr_Vec       = _mm512_load_pd  (&fX_r[index8j]);
      __m512d fYr_Vec       = _mm512_load_pd  (&fY_r[index8j]);
      __m512d fZr_Vec       = _mm512_load_pd  (&fZ_r[index8j]);

      __m512d f0i_Vec       = _mm512_load_pd  (&f0_i[index8j]);
      __m512d fXi_Vec       = _mm512_load_pd  (&fX_i[index8j]);
      __m512d fYi_Vec       = _mm512_load_pd  (&fY_i[index8j]);
      __m512d fZi_Vec       = _mm512_load_pd  (&fZ_i[index8j]);

      //apply mask to q1
      q1r_Vec               = _mm512_mul_pd  (mskVec, q1r_Vec);
      q1i_Vec               = _mm512_mul_pd  (mskVec, q1i_Vec);
      q1Nr_Vec              = _mm512_mul_pd  (mskVec, q1Nr_Vec);

      __m512d V0r_2_Vec     = _mm512_load_pd (&V2_8x[index64j   ]);
      __m512d V0i_2_Vec     = _mm512_load_pd (&V2_8x[index64j+ 8]);
      __m512d VXr_2_Vec     = _mm512_load_pd (&V2_8x[index64j+16]);
      __m512d VXi_2_Vec     = _mm512_load_pd (&V2_8x[index64j+24]);
      __m512d VYr_2_Vec     = _mm512_load_pd (&V2_8x[index64j+32]);
      __m512d VYi_2_Vec     = _mm512_load_pd (&V2_8x[index64j+40]);
      __m512d VZr_2_Vec     = _mm512_load_pd (&V2_8x[index64j+48]);
      __m512d VZi_2_Vec     = _mm512_load_pd (&V2_8x[index64j+56]);


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



void NPME_PotGenFunc_Pair_V1 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, int vecOption, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[nCharge1][4]
//        V2[nCharge2][4]
{
  if (vecOption == 0)
  {
    NPME_PotGenFunc_Pair_V1 (func,
      nCharge1, coord1, q1,
      nCharge2, coord2, q2, V1, V2, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotGenFunc_Pair_V1_AVX (func,
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotGenFunc_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotGenFunc_Pair_V1_AVX_512 (func,
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotGenFunc_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}

void NPME_PotGenFunc_Self_V1 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge, const double *coord, const _Complex double *q, 
  _Complex double *V, int vecOption, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V[nCharge][4]
{
  if (vecOption == 0)
  {
    NPME_PotGenFunc_Self_V1 (func,
      nCharge, coord, q, V, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotGenFunc_Self_V1_AVX (func,
        nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotGenFunc_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotGenFunc_Self_V1_AVX_512 (func,
        nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotGenFunc_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}



void NPME_PotGenFunc_LargePair_V1 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge1, const double *coord1, const _Complex double *charge1,
  const size_t nCharge2, const double *coord2, const _Complex double *charge2,
  _Complex double *V1, _Complex double *V2, int vecOption, 
  size_t blockSize, bool zeroArray)
{
  if (nCharge2 < blockSize)
    return NPME_PotGenFunc_Pair_V1 (func,
              nCharge1, coord1, charge1,
              nCharge2, coord2, charge2, 
              V1, V2, vecOption, zeroArray);

  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotGenFunc_LargePair_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotGenFunc_LargePair_V1\n";
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
    NPME_PotGenFunc_Pair_V1 (func,
      nCharge1B, coord1B, charge1B, 
      nCharge2B, coord2B, charge2B, 
      V1_B, V2_B, vecOption, zeroArrayB);
  }
}


void NPME_PotGenFunc_LargeSelf_V1 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge, const double *coord, const _Complex double *charge, 
  _Complex double *V, int vecOption, size_t blockSize, bool zeroArray)
{
  if (nCharge < blockSize)
    return NPME_PotGenFunc_Self_V1 (func,
        nCharge, coord, charge, V, vecOption, zeroArray);


  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotGenFunc_LargeSelf_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotGenFunc_LargeSelf_V1\n";
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

      bool zeroArray2 = 0;
      NPME_PotGenFunc_Pair_V1 (func,
        nCharge1B, coord1B, charge1B, 
        nCharge2B, coord2B, charge2B, 
        V1B, V2B, vecOption, zeroArray2);
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
      NPME_PotGenFunc_Self_V1 (func,
        nCharge1B, coord1B, charge1B, V1B, vecOption, zeroArrayB);
    }
  }
}



}//end namespace NPME_Library



