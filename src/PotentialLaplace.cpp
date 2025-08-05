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


#include "Constant.h"
#include "AlignedArray.h"
#include "PotentialLaplace.h"
#include "MathFunctions.h"
#include "SupportFunctions.h"
#include "FunctionDerivMatch.h"
#include "PartitionBox.h"
#include "PartitionEmbeddedBox.h"
#include "PotentialSupportFunctions.h"



namespace NPME_Library
{


void NPME_PotLaplace_MacroSelf_V1 (
  const size_t nCharge, const double *coord, const double *Q1, 
  double *V1, const int nProc, const int vecOption, 
  const size_t blockSize)
{
  if (blockSize%8 != 0)
  {
    printf("Error in NPME_PotLaplace_MacroSelf_V1.\n");
    printf("blockSize = %lu is not a multiple of 8\n", blockSize);
    exit(0);
  }

  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_MacroSelf_V1.\n");
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
  #pragma omp parallel shared(V1, Q1, coord, nPair, nBlock, remain, vecOption, blockSize) private(k) default(none) num_threads(nProc)
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

        NPME_PotLaplace_Pair_V1 (
          nCharge1, coord1, Q1_1, 
          nCharge2, coord2, Q1_2, 
          V1loc_1, V1loc_2, vecOption);

        #pragma omp critical (update_NPME_PotLaplace_MacroSelf_V1)
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

        NPME_PotLaplace_Self_V1 (nCharge1, 
          coord1, Q1_1, V1loc_1, vecOption);

        #pragma omp critical (update_NPME_PotLaplace_MacroSelf_V1)
        {
          for (size_t n = 0; n < 4*nCharge1; n++)   V1_1[n] += V1loc_1[n];
        }
      }
    }
  }
}












void NPME_PotLaplace_SR_DM_ClusterElement_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
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
      NPME_PotLaplace_SR_DM_LargePair_V1 (
        Nder, a, b, Rdir,
        nChargeB1, coordB1, chargeB1, 
        nChargeB2, coordB2, chargeB2, 
        VB1, VB2, vecOption, blockSize, zeroArrayB);
    }
    else
    {
      NPME_PotLaplace_SR_DM_LargeSelf_V1 (
        Nder, a, b, Rdir,
        nChargeB1, coordB1, chargeB1, VB1, vecOption, blockSize, zeroArrayB);
    }
  }
}



void NPME_PotLaplace_SR_DM_DirectSum_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge, const double *coord, const double *charge, 
  double *V, const int nProc, const int vecOption,
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster, 
  const size_t blockSize)
{
  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotLaplace_SR_DM_DirectSum_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotLaplace_SR_DM_DirectSum_V1\n";
    sprintf(str, "blockSize = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n",
      blockSize, NPME_Pot_MaxChgBlock_V1);
    std::cout << str;
    exit(0);
  }

  memset(V, 0, 4*nCharge*sizeof(double));

  const size_t nCluster = cluster.size();

  size_t k;
  #pragma omp parallel shared(V, charge, coord, Rdir, nCluster, cluster, Nder, a, b, blockSize, vecOption) private(k) default(none) num_threads(nProc)
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


      NPME_PotLaplace_SR_DM_ClusterElement_V1 (Nder, a, b, Rdir,
            cluster[k], coord, charge, V1loc_1, V1loc_2, 
            vecOption, blockSize, zeroArray);



      #pragma omp critical (update_NPME_PotLaplace_SR_DM_DirectSum_V1)
      {
        for (size_t n = 0; n < 4*nChargeA1; n++)   VA1[n] += V1loc_1[n];
        for (size_t n = 0; n < 4*nChargeA2; n++)   VA2[n] += V1loc_2[n];
      }
    }
  }
}


void NPME_PotLaplace_SR_Original_ClusterElement_V1 (const double beta,
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
      NPME_PotLaplace_SR_Original_LargePair_V1 (beta,
        nChargeB1, coordB1, chargeB1, 
        nChargeB2, coordB2, chargeB2, 
        VB1, VB2, vecOption, blockSize, zeroArrayB);
    }
    else
    {
      NPME_PotLaplace_SR_Original_LargeSelf_V1 (beta,
        nChargeB1, coordB1, chargeB1, VB1, vecOption, blockSize, zeroArrayB);
    }
  }
}



void NPME_PotLaplace_SR_Original_DirectSum_V1 (const double beta,
  const size_t nCharge, const double *coord, const double *charge, 
  double *V, const int nProc, const int vecOption,
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster, 
  const size_t blockSize)
{
  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotLaplace_SR_Original_DirectSum_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotLaplace_SR_Original_DirectSum_V1\n";
    sprintf(str, "blockSize = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n",
      blockSize, NPME_Pot_MaxChgBlock_V1);
    std::cout << str;
    exit(0);
  }

  memset(V, 0, 4*nCharge*sizeof(double));

  const size_t nCluster = cluster.size();

  size_t k;
  #pragma omp parallel shared(V, charge, coord, cluster, beta, nCharge, nCluster, vecOption, blockSize) private(k) default(none) num_threads(nProc)
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


      NPME_PotLaplace_SR_Original_ClusterElement_V1 (beta,
            cluster[k], coord, charge, V1loc_1, V1loc_2, 
            vecOption, blockSize, zeroArray);

      #pragma omp critical (update_NPME_PotLaplace_SR_DM_DirectSum_V1)
      {
        for (size_t n = 0; n < 4*nChargeA1; n++)   VA1[n] += V1loc_1[n];
        for (size_t n = 0; n < 4*nChargeA2; n++)   VA2[n] += V1loc_2[n];
      }
    }
  }
}

void NPME_PotLaplace_Pair_V1 (
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//output: V1[nCharge1][4]
//        V2[nCharge2][4]
//        V[nCharge][4] = (V0[0], VX[0], VY[0], VZ[0],
//                         V0[1], VX[1], VY[1], VZ[1],..)
{
  if (zeroArray)
  {
    memset(V1, 0, 4*nCharge1*sizeof(double));
    memset(V2, 0, 4*nCharge2*sizeof(double));
  }

  for (size_t i = 0; i < nCharge1; i++)
  {
    const double x1 = coord1[3*i  ];
    const double y1 = coord1[3*i+1];
    const double z1 = coord1[3*i+2];

    for (size_t j = 0; j < nCharge2; j++)
    {
      const double x = x1 - coord2[3*j  ];
      const double y = y1 - coord2[3*j+1];
      const double z = z1 - coord2[3*j+2];

      const double r2 = x*x + y*y + z*z;
      const double r  = sqrt(fabs(r2));
      const double f0 = 1.0/r;
      const double f1 = -f0/r2;

      const double fX = x*f1;
      const double fY = y*f1;
      const double fZ = z*f1;


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

void NPME_PotLaplace_Self_V1 (const size_t nCharge, 
  const double *coord, const double *q, double *V, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//        V[nCharge][4] = (V0[0], VX[0], VY[0], VZ[0],
//                         V0[1], VX[1], VY[1], VZ[1],..)
{
  if (zeroArray)
    memset(V, 0, 4*nCharge*sizeof(double));

  for (size_t i = 0; i < nCharge; i++)
  {
    const double x1 = coord[3*i  ];
    const double y1 = coord[3*i+1];
    const double z1 = coord[3*i+2];

    for (size_t j = 0; j < i; j++)
    {
      const double x = x1 - coord[3*j  ];
      const double y = y1 - coord[3*j+1];
      const double z = z1 - coord[3*j+2];

      const double r2 = x*x + y*y + z*z;
      const double r  = sqrt(fabs(r2));
      const double f0 = 1.0/r;
      const double f1 = -f0/r2;

      const double fX = x*f1;
      const double fY = y*f1;
      const double fZ = z*f1;


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











void NPME_PotLaplace_SR_DM_Pair_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        a[Nder+1], b[Nder+1]
//output: V1[nCharge1][4]
//        V2[nCharge2][4]
//        V[nCharge][4] = (V0[0], VX[0], VY[0], VZ[0],
//                         V0[1], VX[1], VY[1], VZ[1],..)
{
  if (zeroArray)
  {
    memset(V1, 0, 4*nCharge1*sizeof(double));
    memset(V2, 0, 4*nCharge2*sizeof(double));
  }

  for (size_t i = 0; i < nCharge1; i++)
  {
    const double x1 = coord1[3*i  ];
    const double y1 = coord1[3*i+1];
    const double z1 = coord1[3*i+2];

    for (size_t j = 0; j < nCharge2; j++)
    {
      const double x = x1 - coord2[3*j  ];
      const double y = y1 - coord2[3*j+1];
      const double z = z1 - coord2[3*j+2];

      const double r2 = x*x + y*y + z*z;
      const double r  = sqrt(fabs(r2));
      const double r3 = r*r2;

      double f0, f1;
      if (r > Rdir)
      {
        f0  = 0;
        f1  = 0;
      }
      else
      {
        f0  = 1.0/r - 
                NPME_FunctionDerivMatch_EvenSeriesReal (f1, 
                  Nder, a, b, r2);
        f1  = -1.0/r3 - f1;
      }

      const double fX = x*f1;
      const double fY = y*f1;
      const double fZ = z*f1;

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

void NPME_PotLaplace_SR_DM_Self_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge, const double *coord, const double *q, double *V,
  bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//        a[Nder+1], b[Nder+1]
//output: V[nCharge][4] = (V0[0], VX[0], VY[0], VZ[0],
//                         V0[1], VX[1], VY[1], VZ[1],..)
{
  if (zeroArray)
    memset(V, 0, 4*nCharge*sizeof(double));

  for (size_t i = 0; i < nCharge; i++)
  {
    const double x1 = coord[3*i  ];
    const double y1 = coord[3*i+1];
    const double z1 = coord[3*i+2];

    for (size_t j = 0; j < i; j++)
    {
      const double x = x1 - coord[3*j  ];
      const double y = y1 - coord[3*j+1];
      const double z = z1 - coord[3*j+2];

      const double r2 = x*x + y*y + z*z;
      const double r  = sqrt(fabs(r2));
      const double r3 = r*r2;

      double f0, f1;
      if (r > Rdir)
      {
        f0  = 0;
        f1  = 0;
      }
      else
      {
        f0  = 1.0/r - 
                NPME_FunctionDerivMatch_EvenSeriesReal (f1, 
                  Nder, a, b, r2);
        f1  = -1.0/r3 - f1;
      }

      const double fX = x*f1;
      const double fY = y*f1;
      const double fZ = z*f1;


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

void NPME_PotLaplace_SR_Original_Pair_V1 (const double beta, 
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//output: V1[nCharge1][4]
//        V2[nCharge2][4]
//        V[nCharge][4] = (V0[0], VX[0], VY[0], VZ[0],
//                         V0[1], VX[1], VY[1], VZ[1],..)
{
  if (zeroArray)
  {
    memset(V1, 0, 4*nCharge1*sizeof(double));
    memset(V2, 0, 4*nCharge2*sizeof(double));
  }

  const double beta3 = beta*beta*beta;

  for (size_t i = 0; i < nCharge1; i++)
  {
    const double x1 = coord1[3*i  ];
    const double y1 = coord1[3*i+1];
    const double z1 = coord1[3*i+2];

    for (size_t j = 0; j < nCharge2; j++)
    {
      const double x  = x1 - coord2[3*j  ];
      const double y  = y1 - coord2[3*j+1];
      const double z  = z1 - coord2[3*j+2];

      const double r2 = x*x + y*y + z*z;
      const double r  = sqrt(fabs(r2));
      double B0, B1;
      B0 = NPME_Berfc_1 (B1, beta*r);

      const double f0 = beta*B0;
      const double f1 = beta3*B1;

      const double fX = x*f1;
      const double fY = y*f1;
      const double fZ = z*f1;


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

void NPME_PotLaplace_SR_Original_Self_V1 (const double beta, 
  const size_t nCharge, const double *coord, const double *q, double *V,
  bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//        V[nCharge][4] = (V0[0], VX[0], VY[0], VZ[0],
//                         V0[1], VX[1], VY[1], VZ[1],..)
{
  if (zeroArray)
    memset(V, 0, 4*nCharge*sizeof(double));

  const double beta3 = beta*beta*beta;

  for (size_t i = 0; i < nCharge; i++)
  {
    const double x1 = coord[3*i  ];
    const double y1 = coord[3*i+1];
    const double z1 = coord[3*i+2];

    for (size_t j = 0; j < i; j++)
    {
      const double x = x1 - coord[3*j  ];
      const double y = y1 - coord[3*j+1];
      const double z = z1 - coord[3*j+2];

      const double r2 = x*x + y*y + z*z;
      const double r  = sqrt(fabs(r2));
      double B0, B1;
      B0 = NPME_Berfc_1 (B1, beta*r);

      const double f0 = beta*B0;
      const double f1 = beta3*B1;

      const double fX = x*f1;
      const double fY = y*f1;
      const double fZ = z*f1;


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


void NPME_PotLaplace_Pair_V1_AVX (
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        nCharge2 <= NPME_Pot_MaxChgBlock_V1
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_Pair_V1_AVX.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain2    = nCharge2%4;
  const size_t nLoop2     = (nCharge2 - remain2)/4;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  double qReCrd_4x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_4x    [4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_4x, 0, 16*nLoop2wRemainder*sizeof(double));
  NPME_TransformQrealCoord_4x_AVX (nCharge2, q2, coord2, qReCrd_4x);


  const __m256d negOneVec = _mm256_set1_pd(-1.0);
  const __m256d oneVec    = _mm256_set1_pd( 1.0);


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


    size_t index16j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      __m256d q2Vec     = _mm256_load_pd (&qReCrd_4x[index16j   ]);
      __m256d xVec      = _mm256_load_pd (&qReCrd_4x[index16j+ 4]);
      __m256d yVec      = _mm256_load_pd (&qReCrd_4x[index16j+ 8]);
      __m256d zVec      = _mm256_load_pd (&qReCrd_4x[index16j+12]);

      __m256d V0_2_Vec  = _mm256_load_pd (&V2_4x[index16j   ]);
      __m256d VX_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 4]);
      __m256d VY_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 8]);
      __m256d VZ_2_Vec  = _mm256_load_pd (&V2_4x[index16j+12]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      __m256d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
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
        __m256d f1_Vec;
        f0_Vec  = _mm256_div_pd  (oneVec, rVec);
        f1_Vec  = _mm256_div_pd  (f0_Vec, r2Vec);
        f1_Vec  = _mm256_mul_pd  (f1_Vec, negOneVec);

        fX_Vec  = _mm256_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm256_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm256_mul_pd  (f1_Vec, zVec);
      }

  
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

void NPME_PotLaplace_Self_V1_AVX (
  const size_t nCharge, const double *coord, const double *q, double *V,
  bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_Self_V1_AVX.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  if (NPME_Pot_MaxChgBlock_V1%8 != 0)
  {
    printf("Error in NPME_PotLaplace_Self_V1_AVX.\n");
    printf("NPME_Pot_MaxChgBlock_V1 = %lu is not a multiple of 8\n",
      NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain    = nCharge%4;
  const size_t nLoop     = (nCharge - remain)/4;
  size_t nLoopwRemainder = nLoop;
  if (remain > 0)
    nLoopwRemainder++;
  double qReCrd_4x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_4x    [4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));


//The following function should work, but results in a compiler error
//  and should work in newer updated compilers
//memset(V2_4x, 0, 16*nLoopwRemainder*sizeof(double));

  //current work-around:
  {
    __m256d zeroVec = _mm256_setzero_pd();
    for (size_t i = 0; i < 16 * nLoopwRemainder; i += 4)
      _mm256_store_pd(&V2_4x[i], zeroVec);
  }
  NPME_TransformQrealCoord_4x_AVX (nCharge, q, coord, qReCrd_4x);



  const __m256d negOneVec = _mm256_set1_pd(-1.0);
  const __m256d oneVec    = _mm256_set1_pd( 1.0);



  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r1      = &coord[3*i];
    const __m256d q1Vec   = _mm256_set1_pd( q[i]);
    const __m256d q1NVec  = _mm256_set1_pd(-q[i]);
    const __m256d x1Vec   = _mm256_set1_pd(r1[0]);
    const __m256d y1Vec   = _mm256_set1_pd(r1[1]);
    const __m256d z1Vec   = _mm256_set1_pd(r1[2]);

    __m256d V0_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VX_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VY_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VZ_1_Vec      = _mm256_set1_pd(0.0);

    const size_t remain2 = (i)%4;
    const size_t nLoop2  = (i-remain2)/4;

    size_t index16j = 0;
    for (size_t j = 0; j < nLoop2; j++)
    {
      __m256d q2Vec     = _mm256_load_pd (&qReCrd_4x[index16j   ]);
      __m256d xVec      = _mm256_load_pd (&qReCrd_4x[index16j+ 4]);
      __m256d yVec      = _mm256_load_pd (&qReCrd_4x[index16j+ 8]);
      __m256d zVec      = _mm256_load_pd (&qReCrd_4x[index16j+12]);

      __m256d V0_2_Vec  = _mm256_load_pd (&V2_4x[index16j   ]);
      __m256d VX_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 4]);
      __m256d VY_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 8]);
      __m256d VZ_2_Vec  = _mm256_load_pd (&V2_4x[index16j+12]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);


      __m256d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
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
        __m256d f1_Vec;
        f0_Vec  = _mm256_div_pd  (oneVec, rVec);
        f1_Vec  = _mm256_div_pd  (f0_Vec, r2Vec);
        f1_Vec  = _mm256_mul_pd  (f1_Vec, negOneVec);

        fX_Vec  = _mm256_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm256_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm256_mul_pd  (f1_Vec, zVec);
      }
      
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

      index16j += 16;
    }
    if (remain2 > 0)
    {
      const size_t indexStart = 4*nLoop2;
      const double *qLoc      = &q[indexStart];
      const double *crdLoc    = &coord[3*indexStart];
      const double X          = r1[0]+1.0;

      double x2Array[4]  __attribute__((aligned(64))) = {X, X, X, X};
      double y2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double z2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double q2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double mskArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};

      for (size_t k = 0; k < remain2; k++)
      {
        q2Array[k]  = qLoc[k];
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
        mskArray[k] = 1.0;
      }
      __m256d q2Vec     = _mm256_load_pd (q2Array);
      __m256d xVec      = _mm256_load_pd (x2Array);
      __m256d yVec      = _mm256_load_pd (y2Array);
      __m256d zVec      = _mm256_load_pd (z2Array);
      __m256d mskVec    = _mm256_load_pd (mskArray);
      __m256d V0_2_Vec  = _mm256_load_pd (&V2_4x[index16j   ]);
      __m256d VX_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 4]);
      __m256d VY_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 8]);
      __m256d VZ_2_Vec  = _mm256_load_pd (&V2_4x[index16j+12]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      __m256d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
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
        __m256d f1_Vec;
        f0_Vec  = _mm256_div_pd  (oneVec, rVec);
        f1_Vec  = _mm256_div_pd  (f0_Vec, r2Vec);
        f1_Vec  = _mm256_mul_pd  (f1_Vec, negOneVec);

        //apply mask
        f0_Vec  = _mm256_mul_pd  (mskVec, f0_Vec);
        f1_Vec  = _mm256_mul_pd  (mskVec, f1_Vec);

        fX_Vec  = _mm256_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm256_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm256_mul_pd  (f1_Vec, zVec);
      }
      
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

      index16j += 16;
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


void NPME_PotLaplace_SR_Original_Pair_V1_AVX (const double beta,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        nCharge2 <= NPME_Pot_MaxChgBlock_V1
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_SR_Original_Pair_V1_AVX.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain2    = nCharge2%4;
  const size_t nLoop2     = (nCharge2 - remain2)/4;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  double qReCrd_4x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_4x    [4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_4x, 0, 16*nLoop2wRemainder*sizeof(double));
  NPME_TransformQrealCoord_4x_AVX (nCharge2, q2, coord2, qReCrd_4x);


  const __m256d beta_Vec      = _mm256_set1_pd(beta);
  const __m256d minBetaSqVec  = _mm256_set1_pd(-beta*beta);
  const __m256d minCVec       = _mm256_set1_pd(-2.0/sqrt(NPME_Pi)*beta);



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


    size_t index16j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      __m256d q2Vec     = _mm256_load_pd (&qReCrd_4x[index16j   ]);
      __m256d xVec      = _mm256_load_pd (&qReCrd_4x[index16j+ 4]);
      __m256d yVec      = _mm256_load_pd (&qReCrd_4x[index16j+ 8]);
      __m256d zVec      = _mm256_load_pd (&qReCrd_4x[index16j+12]);

      __m256d V0_2_Vec  = _mm256_load_pd (&V2_4x[index16j   ]);
      __m256d VX_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 4]);
      __m256d VY_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 8]);
      __m256d VZ_2_Vec  = _mm256_load_pd (&V2_4x[index16j+12]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      __m256d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
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
        __m256d f1_Vec;

        {
          f0_Vec = _mm256_erfc_pd (_mm256_mul_pd (beta_Vec, rVec));
          f0_Vec = _mm256_div_pd (f0_Vec, rVec);

          //B1 = (-B0 - C0_0*exp(-x2))/x2;
          __m256d expVec = _mm256_exp_pd ( _mm256_mul_pd(minBetaSqVec, r2Vec) );

          #if GL_USE_VECTOR_LIBRARY_FMA
          {
            f1_Vec  = _mm256_fmsub_pd (expVec, minCVec, f0_Vec);
          }
          #else
          {
            f1_Vec  = _mm256_sub_pd (_mm256_mul_pd (expVec, minCVec), f0_Vec);
          }
          #endif
          f1_Vec  = _mm256_div_pd (f1_Vec,  r2Vec);
        }

        fX_Vec  = _mm256_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm256_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm256_mul_pd  (f1_Vec, zVec);
      }

  
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

void NPME_PotLaplace_SR_Original_Self_V1_AVX (const double beta,
  const size_t nCharge, const double *coord, const double *q, double *V,
  bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_SR_Original_Self_V1_AVX.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  if (NPME_Pot_MaxChgBlock_V1%8 != 0)
  {
    printf("Error in NPME_PotLaplace_SR_Original_Self_V1_AVX.\n");
    printf("NPME_Pot_MaxChgBlock_V1 = %lu is not a multiple of 8\n",
      NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain    = nCharge%4;
  const size_t nLoop     = (nCharge - remain)/4;
  size_t nLoopwRemainder = nLoop;
  if (remain > 0)
    nLoopwRemainder++;
  double qReCrd_4x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_4x    [4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_4x, 0, 16*nLoopwRemainder*sizeof(double));
  NPME_TransformQrealCoord_4x_AVX (nCharge, q, coord, qReCrd_4x);



  const __m256d beta_Vec      = _mm256_set1_pd(beta);
  const __m256d minBetaSqVec  = _mm256_set1_pd(-beta*beta);
  const __m256d minCVec       = _mm256_set1_pd(-2.0/sqrt(NPME_Pi)*beta);


  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r1      = &coord[3*i];
    const __m256d q1Vec   = _mm256_set1_pd( q[i]);
    const __m256d q1NVec  = _mm256_set1_pd(-q[i]);
    const __m256d x1Vec   = _mm256_set1_pd(r1[0]);
    const __m256d y1Vec   = _mm256_set1_pd(r1[1]);
    const __m256d z1Vec   = _mm256_set1_pd(r1[2]);

    __m256d V0_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VX_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VY_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VZ_1_Vec      = _mm256_set1_pd(0.0);

    const size_t remain2 = (i)%4;
    const size_t nLoop2  = (i-remain2)/4;

    size_t index16j = 0;
    for (size_t j = 0; j < nLoop2; j++)
    {
      __m256d q2Vec     = _mm256_load_pd (&qReCrd_4x[index16j   ]);
      __m256d xVec      = _mm256_load_pd (&qReCrd_4x[index16j+ 4]);
      __m256d yVec      = _mm256_load_pd (&qReCrd_4x[index16j+ 8]);
      __m256d zVec      = _mm256_load_pd (&qReCrd_4x[index16j+12]);

      __m256d V0_2_Vec  = _mm256_load_pd (&V2_4x[index16j   ]);
      __m256d VX_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 4]);
      __m256d VY_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 8]);
      __m256d VZ_2_Vec  = _mm256_load_pd (&V2_4x[index16j+12]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);


      __m256d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
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
        __m256d f1_Vec;

        {
          f0_Vec = _mm256_erfc_pd (_mm256_mul_pd (beta_Vec, rVec));
          f0_Vec = _mm256_div_pd (f0_Vec, rVec);

          //B1 = (-B0 - C0_0*exp(-x2))/x2;
          __m256d expVec = _mm256_exp_pd ( _mm256_mul_pd(minBetaSqVec, r2Vec) );

          #if GL_USE_VECTOR_LIBRARY_FMA
          {
            f1_Vec  = _mm256_fmsub_pd (expVec, minCVec, f0_Vec);
          }
          #else
          {
            f1_Vec  = _mm256_sub_pd (_mm256_mul_pd (expVec, minCVec), f0_Vec);
          }
          #endif
          f1_Vec  = _mm256_div_pd (f1_Vec,  r2Vec);
        }

        fX_Vec  = _mm256_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm256_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm256_mul_pd  (f1_Vec, zVec);
      }
      
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

      index16j += 16;
    }
    if (remain2 > 0)
    {
      const size_t indexStart = 4*nLoop2;
      const double *qLoc      = &q[indexStart];
      const double *crdLoc    = &coord[3*indexStart];
      const double X          = r1[0]+1.0;
      double x2Array[4]  __attribute__((aligned(64))) = {X, X, X, X};
      double y2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double z2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double q2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double mskArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};


      for (size_t k = 0; k < remain2; k++)
      {
        q2Array[k]  = qLoc[k];
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
        mskArray[k] = 1.0;
      }
      __m256d q2Vec     = _mm256_load_pd (q2Array);
      __m256d xVec      = _mm256_load_pd (x2Array);
      __m256d yVec      = _mm256_load_pd (y2Array);
      __m256d zVec      = _mm256_load_pd (z2Array);
      __m256d mskVec    = _mm256_load_pd (mskArray);
      __m256d V0_2_Vec  = _mm256_load_pd (&V2_4x[index16j   ]);
      __m256d VX_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 4]);
      __m256d VY_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 8]);
      __m256d VZ_2_Vec  = _mm256_load_pd (&V2_4x[index16j+12]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      __m256d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
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
        __m256d f1_Vec;

        {
          f0_Vec = _mm256_erfc_pd (_mm256_mul_pd (beta_Vec, rVec));
          f0_Vec = _mm256_div_pd (f0_Vec, rVec);

          //B1 = (-B0 - C0_0*exp(-x2))/x2;
          __m256d expVec = _mm256_exp_pd ( _mm256_mul_pd(minBetaSqVec, r2Vec) );

          #if GL_USE_VECTOR_LIBRARY_FMA
          {
            f1_Vec  = _mm256_fmsub_pd (expVec, minCVec, f0_Vec);
          }
          #else
          {
            f1_Vec  = _mm256_sub_pd (_mm256_mul_pd (expVec, minCVec), f0_Vec);
          }
          #endif
          f1_Vec  = _mm256_div_pd (f1_Vec,  r2Vec);
        }

        //apply mask
        f0_Vec  = _mm256_mul_pd  (mskVec, f0_Vec);
        f1_Vec  = _mm256_mul_pd  (mskVec, f1_Vec);

        fX_Vec  = _mm256_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm256_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm256_mul_pd  (f1_Vec, zVec);
      }
      
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

      index16j += 16;
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

void NPME_PotLaplace_SR_DM_Pair_V1_AVX (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        a[Nder+1], b[Nder+1]
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_SR_DM_Pair_V1_AVX.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain2    = nCharge2%4;
  const size_t nLoop2     = (nCharge2 - remain2)/4;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  double qReCrd_4x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_4x    [4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_4x, 0, 16*nLoop2wRemainder*sizeof(double));
  NPME_TransformQrealCoord_4x_AVX (nCharge2, q2, coord2, qReCrd_4x);


  const __m256d zeroVec   = _mm256_set1_pd( 0.0);
  const __m256d oneVec    = _mm256_set1_pd( 1.0);
  const __m256d negOneVec = _mm256_set1_pd(-1.0);
  const __m256d RdirVec   = _mm256_set1_pd(Rdir);

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


    size_t index16j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      __m256d q2Vec     = _mm256_load_pd (&qReCrd_4x[index16j   ]);
      __m256d xVec      = _mm256_load_pd (&qReCrd_4x[index16j+ 4]);
      __m256d yVec      = _mm256_load_pd (&qReCrd_4x[index16j+ 8]);
      __m256d zVec      = _mm256_load_pd (&qReCrd_4x[index16j+12]);

      __m256d V0_2_Vec  = _mm256_load_pd (&V2_4x[index16j   ]);
      __m256d VX_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 4]);
      __m256d VY_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 8]);
      __m256d VZ_2_Vec  = _mm256_load_pd (&V2_4x[index16j+12]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      __m256d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
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

        __m256d f1_Vec;
        {
          __m256d f0_AVec, f1_AVec;
          __m256d f0_BVec, f1_BVec;
          NPME_FunctionDerivMatch_EvenSeriesReal_AVX (f0_AVec, f1_AVec, 
            r2Vec, Nder, &a[0], &b[0]);

          f0_BVec = _mm256_div_pd  (oneVec, rVec);
          f1_BVec = _mm256_div_pd  (f0_BVec, r2Vec);
          f1_BVec = _mm256_mul_pd  (f1_BVec, negOneVec);

          f0_BVec = _mm256_sub_pd (f0_BVec, f0_AVec);
          f1_BVec = _mm256_sub_pd (f1_BVec, f1_AVec);

          //use (f0_BVec) if r < Rdir
          //use (zeroVec) if r > Rdir
          {
            __m256d t0, dless, dmore;
            t0      = _mm256_cmp_pd (rVec, RdirVec, 1);

            dless   = _mm256_and_pd    (t0, f0_BVec);
            dmore   = _mm256_andnot_pd (t0, zeroVec);
            f0_Vec  = _mm256_add_pd (dless, dmore);

            dless   = _mm256_and_pd    (t0, f1_BVec);
            dmore   = _mm256_andnot_pd (t0, zeroVec);
            f1_Vec  = _mm256_add_pd (dless, dmore);
          }
        }

        fX_Vec  = _mm256_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm256_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm256_mul_pd  (f1_Vec, zVec);
      }

  
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

void NPME_PotLaplace_SR_DM_Self_V1_AVX (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge, const double *coord, const double *q, double *V,
  bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//        a[Nder+1], b[Nder+1]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_SR_Original_Self_V1_AVX.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  if (NPME_Pot_MaxChgBlock_V1%8 != 0)
  {
    printf("Error in NPME_PotLaplace_SR_Original_Self_V1_AVX.\n");
    printf("NPME_Pot_MaxChgBlock_V1 = %lu is not a multiple of 8\n",
      NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain    = nCharge%4;
  const size_t nLoop     = (nCharge - remain)/4;
  size_t nLoopwRemainder = nLoop;
  if (remain > 0)
    nLoopwRemainder++;
  double qReCrd_4x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_4x    [4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_4x, 0, 16*nLoopwRemainder*sizeof(double));
  NPME_TransformQrealCoord_4x_AVX (nCharge, q, coord, qReCrd_4x);


  const __m256d zeroVec   = _mm256_set1_pd( 0.0);
  const __m256d oneVec    = _mm256_set1_pd( 1.0);
  const __m256d negOneVec = _mm256_set1_pd(-1.0);
  const __m256d RdirVec   = _mm256_set1_pd(Rdir);


  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r1      = &coord[3*i];
    const __m256d q1Vec   = _mm256_set1_pd( q[i]);
    const __m256d q1NVec  = _mm256_set1_pd(-q[i]);
    const __m256d x1Vec   = _mm256_set1_pd(r1[0]);
    const __m256d y1Vec   = _mm256_set1_pd(r1[1]);
    const __m256d z1Vec   = _mm256_set1_pd(r1[2]);

    __m256d V0_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VX_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VY_1_Vec      = _mm256_set1_pd(0.0);
    __m256d VZ_1_Vec      = _mm256_set1_pd(0.0);

    const size_t remain2 = (i)%4;
    const size_t nLoop2  = (i-remain2)/4;

    size_t index16j = 0;
    for (size_t j = 0; j < nLoop2; j++)
    {
      __m256d q2Vec     = _mm256_load_pd (&qReCrd_4x[index16j   ]);
      __m256d xVec      = _mm256_load_pd (&qReCrd_4x[index16j+ 4]);
      __m256d yVec      = _mm256_load_pd (&qReCrd_4x[index16j+ 8]);
      __m256d zVec      = _mm256_load_pd (&qReCrd_4x[index16j+12]);

      __m256d V0_2_Vec  = _mm256_load_pd (&V2_4x[index16j   ]);
      __m256d VX_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 4]);
      __m256d VY_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 8]);
      __m256d VZ_2_Vec  = _mm256_load_pd (&V2_4x[index16j+12]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);


      __m256d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
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

        __m256d f1_Vec;
        {
          __m256d f0_AVec, f1_AVec;
          __m256d f0_BVec, f1_BVec;
          NPME_FunctionDerivMatch_EvenSeriesReal_AVX (f0_AVec, f1_AVec, 
            r2Vec, Nder, &a[0], &b[0]);

          f0_BVec = _mm256_div_pd  (oneVec, rVec);
          f1_BVec = _mm256_div_pd  (f0_BVec, r2Vec);
          f1_BVec = _mm256_mul_pd  (f1_BVec, negOneVec);

          f0_BVec = _mm256_sub_pd (f0_BVec, f0_AVec);
          f1_BVec = _mm256_sub_pd (f1_BVec, f1_AVec);

          //use (f0_BVec) if r < Rdir
          //use (zeroVec) if r > Rdir
          {
            __m256d t0, dless, dmore;
            t0      = _mm256_cmp_pd (rVec, RdirVec, 1);

            dless   = _mm256_and_pd    (t0, f0_BVec);
            dmore   = _mm256_andnot_pd (t0, zeroVec);
            f0_Vec  = _mm256_add_pd (dless, dmore);

            dless   = _mm256_and_pd    (t0, f1_BVec);
            dmore   = _mm256_andnot_pd (t0, zeroVec);
            f1_Vec  = _mm256_add_pd (dless, dmore);
          }
        }

        fX_Vec  = _mm256_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm256_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm256_mul_pd  (f1_Vec, zVec);
      }
      
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

      index16j += 16;
    }
    if (remain2 > 0)
    {
      const size_t indexStart = 4*nLoop2;
      const double *qLoc      = &q[indexStart];
      const double *crdLoc    = &coord[3*indexStart];
      const double X          = r1[0]+1.0;
      double x2Array[4]  __attribute__((aligned(64))) = {X, X, X, X};
      double y2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double z2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double q2Array[4]  __attribute__((aligned(64))) = {0, 0, 0, 0};
      double mskArray[4] __attribute__((aligned(64))) = {0, 0, 0, 0};


      for (size_t k = 0; k < remain2; k++)
      {
        q2Array[k]  = qLoc[k];
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
        mskArray[k] = 1.0;
      }
      __m256d q2Vec     = _mm256_load_pd (q2Array);
      __m256d xVec      = _mm256_load_pd (x2Array);
      __m256d yVec      = _mm256_load_pd (y2Array);
      __m256d zVec      = _mm256_load_pd (z2Array);
      __m256d mskVec    = _mm256_load_pd (mskArray);
      __m256d V0_2_Vec  = _mm256_load_pd (&V2_4x[index16j   ]);
      __m256d VX_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 4]);
      __m256d VY_2_Vec  = _mm256_load_pd (&V2_4x[index16j+ 8]);
      __m256d VZ_2_Vec  = _mm256_load_pd (&V2_4x[index16j+12]);

      xVec = _mm256_sub_pd (x1Vec, xVec);
      yVec = _mm256_sub_pd (y1Vec, yVec);
      zVec = _mm256_sub_pd (z1Vec, zVec);

      __m256d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
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
        __m256d f1_Vec;
        {
          __m256d f0_AVec, f1_AVec;
          __m256d f0_BVec, f1_BVec;
          NPME_FunctionDerivMatch_EvenSeriesReal_AVX (f0_AVec, f1_AVec, 
            r2Vec, Nder, &a[0], &b[0]);

          f0_BVec = _mm256_div_pd  (oneVec, rVec);
          f1_BVec = _mm256_div_pd  (f0_BVec, r2Vec);
          f1_BVec = _mm256_mul_pd  (f1_BVec, negOneVec);

          f0_BVec = _mm256_sub_pd (f0_BVec, f0_AVec);
          f1_BVec = _mm256_sub_pd (f1_BVec, f1_AVec);

          //use (f0_BVec) if r < Rdir
          //use (zeroVec) if r > Rdir
          {
            __m256d t0, dless, dmore;
            t0      = _mm256_cmp_pd (rVec, RdirVec, 1);

            dless   = _mm256_and_pd    (t0, f0_BVec);
            dmore   = _mm256_andnot_pd (t0, zeroVec);
            f0_Vec  = _mm256_add_pd (dless, dmore);

            dless   = _mm256_and_pd    (t0, f1_BVec);
            dmore   = _mm256_andnot_pd (t0, zeroVec);
            f1_Vec  = _mm256_add_pd (dless, dmore);
          }
        }

        //apply mask
        f0_Vec  = _mm256_mul_pd  (mskVec, f0_Vec);
        f1_Vec  = _mm256_mul_pd  (mskVec, f1_Vec);

        fX_Vec  = _mm256_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm256_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm256_mul_pd  (f1_Vec, zVec);
      }
      
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

      index16j += 16;
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

void NPME_PotLaplace_Pair_V1_AVX_512 (
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        nCharge2 <= NPME_Pot_MaxChgBlock_V1
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_Pair_V1_AVX_512.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain2    = nCharge2%8;
  const size_t nLoop2     = (nCharge2 - remain2)/8;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  double qReCrd_8x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_8x    [4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_8x, 0, 32*nLoop2wRemainder*sizeof(double));
  NPME_TransformQrealCoord_8x_AVX_512 (nCharge2, q2, coord2, qReCrd_8x);


  const __m512d negOneVec = _mm512_set1_pd(-1.0);
  const __m512d oneVec    = _mm512_set1_pd( 1.0);



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


    size_t index32j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      __m512d q2Vec     = _mm512_load_pd (&qReCrd_8x[index32j   ]);
      __m512d xVec      = _mm512_load_pd (&qReCrd_8x[index32j+ 8]);
      __m512d yVec      = _mm512_load_pd (&qReCrd_8x[index32j+16]);
      __m512d zVec      = _mm512_load_pd (&qReCrd_8x[index32j+24]);

      __m512d V0_2_Vec  = _mm512_load_pd (&V2_8x[index32j   ]);
      __m512d VX_2_Vec  = _mm512_load_pd (&V2_8x[index32j+ 8]);
      __m512d VY_2_Vec  = _mm512_load_pd (&V2_8x[index32j+16]);
      __m512d VZ_2_Vec  = _mm512_load_pd (&V2_8x[index32j+24]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      __m512d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
        __m512d r2Vec = _mm512_mul_pd  (xVec, xVec);
        r2Vec  = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
        r2Vec  = _mm512_fmadd_pd  (zVec, zVec, r2Vec);

        __m512d rVec  = _mm512_sqrt_pd (r2Vec);
        __m512d f1_Vec;
        f0_Vec  = _mm512_div_pd  (oneVec, rVec);
        f1_Vec  = _mm512_div_pd  (f0_Vec, r2Vec);
        f1_Vec  = _mm512_mul_pd  (f1_Vec, negOneVec);

        fX_Vec  = _mm512_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm512_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm512_mul_pd  (f1_Vec, zVec);
      }

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




void NPME_PotLaplace_Self_V1_AVX_512 (
  const size_t nCharge, const double *coord, const double *q, double *V,
  bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_Self_V1_AVX_512.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  if (NPME_Pot_MaxChgBlock_V1%8 != 0)
  {
    printf("Error in NPME_PotLaplace_Self_V1_AVX_512.\n");
    printf("NPME_Pot_MaxChgBlock_V1 = %lu is not a multiple of 8\n",
      NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain    = nCharge%8;
  const size_t nLoop     = (nCharge - remain)/8;
  size_t nLoopwRemainder = nLoop;
  if (remain > 0)
    nLoopwRemainder++;
  double qReCrd_8x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_8x    [4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_8x, 0, 32*nLoopwRemainder*sizeof(double));
  NPME_TransformQrealCoord_8x_AVX_512 (nCharge, q, coord, qReCrd_8x);



  const __m512d negOneVec = _mm512_set1_pd(-1.0);
  const __m512d oneVec    = _mm512_set1_pd( 1.0);



  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r1      = &coord[3*i];
    const __m512d q1Vec   = _mm512_set1_pd( q[i]);
    const __m512d q1NVec  = _mm512_set1_pd(-q[i]);
    const __m512d x1Vec   = _mm512_set1_pd(r1[0]);
    const __m512d y1Vec   = _mm512_set1_pd(r1[1]);
    const __m512d z1Vec   = _mm512_set1_pd(r1[2]);

    __m512d V0_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VX_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VY_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VZ_1_Vec      = _mm512_set1_pd(0.0);

    const size_t remain2 = (i)%8;
    const size_t nLoop2  = (i-remain2)/8;

    size_t index32j = 0;
    for (size_t j = 0; j < nLoop2; j++)
    {
      __m512d q2Vec     = _mm512_load_pd (&qReCrd_8x[index32j   ]);
      __m512d xVec      = _mm512_load_pd (&qReCrd_8x[index32j+ 8]);
      __m512d yVec      = _mm512_load_pd (&qReCrd_8x[index32j+16]);
      __m512d zVec      = _mm512_load_pd (&qReCrd_8x[index32j+24]);

      __m512d V0_2_Vec  = _mm512_load_pd (&V2_8x[index32j   ]);
      __m512d VX_2_Vec  = _mm512_load_pd (&V2_8x[index32j+ 8]);
      __m512d VY_2_Vec  = _mm512_load_pd (&V2_8x[index32j+16]);
      __m512d VZ_2_Vec  = _mm512_load_pd (&V2_8x[index32j+24]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);


      __m512d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
        __m512d r2Vec = _mm512_mul_pd  (xVec, xVec);
        r2Vec  = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
        r2Vec  = _mm512_fmadd_pd  (zVec, zVec, r2Vec);

        __m512d rVec  = _mm512_sqrt_pd (r2Vec);
        __m512d f1_Vec;
        f0_Vec  = _mm512_div_pd  (oneVec, rVec);
        f1_Vec  = _mm512_div_pd  (f0_Vec, r2Vec);
        f1_Vec  = _mm512_mul_pd  (f1_Vec, negOneVec);

        fX_Vec  = _mm512_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm512_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm512_mul_pd  (f1_Vec, zVec);
      }

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

      index32j += 32;
    }
    if (remain2 > 0)
    {
      const size_t indexStart = 8*nLoop2;
      const double *qLoc      = &q[indexStart];
      const double *crdLoc    = &coord[3*indexStart];
      const double X          = r1[0]+1.0;
      double x2Array[8] __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};
      double y2Array[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
      double z2Array[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
      double q2Array[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
      double mskArray[8] __attribute__((aligned(64)))= {0, 0, 0, 0, 0, 0, 0, 0};

      for (size_t k = 0; k < remain2; k++)
      {
        q2Array[k]  = qLoc[k];
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
        mskArray[k] = 1.0;
      }
      __m512d q2Vec     = _mm512_load_pd (q2Array);
      __m512d xVec      = _mm512_load_pd (x2Array);
      __m512d yVec      = _mm512_load_pd (y2Array);
      __m512d zVec      = _mm512_load_pd (z2Array);
      __m512d mskVec    = _mm512_load_pd (mskArray);
      __m512d V0_2_Vec  = _mm512_load_pd (&V2_8x[index32j   ]);
      __m512d VX_2_Vec  = _mm512_load_pd (&V2_8x[index32j+ 8]);
      __m512d VY_2_Vec  = _mm512_load_pd (&V2_8x[index32j+16]);
      __m512d VZ_2_Vec  = _mm512_load_pd (&V2_8x[index32j+24]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      __m512d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
        __m512d r2Vec = _mm512_mul_pd  (xVec, xVec);
        r2Vec  = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
        r2Vec  = _mm512_fmadd_pd  (zVec, zVec, r2Vec);

        __m512d rVec  = _mm512_sqrt_pd (r2Vec);
        __m512d f1_Vec;
        f0_Vec  = _mm512_div_pd  (oneVec, rVec);
        f1_Vec  = _mm512_div_pd  (f0_Vec, r2Vec);
        f1_Vec  = _mm512_mul_pd  (f1_Vec, negOneVec);

        //apply mask
        f0_Vec  = _mm512_mul_pd  (mskVec, f0_Vec);
        f1_Vec  = _mm512_mul_pd  (mskVec, f1_Vec);

        fX_Vec  = _mm512_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm512_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm512_mul_pd  (f1_Vec, zVec);
      }
      
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

      index32j += 32;
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



void NPME_PotLaplace_SR_Original_Pair_V1_AVX_512 (const double beta,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        nCharge2 <= NPME_Pot_MaxChgBlock_V1
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_SR_Original_Pair_V1_AVX_512.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain2    = nCharge2%8;
  const size_t nLoop2     = (nCharge2 - remain2)/8;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  double qReCrd_8x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_8x    [4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_8x, 0, 32*nLoop2wRemainder*sizeof(double));
  NPME_TransformQrealCoord_8x_AVX_512 (nCharge2, q2, coord2, qReCrd_8x);

  const __m512d beta_Vec      = _mm512_set1_pd(beta);
  const __m512d minBetaSqVec  = _mm512_set1_pd(-beta*beta);
  const __m512d minCVec       = _mm512_set1_pd(-2.0/sqrt(NPME_Pi)*beta);


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


    size_t index32j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      __m512d q2Vec     = _mm512_load_pd (&qReCrd_8x[index32j   ]);
      __m512d xVec      = _mm512_load_pd (&qReCrd_8x[index32j+ 8]);
      __m512d yVec      = _mm512_load_pd (&qReCrd_8x[index32j+16]);
      __m512d zVec      = _mm512_load_pd (&qReCrd_8x[index32j+24]);

      __m512d V0_2_Vec  = _mm512_load_pd (&V2_8x[index32j   ]);
      __m512d VX_2_Vec  = _mm512_load_pd (&V2_8x[index32j+ 8]);
      __m512d VY_2_Vec  = _mm512_load_pd (&V2_8x[index32j+16]);
      __m512d VZ_2_Vec  = _mm512_load_pd (&V2_8x[index32j+24]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      __m512d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
        __m512d r2Vec = _mm512_mul_pd  (xVec, xVec);
        r2Vec  = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
        r2Vec  = _mm512_fmadd_pd  (zVec, zVec, r2Vec);

        __m512d rVec  = _mm512_sqrt_pd (r2Vec);
        __m512d f1_Vec;
        {
          __m512d expVec;
          f0_Vec = _mm512_erfc_pd (_mm512_mul_pd (beta_Vec, rVec));
          f0_Vec = _mm512_div_pd (f0_Vec, rVec);

          //B1 = (-B0 - C0_0*exp(-x2))/x2;
          expVec  = _mm512_exp_pd ( _mm512_mul_pd(minBetaSqVec, r2Vec) );
          f1_Vec  = _mm512_fmsub_pd (expVec, minCVec, f0_Vec);
          f1_Vec  = _mm512_div_pd (f1_Vec,  r2Vec);
        }

        fX_Vec  = _mm512_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm512_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm512_mul_pd  (f1_Vec, zVec);
      }

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




void NPME_PotLaplace_SR_Original_Self_V1_AVX_512 (const double beta,
  const size_t nCharge, const double *coord, const double *q, double *V,
  bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_SR_Original_Self_V1_AVX_512.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  if (NPME_Pot_MaxChgBlock_V1%8 != 0)
  {
    printf("Error in NPME_PotLaplace_SR_Original_Self_V1_AVX_512.\n");
    printf("NPME_Pot_MaxChgBlock_V1 = %lu is not a multiple of 8\n",
      NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain    = nCharge%8;
  const size_t nLoop     = (nCharge - remain)/8;
  size_t nLoopwRemainder = nLoop;
  if (remain > 0)
    nLoopwRemainder++;
  double qReCrd_8x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_8x    [4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_8x, 0, 32*nLoopwRemainder*sizeof(double));
  NPME_TransformQrealCoord_8x_AVX_512 (nCharge, q, coord, qReCrd_8x);


  const __m512d beta_Vec      = _mm512_set1_pd(beta);
  const __m512d minBetaSqVec  = _mm512_set1_pd(-beta*beta);
  const __m512d minCVec       = _mm512_set1_pd(-2.0/sqrt(NPME_Pi)*beta);


  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r1      = &coord[3*i];
    const __m512d q1Vec   = _mm512_set1_pd( q[i]);
    const __m512d q1NVec  = _mm512_set1_pd(-q[i]);
    const __m512d x1Vec   = _mm512_set1_pd(r1[0]);
    const __m512d y1Vec   = _mm512_set1_pd(r1[1]);
    const __m512d z1Vec   = _mm512_set1_pd(r1[2]);

    __m512d V0_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VX_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VY_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VZ_1_Vec      = _mm512_set1_pd(0.0);

    const size_t remain2 = (i)%8;
    const size_t nLoop2  = (i-remain2)/8;

    size_t index32j = 0;
    for (size_t j = 0; j < nLoop2; j++)
    {
      __m512d q2Vec     = _mm512_load_pd (&qReCrd_8x[index32j   ]);
      __m512d xVec      = _mm512_load_pd (&qReCrd_8x[index32j+ 8]);
      __m512d yVec      = _mm512_load_pd (&qReCrd_8x[index32j+16]);
      __m512d zVec      = _mm512_load_pd (&qReCrd_8x[index32j+24]);

      __m512d V0_2_Vec  = _mm512_load_pd (&V2_8x[index32j   ]);
      __m512d VX_2_Vec  = _mm512_load_pd (&V2_8x[index32j+ 8]);
      __m512d VY_2_Vec  = _mm512_load_pd (&V2_8x[index32j+16]);
      __m512d VZ_2_Vec  = _mm512_load_pd (&V2_8x[index32j+24]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);


      __m512d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
        __m512d r2Vec = _mm512_mul_pd  (xVec, xVec);
        r2Vec  = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
        r2Vec  = _mm512_fmadd_pd  (zVec, zVec, r2Vec);

        __m512d rVec  = _mm512_sqrt_pd (r2Vec);
        __m512d f1_Vec;
        {
          __m512d expVec;
          f0_Vec = _mm512_erfc_pd (_mm512_mul_pd (beta_Vec, rVec));
          f0_Vec = _mm512_div_pd (f0_Vec, rVec);

          //B1 = (-B0 - C0_0*exp(-x2))/x2;
          expVec  = _mm512_exp_pd ( _mm512_mul_pd(minBetaSqVec, r2Vec) );
          f1_Vec  = _mm512_fmsub_pd (expVec, minCVec, f0_Vec);
          f1_Vec  = _mm512_div_pd (f1_Vec,  r2Vec);
        }

        fX_Vec  = _mm512_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm512_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm512_mul_pd  (f1_Vec, zVec);
      }

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

      index32j += 32;
    }
    if (remain2 > 0)
    {
      const size_t indexStart = 8*nLoop2;
      const double *qLoc      = &q[indexStart];
      const double *crdLoc    = &coord[3*indexStart];
      const double X          = r1[0]+1.0;
      double x2Array[8] __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};
      double y2Array[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
      double z2Array[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
      double q2Array[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
      double mskArray[8] __attribute__((aligned(64)))= {0, 0, 0, 0, 0, 0, 0, 0};

      for (size_t k = 0; k < remain2; k++)
      {
        q2Array[k]  = qLoc[k];
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
        mskArray[k] = 1.0;
      }
      __m512d q2Vec     = _mm512_load_pd (q2Array);
      __m512d xVec      = _mm512_load_pd (x2Array);
      __m512d yVec      = _mm512_load_pd (y2Array);
      __m512d zVec      = _mm512_load_pd (z2Array);
      __m512d mskVec    = _mm512_load_pd (mskArray);
      __m512d V0_2_Vec  = _mm512_load_pd (&V2_8x[index32j   ]);
      __m512d VX_2_Vec  = _mm512_load_pd (&V2_8x[index32j+ 8]);
      __m512d VY_2_Vec  = _mm512_load_pd (&V2_8x[index32j+16]);
      __m512d VZ_2_Vec  = _mm512_load_pd (&V2_8x[index32j+24]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      __m512d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
        __m512d r2Vec = _mm512_mul_pd  (xVec, xVec);
        r2Vec  = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
        r2Vec  = _mm512_fmadd_pd  (zVec, zVec, r2Vec);

        __m512d rVec  = _mm512_sqrt_pd (r2Vec);
        __m512d f1_Vec;
        {
          __m512d expVec;
          f0_Vec = _mm512_erfc_pd (_mm512_mul_pd (beta_Vec, rVec));
          f0_Vec = _mm512_div_pd (f0_Vec, rVec);

          //B1 = (-B0 - C0_0*exp(-x2))/x2;
          expVec  = _mm512_exp_pd ( _mm512_mul_pd(minBetaSqVec, r2Vec) );
          f1_Vec  = _mm512_fmsub_pd (expVec, minCVec, f0_Vec);
          f1_Vec  = _mm512_div_pd (f1_Vec,  r2Vec);
        }

        //apply mask
        f0_Vec  = _mm512_mul_pd  (mskVec, f0_Vec);
        f1_Vec  = _mm512_mul_pd  (mskVec, f1_Vec);

        fX_Vec  = _mm512_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm512_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm512_mul_pd  (f1_Vec, zVec);
      }
      
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

      index32j += 32;
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


void NPME_PotLaplace_SR_DM_Pair_V1_AVX_512 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        a[Nder+1], b[Nder+1]
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge2 > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_SR_DM_Pair_V1_AVX_512.\n");
    printf("nCharge2 = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge2, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain2    = nCharge2%8;
  const size_t nLoop2     = (nCharge2 - remain2)/8;
  size_t nLoop2wRemainder = nLoop2;
  if (remain2 > 0)
    nLoop2wRemainder++;
  double qReCrd_8x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_8x    [4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_8x, 0, 32*nLoop2wRemainder*sizeof(double));
  NPME_TransformQrealCoord_8x_AVX_512 (nCharge2, q2, coord2, qReCrd_8x);


  const __m512d zeroVec   = _mm512_set1_pd( 0.0);
  const __m512d oneVec    = _mm512_set1_pd( 1.0);
  const __m512d negOneVec = _mm512_set1_pd(-1.0);
  const __m512d RdirVec   = _mm512_set1_pd(Rdir);



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


    size_t index32j = 0;
    for (size_t j = 0; j < nLoop2wRemainder; j++)
    {
      __m512d q2Vec     = _mm512_load_pd (&qReCrd_8x[index32j   ]);
      __m512d xVec      = _mm512_load_pd (&qReCrd_8x[index32j+ 8]);
      __m512d yVec      = _mm512_load_pd (&qReCrd_8x[index32j+16]);
      __m512d zVec      = _mm512_load_pd (&qReCrd_8x[index32j+24]);

      __m512d V0_2_Vec  = _mm512_load_pd (&V2_8x[index32j   ]);
      __m512d VX_2_Vec  = _mm512_load_pd (&V2_8x[index32j+ 8]);
      __m512d VY_2_Vec  = _mm512_load_pd (&V2_8x[index32j+16]);
      __m512d VZ_2_Vec  = _mm512_load_pd (&V2_8x[index32j+24]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      __m512d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
        __m512d r2Vec = _mm512_mul_pd  (xVec, xVec);
        r2Vec  = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
        r2Vec  = _mm512_fmadd_pd  (zVec, zVec, r2Vec);

        __m512d rVec  = _mm512_sqrt_pd (r2Vec);
        __m512d f1_Vec;
        {
          __m512d f0_AVec, f1_AVec;
          __m512d f0_BVec, f1_BVec;
          NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (f0_AVec, f1_AVec, 
            r2Vec, Nder, &a[0], &b[0]);

          f0_BVec = _mm512_div_pd  (oneVec, rVec);
          f1_BVec = _mm512_div_pd  (f0_BVec, r2Vec);
          f1_BVec = _mm512_mul_pd  (f1_BVec, negOneVec);

          f0_BVec = _mm512_sub_pd (f0_BVec, f0_AVec);
          f1_BVec = _mm512_sub_pd (f1_BVec, f1_AVec);

          //use (f0_AVec) if r < Rdir
          //use (f0_BVec) if r > Rdir
          {
            __mmask8 maskVec = _mm512_cmp_pd_mask (rVec, RdirVec, 1);
            f0_Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f0_BVec);
            f1_Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f1_BVec);
          }
        }

        fX_Vec  = _mm512_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm512_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm512_mul_pd  (f1_Vec, zVec);
      }

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




void NPME_PotLaplace_SR_DM_Self_V1_AVX_512 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge, const double *coord, const double *q, double *V,
  bool zeroArray)
//input:  coord[nCharge*3], q[nCharge]
//        a[Nder+1], b[Nder+1]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (nCharge > NPME_Pot_MaxChgBlock_V1)
  {
    printf("Error in NPME_PotLaplace_SR_DM_Self_V1_AVX_512.\n");
    printf("nCharge = %lu > %lu = NPME_Pot_MaxChgBlock_V1\n", 
      nCharge, NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  if (NPME_Pot_MaxChgBlock_V1%8 != 0)
  {
    printf("Error in NPME_PotLaplace_SR_DM_Self_V1_AVX_512.\n");
    printf("NPME_Pot_MaxChgBlock_V1 = %lu is not a multiple of 8\n",
      NPME_Pot_MaxChgBlock_V1);
    exit(0);
  }
  const size_t remain    = nCharge%8;
  const size_t nLoop     = (nCharge - remain)/8;
  size_t nLoopwRemainder = nLoop;
  if (remain > 0)
    nLoopwRemainder++;
  double qReCrd_8x[4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  double V2_8x    [4*NPME_Pot_MaxChgBlock_V1] __attribute__((aligned(64)));
  memset(V2_8x, 0, 32*nLoopwRemainder*sizeof(double));
  NPME_TransformQrealCoord_8x_AVX_512 (nCharge, q, coord, qReCrd_8x);



  const __m512d zeroVec   = _mm512_set1_pd( 0.0);
  const __m512d oneVec    = _mm512_set1_pd( 1.0);
  const __m512d negOneVec = _mm512_set1_pd(-1.0);
  const __m512d RdirVec   = _mm512_set1_pd(Rdir);



  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r1      = &coord[3*i];
    const __m512d q1Vec   = _mm512_set1_pd( q[i]);
    const __m512d q1NVec  = _mm512_set1_pd(-q[i]);
    const __m512d x1Vec   = _mm512_set1_pd(r1[0]);
    const __m512d y1Vec   = _mm512_set1_pd(r1[1]);
    const __m512d z1Vec   = _mm512_set1_pd(r1[2]);

    __m512d V0_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VX_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VY_1_Vec      = _mm512_set1_pd(0.0);
    __m512d VZ_1_Vec      = _mm512_set1_pd(0.0);

    const size_t remain2 = (i)%8;
    const size_t nLoop2  = (i-remain2)/8;

    size_t index32j = 0;
    for (size_t j = 0; j < nLoop2; j++)
    {
      __m512d q2Vec     = _mm512_load_pd (&qReCrd_8x[index32j   ]);
      __m512d xVec      = _mm512_load_pd (&qReCrd_8x[index32j+ 8]);
      __m512d yVec      = _mm512_load_pd (&qReCrd_8x[index32j+16]);
      __m512d zVec      = _mm512_load_pd (&qReCrd_8x[index32j+24]);

      __m512d V0_2_Vec  = _mm512_load_pd (&V2_8x[index32j   ]);
      __m512d VX_2_Vec  = _mm512_load_pd (&V2_8x[index32j+ 8]);
      __m512d VY_2_Vec  = _mm512_load_pd (&V2_8x[index32j+16]);
      __m512d VZ_2_Vec  = _mm512_load_pd (&V2_8x[index32j+24]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);


      __m512d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
        __m512d r2Vec = _mm512_mul_pd  (xVec, xVec);
        r2Vec  = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
        r2Vec  = _mm512_fmadd_pd  (zVec, zVec, r2Vec);

        __m512d rVec  = _mm512_sqrt_pd (r2Vec);
        __m512d f1_Vec;
        {
          __m512d f0_AVec, f1_AVec;
          __m512d f0_BVec, f1_BVec;
          NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (f0_AVec, f1_AVec, 
            r2Vec, Nder, &a[0], &b[0]);

          f0_BVec = _mm512_div_pd  (oneVec, rVec);
          f1_BVec = _mm512_div_pd  (f0_BVec, r2Vec);
          f1_BVec = _mm512_mul_pd  (f1_BVec, negOneVec);

          f0_BVec = _mm512_sub_pd (f0_BVec, f0_AVec);
          f1_BVec = _mm512_sub_pd (f1_BVec, f1_AVec);

          //use (f0_AVec) if r < Rdir
          //use (f0_BVec) if r > Rdir
          {
            __mmask8 maskVec = _mm512_cmp_pd_mask (rVec, RdirVec, 1);
            f0_Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f0_BVec);
            f1_Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f1_BVec);
          }
        }

        fX_Vec  = _mm512_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm512_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm512_mul_pd  (f1_Vec, zVec);
      }

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

      index32j += 32;
    }
    if (remain2 > 0)
    {
      const size_t indexStart = 8*nLoop2;
      const double *qLoc      = &q[indexStart];
      const double *crdLoc    = &coord[3*indexStart];
      const double X          = r1[0] + 1.0;
      double x2Array[8] __attribute__((aligned(64))) = {X, X, X, X, X, X, X, X};
      double y2Array[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
      double z2Array[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
      double q2Array[8] __attribute__((aligned(64))) = {0, 0, 0, 0, 0, 0, 0, 0};
      double mskArray[8] __attribute__((aligned(64)))= {0, 0, 0, 0, 0, 0, 0, 0};

      for (size_t k = 0; k < remain2; k++)
      {
        q2Array[k]  = qLoc[k];
        x2Array[k]  = crdLoc[3*k  ];
        y2Array[k]  = crdLoc[3*k+1];
        z2Array[k]  = crdLoc[3*k+2];
        mskArray[k] = 1.0;
      }
      __m512d q2Vec     = _mm512_load_pd (q2Array);
      __m512d xVec      = _mm512_load_pd (x2Array);
      __m512d yVec      = _mm512_load_pd (y2Array);
      __m512d zVec      = _mm512_load_pd (z2Array);
      __m512d mskVec    = _mm512_load_pd (mskArray);
      __m512d V0_2_Vec  = _mm512_load_pd (&V2_8x[index32j   ]);
      __m512d VX_2_Vec  = _mm512_load_pd (&V2_8x[index32j+ 8]);
      __m512d VY_2_Vec  = _mm512_load_pd (&V2_8x[index32j+16]);
      __m512d VZ_2_Vec  = _mm512_load_pd (&V2_8x[index32j+24]);

      xVec = _mm512_sub_pd (x1Vec, xVec);
      yVec = _mm512_sub_pd (y1Vec, yVec);
      zVec = _mm512_sub_pd (z1Vec, zVec);

      __m512d f0_Vec, fX_Vec, fY_Vec, fZ_Vec;
      {
        __m512d r2Vec = _mm512_mul_pd  (xVec, xVec);
        r2Vec  = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
        r2Vec  = _mm512_fmadd_pd  (zVec, zVec, r2Vec);

        __m512d rVec  = _mm512_sqrt_pd (r2Vec);
        __m512d f1_Vec;
        {
          __m512d f0_AVec, f1_AVec;
          __m512d f0_BVec, f1_BVec;
          NPME_FunctionDerivMatch_EvenSeriesReal_AVX_512 (f0_AVec, f1_AVec, 
            r2Vec, Nder, &a[0], &b[0]);
          f0_BVec = _mm512_div_pd  (oneVec, rVec);
          f1_BVec = _mm512_div_pd  (f0_BVec, r2Vec);
          f1_BVec = _mm512_mul_pd  (f1_BVec, negOneVec);

          f0_BVec = _mm512_sub_pd (f0_BVec, f0_AVec);
          f1_BVec = _mm512_sub_pd (f1_BVec, f1_AVec);

          //use (f0_AVec) if r < Rdir
          //use (f0_BVec) if r > Rdir
          {
            __mmask8 maskVec = _mm512_cmp_pd_mask (rVec, RdirVec, 1);
            f0_Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f0_BVec);
            f1_Vec = _mm512_mask_mov_pd (zeroVec, maskVec, f1_BVec);
          }

          //apply mask
          f0_Vec = _mm512_mul_pd  (mskVec, f0_Vec);
          f1_Vec = _mm512_mul_pd  (mskVec, f1_Vec);
        }

        fX_Vec  = _mm512_mul_pd  (f1_Vec, xVec);
        fY_Vec  = _mm512_mul_pd  (f1_Vec, yVec);
        fZ_Vec  = _mm512_mul_pd  (f1_Vec, zVec);
      }
      
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

      index32j += 32;
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




void NPME_PotLaplace_Pair_V1 (
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, int vecOption, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (vecOption == 0)
  {
    NPME_PotLaplace_Pair_V1 (
      nCharge1, coord1, q1,
      nCharge2, coord2, q2, V1, V2, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotLaplace_Pair_V1_AVX (
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotLaplace_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotLaplace_Pair_V1_AVX_512 (
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotLaplace_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}

void NPME_PotLaplace_Self_V1 (const size_t nCharge, 
  const double *coord, const double *q, double *V, int vecOption, 
  bool zeroArray)
//input:  coord[nCharge*3], q[nCharge1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (vecOption == 0)
  {
    NPME_PotLaplace_Self_V1 (nCharge, coord, q, V, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotLaplace_Self_V1_AVX (nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotLaplace_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotLaplace_Self_V1_AVX_512 (nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotLaplace_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}



void NPME_PotLaplace_SR_DM_Pair_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, int vecOption, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        a[Nder+1], b[Nder+1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (vecOption == 0)
  {
    NPME_PotLaplace_SR_DM_Pair_V1 (
      Nder, a, b, Rdir,
      nCharge1, coord1, q1,
      nCharge2, coord2, q2, V1, V2, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotLaplace_SR_DM_Pair_V1_AVX (
        Nder, a, b, Rdir,
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotLaplace_SR_DM_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotLaplace_SR_DM_Pair_V1_AVX_512 (
        Nder, a, b, Rdir,
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotLaplace_SR_DM_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}

void NPME_PotLaplace_SR_DM_Self_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge, const double *coord, const double *q, double *V, 
  int vecOption, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge1]
//        a[Nder+1], b[Nder+1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (vecOption == 0)
  {
    NPME_PotLaplace_SR_DM_Self_V1 (
      Nder, a, b, Rdir,
      nCharge, coord, q, V, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotLaplace_SR_DM_Self_V1_AVX (
        Nder, a, b, Rdir,
        nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotLaplace_SR_DM_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotLaplace_SR_DM_Self_V1_AVX_512 (
        Nder, a, b, Rdir,
        nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotLaplace_SR_DM_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}

void NPME_PotLaplace_SR_DM_LargePair_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge1, const double *coord1, const double *charge1,
  const size_t nCharge2, const double *coord2, const double *charge2,
  double *V1, double *V2, int vecOption, size_t blockSize, bool zeroArray)
{
  if (nCharge2 < blockSize)
    return NPME_PotLaplace_SR_DM_Pair_V1 (Nder, a, b, Rdir,
                nCharge1, coord1, charge1, 
                nCharge2, coord2, charge2, 
                V1, V2, vecOption, zeroArray);

  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotLaplace_SR_DM_LargePair_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotLaplace_SR_DM_LargePair_V1\n";
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
    NPME_PotLaplace_SR_DM_Pair_V1 (
      Nder, a, b, Rdir,
      nCharge1B, coord1B, charge1B, 
      nCharge2B, coord2B, charge2B, 
      V1_B, V2_B, vecOption, zeroArrayB);
  }
}


void NPME_PotLaplace_SR_DM_LargeSelf_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge, const double *coord, const double *charge,
  double *V, int vecOption, size_t blockSize, bool zeroArray)
{
  if (nCharge < blockSize)
    return NPME_PotLaplace_SR_DM_Self_V1 (Nder, a, b, Rdir,
        nCharge, coord, charge, V, vecOption, zeroArray);


  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotLaplace_SR_DM_LargeSelf_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotLaplace_SR_DM_LargeSelf_V1\n";
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
      NPME_PotLaplace_SR_DM_Pair_V1 (
        Nder, a, b, Rdir,
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
      NPME_PotLaplace_SR_DM_Self_V1 (
        Nder, a, b, Rdir,
        nCharge1B, coord1B, charge1B, V1B, vecOption, zeroArrayB);
    }
  }
}
void NPME_PotLaplace_SR_Original_Pair_V1 (const double beta,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, int vecOption, bool zeroArray)
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (vecOption == 0)
  {
    NPME_PotLaplace_SR_Original_Pair_V1 (beta,
      nCharge1, coord1, q1,
      nCharge2, coord2, q2, V1, V2, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotLaplace_SR_Original_Pair_V1_AVX (beta,
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotLaplace_SR_Original_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotLaplace_SR_Original_Pair_V1_AVX_512 (beta,
        nCharge1, coord1, q1,
        nCharge2, coord2, q2, V1, V2, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotLaplace_SR_Original_Pair_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}


void NPME_PotLaplace_SR_Original_Self_V1 (const double beta,
  const size_t nCharge, const double *coord, const double *q, double *V, 
  int vecOption, bool zeroArray)
//input:  coord[nCharge*3], q[nCharge1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
{
  if (vecOption == 0)
  {
    NPME_PotLaplace_SR_Original_Self_V1 (beta,
      nCharge, coord, q, V, zeroArray);
  }
  else if (vecOption == 1)
  {
    #if NPME_USE_AVX
    {
      NPME_PotLaplace_SR_Original_Self_V1_AVX (beta,
        nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotLaplace_SR_Original_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
  else if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    {
      NPME_PotLaplace_SR_Original_Self_V1_AVX_512 (beta,
        nCharge, coord, q, V, zeroArray);
    }
    #else
    {
      printf("Error in NPME_PotLaplace_SR_Original_Self_V1.\n");
      printf("vecOption = %d but NPME_USE_AVX_512 = 0\n", vecOption);
      exit(0);
    }
    #endif
  }
}

void NPME_PotLaplace_SR_Original_LargePair_V1 (const double beta,
  const size_t nCharge1, const double *coord1, const double *charge1,
  const size_t nCharge2, const double *coord2, const double *charge2,
  double *V1, double *V2, int vecOption, size_t blockSize, bool zeroArray)
{
  if (nCharge2 < blockSize)
    return NPME_PotLaplace_SR_Original_Pair_V1 (beta,
                nCharge1, coord1, charge1, 
                nCharge2, coord2, charge2, 
                V1, V2, vecOption, zeroArray);


  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotLaplace_SR_Original_LargePair_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotLaplace_SR_Original_LargePair_V1\n";
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
    NPME_PotLaplace_SR_Original_Pair_V1 (beta,
      nCharge1B, coord1B, charge1B, 
      nCharge2B, coord2B, charge2B, 
      V1_B, V2_B, vecOption, zeroArrayB);
  }
}


void NPME_PotLaplace_SR_Original_LargeSelf_V1 (const double beta,
  const size_t nCharge, const double *coord, const double *charge,
  double *V, int vecOption, size_t blockSize, bool zeroArray)
{
  if (nCharge < blockSize)
    return NPME_PotLaplace_SR_Original_Self_V1 (beta,
        nCharge, coord, charge, V, vecOption, zeroArray);

  if (blockSize%8 != 0)
  {
    char str[2000];
    std::cout << "Error in NPME_PotLaplace_SR_Original_LargeSelf_V1\n";
    sprintf(str, "blockSize = %lu is not a multiple of 8\n", blockSize);
    std::cout << str;
    exit(0);
  }
  if (blockSize > NPME_Pot_MaxChgBlock_V1)
  {
    char str[2000];
    std::cout << "Error in NPME_PotLaplace_SR_Original_LargeSelf_V1\n";
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
      NPME_PotLaplace_SR_Original_Pair_V1 (beta,
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
      NPME_PotLaplace_SR_Original_Self_V1 (beta,
        nCharge1B, coord1B, charge1B, V1B, vecOption, zeroArrayB);
    }
  }
}






}//end namespace NPME_Library



