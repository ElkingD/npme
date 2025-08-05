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

#ifndef NPME_POTENTIAL_LAPLACE_H
#define NPME_POTENTIAL_LAPLACE_H



#include "Constant.h"
#include "PartitionBox.h"
#include "PartitionEmbeddedBox.h"

namespace NPME_Library
{

void NPME_PotLaplace_MacroSelf_V1 (
  const size_t nCharge, const double *coord, const double *Q1, 
  double *V1, const int nProc, const int vecOption, 
  const size_t blockSize = NPME_Pot_MaxChgBlock_V1);


void NPME_PotLaplace_SR_DM_DirectSum_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge, const double *coord, const double *charge, 
  double *V, const int nProc, const int vecOption,
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster, 
  const size_t blockSize = NPME_Pot_MaxChgBlock_V1);

void NPME_PotLaplace_SR_Original_DirectSum_V1 (const double beta,
  const size_t nCharge, const double *coord, const double *charge, 
  double *V, const int nProc, const int vecOption,
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster, 
  const size_t blockSize = NPME_Pot_MaxChgBlock_V1);

//nCharge <= NPME_Pot_MaxChgBlock for the following low level functions


void NPME_PotLaplace_Pair_V1 (
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, int vecOption, 
  bool zeroArray = 1);
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[nCharge1][4]
//        V2[nCharge2][4]
//        V[nCharge][4] = (V0[0], VX[0], VY[0], VZ[0],
//                         V0[1], VX[1], VY[1], VZ[1],..)

void NPME_PotLaplace_Self_V1 (const size_t nCharge, 
  const double *coord, const double *q, double *V, int vecOption, 
  bool zeroArray = 1);
//input:  coord[nCharge*3], q[nCharge1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//input:  coord[nCharge*3], q[nCharge]
//        V[nCharge][4] = (V0[0], VX[0], VY[0], VZ[0],
//                         V0[1], VX[1], VY[1], VZ[1],..)


//**************Short Range Derivative Match Direct Sum Functions***************
void NPME_PotLaplace_SR_DM_Pair_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, int vecOption, bool zeroArray = 1);
//nCharge <= NPME_Pot_MaxChgBlock_V1
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        a[Nder+1], b[Nder+1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
void NPME_PotLaplace_SR_DM_Self_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge, const double *coord, const double *q, double *V, 
  int vecOption, bool zeroArray = 1);
//nCharge <= NPME_Pot_MaxChgBlock_V1
//input:  coord[nCharge*3], q[nCharge1]
//        a[Nder+1], b[Nder+1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])


void NPME_PotLaplace_SR_DM_LargePair_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge1, const double *coord1, const double *charge1,
  const size_t nCharge2, const double *coord2, const double *charge2,
  double *V1, double *V2, int vecOption, 
  size_t blockSize = NPME_Pot_MaxChgBlock_V1, bool zeroArray = 1);
//no restriction on nCharge
//blockSize <= NPME_Pot_MaxChgBlock_V1
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        a[Nder+1], b[Nder+1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])

void NPME_PotLaplace_SR_DM_LargeSelf_V1 (
  const int Nder, const double *a, const double *b, const double Rdir,
  const size_t nCharge, const double *coord, const double *charge,
  double *V, int vecOption, 
  size_t blockSize = NPME_Pot_MaxChgBlock_V1, bool zeroArray = 1);

//no restriction on nCharge
//blockSize <= NPME_Pot_MaxChgBlock_V1
//input:  coord[nCharge*3], q[nCharge1]
//        a[Nder+1], b[Nder+1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])

//**************Original Ewald Split Direct Sum Functions***********************
void NPME_PotLaplace_SR_Original_Pair_V1 (const double beta,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, int vecOption, bool zeroArray = 1);
//nCharge <= NPME_Pot_MaxChgBlock_V1
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])
void NPME_PotLaplace_SR_Original_Self_V1 (const double beta,
  const size_t nCharge, const double *coord, const double *q, double *V, 
  int vecOption, bool zeroArray = 1);
//nCharge <= NPME_Pot_MaxChgBlock_V1
//input:  coord[nCharge*3], q[nCharge1]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])

void NPME_PotLaplace_SR_Original_LargePair_V1 (const double beta,
  const size_t nCharge1, const double *coord1, const double *charge1,
  const size_t nCharge2, const double *coord2, const double *charge2,
  double *V1, double *V2, int vecOption, 
  size_t blockSize = NPME_Pot_MaxChgBlock_V1, bool zeroArray = 1);
//no restriction on nCharge
//blockSize <= NPME_Pot_MaxChgBlock_V1
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])

void NPME_PotLaplace_SR_Original_LargeSelf_V1 (const double beta,
  const size_t nCharge, const double *coord, const double *charge,
  double *V, int vecOption, 
  size_t blockSize = NPME_Pot_MaxChgBlock_V1, bool zeroArray = 1);
//no restriction on nCharge
//blockSize <= NPME_Pot_MaxChgBlock_V1
//input:  coord[nCharge*3], q[nCharge1]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])


}//end namespace NPME_Library


#endif // NPME_POTENTIAL_LAPLACE_H







