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

#ifndef NPME_POTENTIAL_GEN_FUNC_H
#define NPME_POTENTIAL_GEN_FUNC_H



#include "Constant.h"
#include "KernelFunction.h"
#include "PartitionBox.h"
#include "PartitionEmbeddedBox.h"

namespace NPME_Library
{
void NPME_PotGenFunc_AddSelfTerm_V (const NPME_Library::NPME_KfuncReal& funcLR, 
  const size_t nCharge, const double *Q, double *V);
void NPME_PotGenFunc_AddSelfTerm_V (
  const NPME_Library::NPME_KfuncComplex& funcLR, 
  const size_t nCharge, const _Complex double *Q, _Complex double *V);
//input:  Q[nCharge]
//output: V[nCharge][4]

void NPME_PotGenFunc_DirectSum_V1 (const NPME_Library::NPME_KfuncReal& func, 
  const size_t nCharge, const double *coord, const double *charge, 
  double *V, const int nProc, const int vecOption,
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster, 
  const size_t blockSize = NPME_Pot_MaxChgBlock_V1);
void NPME_PotGenFunc_DirectSum_V1 (const NPME_Library::NPME_KfuncComplex& func, 
  const size_t nCharge, const double *coord, const _Complex double *charge, 
  _Complex double *V, const int nProc, const int vecOption,
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster, 
  const size_t blockSize = NPME_Pot_MaxChgBlock_V1);
//input:  Q[nCharge]
//output: V[nCharge][4]

//the optimal settings for NPME_PotGenFunc_DirectSum_V1 are:
//  1) hyper-threading preferred over nProc = num of physical cores
//  2) nNeigh = 2       (cluster input)
//  3) nCellClust1D = 3 (cluster input)






void NPME_PotGenFunc_MacroSelf_V1 (const NPME_Library::NPME_KfuncReal& func, 
  const size_t nCharge, const double *coord, const double *Q1, 
  double *V1, const int nProc, const int vecOption, 
  const size_t blockSize = NPME_Pot_MaxChgBlock_V1);
void NPME_PotGenFunc_MacroSelf_V1 (const NPME_Library::NPME_KfuncComplex& func, 
  const size_t nCharge, const double *coord, const _Complex double *Q1, 
  _Complex double *V1, const int nProc, const int vecOption, 
  const size_t blockSize = NPME_Pot_MaxChgBlock_V1);











//************************Real Low Level V1 Functions***************************
void NPME_PotGenFunc_Pair_V1 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge1, const double *coord1, const double *q1,
  const size_t nCharge2, const double *coord2, const double *q2,
  double *V1, double *V2, int vecOption, bool zeroArray = 1);
//nCharge <= NPME_Pot_MaxChgBlock_V1
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[nCharge1][4]
//        V2[nCharge2][4]

void NPME_PotGenFunc_LargePair_V1 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge1, const double *coord1, const double *charge1,
  const size_t nCharge2, const double *coord2, const double *charge2,
  double *V1, double *V2, int vecOption, 
  size_t blockSize = NPME_Pot_MaxChgBlock_V1, bool zeroArray = 1);
//same as NPME_PotGenFunc_Pair_V1, but no restriction on nCharge
//blockSize <= NPME_Pot_MaxChgBlock_V1

void NPME_PotGenFunc_Self_V1 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge, const double *coord, const double *q, 
  double *V, int vecOption, bool zeroArray = 1);
//nCharge <= NPME_Pot_MaxChgBlock_V1
//input:  coord[nCharge*3], q[nCharge]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V[nCharge][4]

void NPME_PotGenFunc_LargeSelf_V1 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nCharge, const double *coord, const double *charge, 
  double *V, int vecOption, size_t blockSize = NPME_Pot_MaxChgBlock_V1, 
  bool zeroArray = 1);
//same as NPME_PotGenFunc_Self_V1, but no restriction on nCharge
//blockSize <= NPME_Pot_MaxChgBlock_V1

void NPME_PotGenFunc_SrcFld_V1 (const NPME_Library::NPME_KfuncReal& func,
  const size_t nChargeF, const double *coordF, double *VF, 
  const size_t nChargeS, const double *coordS, const double *qS,
  int vecOption, bool zeroArray = 1);
//input:  qS[nChargeS]        = source charge
//        coordS[nChargeS*3]  = source coord
//        coordF[nChargeF*3]  = field  coord
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: VF[4][nChargeF]      = potential at field coord from source charges
//        VF[4][nChargeF] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])



//*********************Complex Low Level V1 Functions***************************
void NPME_PotGenFunc_Pair_V1 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, int vecOption, bool zeroArray = 1);
//nCharge <= NPME_Pot_MaxChgBlock_V1
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[nCharge1][4]
//        V2[nCharge2][4]

void NPME_PotGenFunc_LargePair_V1 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge1, const double *coord1, const _Complex double *charge1,
  const size_t nCharge2, const double *coord2, const _Complex double *charge2,
  _Complex double *V1, _Complex double *V2, int vecOption, 
  size_t blockSize = NPME_Pot_MaxChgBlock_V1, bool zeroArray = 1);
//same as NPME_PotGenFunc_Pair_V1, but no restriction on nCharge
//blockSize <= NPME_Pot_MaxChgBlock_V1


void NPME_PotGenFunc_Self_V1 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge, const double *coord, const _Complex double *q, 
  _Complex double *V, int vecOption, bool zeroArray = 1);
//nCharge <= NPME_Pot_MaxChgBlock_V1
//input:  coord[nCharge*3], q[nCharge]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V[nCharge][4]

void NPME_PotGenFunc_LargeSelf_V1 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nCharge, const double *coord, const _Complex double *charge, 
  _Complex double *V, int vecOption, 
  size_t blockSize = NPME_Pot_MaxChgBlock_V1, bool zeroArray = 1);
//same as NPME_PotGenFunc_Self_V1, but no restriction on nCharge
//blockSize <= NPME_Pot_MaxChgBlock_V1


void NPME_PotGenFunc_SrcFld_V1 (const NPME_Library::NPME_KfuncComplex& func,
  const size_t nChargeF, const double *coordF, _Complex double *VF, 
  const size_t nChargeS, const double *coordS, const _Complex double *qS,
  int vecOption, bool zeroArray = 1);
//input:  qS[nChargeS]        = source charge
//        coordS[nChargeS*3]  = source coord
//        coordF[nChargeF*3]  = field  coord
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: VF[4][nChargeF]      = potential at field coord from source charges
//        VF[4][nChargeF] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])








}//end namespace NPME_Library


#endif // NPME_POTENTIAL_GEN_FUNC_H







