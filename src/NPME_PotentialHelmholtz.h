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

#ifndef NPME_POTENTIAL_HELMHOLTZ_H
#define NPME_POTENTIAL_HELMHOLTZ_H


#include "NPME_Constant.h"
#include "NPME_PartitionBox.h"
#include "NPME_PartitionEmbeddedBox.h"

namespace NPME_Library
{
void NPME_PotHelmholtz_MacroSelf_V1 (const _Complex double k0,
  const size_t nCharge, const double *coord, const _Complex double *Q1, 
  _Complex double *V1, const int nProc, const int vecOption, 
  const size_t blockSize = NPME_Pot_MaxChgBlock_V1);

void NPME_PotHelmholtz_SR_DM_DirectSum_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir,
  const size_t nCharge, const double *coord, const _Complex double *charge, 
  _Complex double *V, const int nProc, const int vecOption,
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster, 
  const size_t blockSize = NPME_Pot_MaxChgBlock_V1);


//nCharge <= NPME_Pot_MaxChgBlock for the following low level functions

void NPME_PotHelmholtz_Pair_V1 (const _Complex double k0,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, int vecOption, bool zeroArray = 1);
//nCharge <= NPME_Pot_MaxChgBlock_V1
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])




void NPME_PotHelmholtz_Self_V1 (const _Complex double k0,
  const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, int vecOption, 
  bool zeroArray = 1);
//nCharge <= NPME_Pot_MaxChgBlock_V1
//input:  coord[nCharge*3], q[nCharge1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])




void NPME_PotHelmholtz_SR_DM_Pair_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir,
  const size_t nCharge1, const double *coord1, const _Complex double *q1,
  const size_t nCharge2, const double *coord2, const _Complex double *q2,
  _Complex double *V1, _Complex double *V2, int vecOption, bool zeroArray = 1);
//nCharge <= NPME_Pot_MaxChgBlock_V1
//input:  coord1[nCharge1*3], q1[nCharge1]
//        coord2[nCharge2*3], q2[nCharge2]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//        a[Nder+1], b[Nder+1]
//output: V1[4*nCharge1]
//        V2[4*nCharge2]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])

void NPME_PotHelmholtz_SR_DM_Self_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir, const size_t nCharge, const double *coord, 
  const _Complex double *q, _Complex double *V, int vecOption, 
  bool zeroArray = 1);
//nCharge <= NPME_Pot_MaxChgBlock_V1
//input:  coord[nCharge*3], q[nCharge1]
//        vecOption = 0, 1, 2 for Non-Vector, AVX, AVX-512
//        a[Nder+1], b[Nder+1]
//output: V[4*nCharge]
//        V[4*nCharge] = (V0[nCharge], VX[nCharge], VY[nCharge], VZ[nCharge])

void NPME_PotHelmholtz_SR_DM_LargePair_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir,
  const size_t nCharge1, const double *coord1, const _Complex double *charge1,
  const size_t nCharge2, const double *coord2, const _Complex double *charge2,
  _Complex double *V1, _Complex double *V2, int vecOption, 
  size_t blockSize = NPME_Pot_MaxChgBlock_V1, bool zeroArray = 1);
//same as NPME_PotHelmholtz_SR_DM_Pair_V1, but no restriction on nCharge
//blockSize <= NPME_Pot_MaxChgBlock_V1

void NPME_PotHelmholtz_SR_DM_LargeSelf_V1 (const _Complex double k0,
  const int Nder, const _Complex double *a, const _Complex double *b, 
  const double Rdir, const size_t nCharge, const double *coord, 
  const _Complex double *charge, _Complex double *V, int vecOption, 
  size_t blockSize = NPME_Pot_MaxChgBlock_V1, bool zeroArray = 1);
//same as NPME_PotHelmholtz_SR_DM_Self_V1, but no restriction on nCharge
//blockSize <= NPME_Pot_MaxChgBlock_V1



//NPME_PotHelmholtz_MacroSelf_V1 is not significantly faster than the
//generic function NPME_PotGenFunc_MacroSelf_V1 with a Helmholtz kernel

//NPME_PotHelmholtz_MacroSelf_V1     
  //nCharge = 100000, nProc = 32, vecOption = 1 
  //blockSize =   16 time = 1.27e+01
  //blockSize =   32 time = 4.03e+00
  //blockSize =   64 time = 2.91e+00
  //blockSize =  128 time = 2.87e+00
  //blockSize =  256 time = 3.11e+00

//NPME_PotHelmholtz_MacroSelf_V1     
  //nCharge = 100000, nProc = 32, blockSize =  200
  //time = 1.02e+01 vecOption = 0
  //time = 2.57e+00 vecOption = 1
  //time = 2.39e+00 vecOption = 2


//NPME_PotGenFunc_MacroSelf_V1 (Helmholtz)
  //nCharge = 100000, nProc = 32, blockSize =  200
  //time = 1.11e+01 vecOption = 0
  //time = 2.48e+00 vecOption = 1
  //time = 2.66e+00 vecOption = 2

//nCharge = 200 200
//NPME_PotHelmholtz_Pair_V1      time = 2.09e-03 vecOption = 0
//NPME_PotHelmholtz_Pair_V1      time = 7.15e-04 vecOption = 1
//NPME_PotHelmholtz_Pair_V1      time = 4.72e-04 vecOption = 2
//NPME_PotGenFunc_Pair_V1 (Helmholtz) time = 2.17e-03 vecOption = 0
//NPME_PotGenFunc_Pair_V1 (Helmholtz) time = 4.82e-04 vecOption = 1
//NPME_PotGenFunc_Pair_V1 (Helmholtz) time = 5.48e-04 vecOption = 2

//NPME_PotHelmholtz_Self_V1      time = 1.09e-03 vecOption = 0
//NPME_PotHelmholtz_Self_V1      time = 3.37e-04 vecOption = 1
//NPME_PotHelmholtz_Self_V1      time = 3.04e-04 vecOption = 2
//NPME_PotGenFunc_Self_V1 (Helmholtz) time = 1.20e-03 vecOption = 0
//NPME_PotGenFunc_Self_V1 (Helmholtz) time = 3.16e-04 vecOption = 1
//NPME_PotGenFunc_Self_V1 (Helmholtz) time = 3.36e-04 vecOption = 2




}//end namespace NPME_Library


#endif // NPME_POTENTIAL_HELMHOLTZ_H







