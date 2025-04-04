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

#ifndef NPME_REC_SUM_Q_H
#define NPME_REC_SUM_Q_H


#include "NPME_Bspline.h"

namespace NPME_Library
{
//******************************************************************************
//***************************Main Functions*************************************
//******************************************************************************

void NPME_RecSumQ_CalcQ1 (_Complex double *Q, 
  const size_t nCharge, const double *charge, const double *coord, 
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int n1, const long int n2, const long int n3,
  const long int BnOrder, const size_t nRecBlockTot, 
  const long int *RecBlock2BlockIndex, const size_t *nRecBlock2Charge, 
  const size_t **RecBlock2Charge, const double R0[3], 
  const int nProc, bool PRINT, std::ostream& os);
//input:  charge[nCharge], coord[3*nCharge]
//output: Q[N1*N2*N3/8] (in block form)

void NPME_RecSumQ_CalcQ1 (_Complex double *Q, 
  const size_t nCharge, const _Complex double *charge, const double *coord, 
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int n1, const long int n2, const long int n3,
  const long int BnOrder, const size_t nRecBlockTot, 
  const long int *RecBlock2BlockIndex, const size_t *nRecBlock2Charge, 
  const size_t **RecBlock2Charge, const double R0[3], 
  const int nProc, bool PRINT, std::ostream& os);
//input:  charge[nCharge], coord[3*nCharge], Qtmp[n1*N2*N3/4];
//output: Q[N1*N2*N3/8] (in block form)



void NPME_RecSumQ_Calc_C1 (_Complex double *C1, const long int N1);
//C1[N1/2] = cexp(-2*Pi*I/N1*m1)  for -N1/4 <= m1 <= N1/4 - 1


void NPME_RecSumQ_CalcQfft (_Complex double *Qf, 
  const _Complex double *Qr,
  const long int N1, const long int N2, const long int N3, 
  const long int a1, const long int a2, const long int a3,
  const _Complex double *C1, 
  const _Complex double *C2, 
  const _Complex double *C3,
  const int nProc, bool PRINT, std::ostream& os);
//input:  Qr[N1*N2*N3/8] = Q on real space grid for 
//        -N1/4 <= m1 <= N1/4 - 1
//        -N2/4 <= m2 <= N2/4 - 1
//        -N3/4 <= m3 <= N3/4 - 1
//        a1, a2, a3 = 0, 1 for even/odd contributions to the Qfft
//        leading to Qfft (2*m1+a1, 2*m2+a2, 2*m3+a3)
//        C1[N1/2] = w1^(-m1) (-N1/4 <= m1 <= N1/4 - 1)
//        C2[N2/2] = w2^(-m2) (-N2/4 <= m2 <= N2/4 - 1)
//        C3[N3/2] = w3^(-m3) (-N3/4 <= m3 <= N3/4 - 1)
//output: Qf[N1*N2*N3/8] = even/odd components of Fourier transform of Qr



//******************************************************************************
//***************************Support Functions**********************************
//******************************************************************************

void NPME_RecSumQ_CalcQ1_Basic (_Complex double *Q, const size_t nCharge, 
  const _Complex double *charge, const double *coord, 
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int BnOrder, const double R0[3]);

void NPME_RecSumQ_CalcBx (int& nBx, long int *p1Vec, long int *c1Vec, 
  double *Bx, const double wX, const long int BnOrder, const long int N1,
  const long int n1, const long int b1);
//input: wX, BnOrder, N1, n1 = blockSize, b1 = blockIndex
//output: nBx <= BnOrder = # of x component contributions to Q due to block b1
//        p1Vec[nBx] = DFT indexes          for X inside block b1 only
//        c1Vec[nBx] = local block indexes  for X inside block b1 only
//        Bx[nBx]    = B-spline values      for X inside block b1 only



void NPME_RecSumQ_CreateBlock (size_t& nNonZeroBlock, 
  std::vector<long int>& NonZeroBlock2BlockIndex_XYZ, 
  std::vector<size_t>& nNonZeroBlock2Charge, 
  std::vector<size_t*>& NonZeroBlock2Charge, 
  std::vector<size_t>&  NonZeroBlock2Charge1D, 
  const size_t nCharge, const double *coord, const long int BnOrder,
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int n1, const long int n2, const long int n3,
  const double R0[3], bool PRINT, std::ostream& os);
//input:  nCharge, coord[3*nCharge], BnOrder = Bspline order
//        N1, N2, N3 = DFT Sizes (each must be multiples of 4)
//        L1, L2, L3 = DFT box size
//        n1, n2, n3 = block sizes.
//        N1/2 must be a multiple of n1
//        N2/2 must be a multiple of n2
//        N3/2 must be a multiple of n3
//        R0[3] is the box center default = (0, 0, 0)
//output: nNonZeroBlock = number of non-zero blocks
//        NonZeroBlock2BlockIndex_XYZ[3*nNonZeroBlock]  = b1, b2, b3 indexes of
//                                                       non-zero blocks
//        nNonZeroBlock2Charge[nNonZeroBlock]           = number of charges
//        NonZeroBlock2Charge[nNonZeroBlock][]          = charge indexes
//        M1 = N1/2/n1
//        M2 = N2/2/n2
//        M3 = N3/2/n3


void NPME_TransformBlock_Q_2_Block (_Complex double *Q, _Complex double *Qtmp,
  const long int N1, const long int N2, const long int N3, 
  const long int n1, const long int n2, const long int n3,
  const int nProc);
//input: N1, N2, N3 and Q[N1*N2*N3] (original form)
//       n1 = block size with N1/n1 = integer
//       n2 = block size with N2/n2 = integer
//       n3 = block size with N3/n3 = integer
//output: Q[N1*N2*N3] in block form
//temporary array of size Qtmp(n1*N2*N3)

void NPME_TransformBlock_Block_2_Q (_Complex double *Q, _Complex double *Qtmp,
  const long int N1, const long int N2, const long int N3, 
  const long int n1, const long int n2, const long int n3,
  int nProc);
//input: N1, N2, N3 and Q[N1*N2*N3] (block form)
//       n1 = block size with N1/n1 = integer
//       n2 = block size with N2/n2 = integer
//       n3 = block size with N3/n3 = integer
//output: Q[N1*N2*N3] in original form
//temporary array of size Qtmp(n1*N2*N3)

//******************************************************************************
//******************************************************************************
//******************************Support Functions*******************************
//******************************************************************************
//******************************************************************************

void NPME_RecSumQ_FindNonZeroBlock (size_t& nNonZeroBlock, 
  std::vector<long int>& NonZeroBlock2BlockIndex_XYZ, 
  std::vector<size_t>& nNonZeroBlock2Charge, 
  std::vector<size_t*>& NonZeroBlock2Charge, 
  std::vector<size_t>&  NonZeroBlock2Charge1D, 
  const long int N1, const long int N2, const long int N3, 
  const long int n1, const long int n2, const long int n3, 
  const std::vector< std::vector <size_t> >& Block2Charge,
  bool PRINT, std::ostream& os);

void NPME_RecSumQ_FindChargeInBlock (const size_t nCharge, 
  const double *coord, const long int BnOrder,
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int n1, const long int n2, const long int n3,
  std::vector< std::vector <long int> >& Charge2Block,
  std::vector< std::vector <size_t> >& Charge2BlockInd,
  std::vector< std::vector <size_t> >& Block2Charge,
  const double R0[3]);
//input:  nCharge, coord[3*nCharge], BnOrder = Bspline order
//        N1, N2, N3 = DFT Sizes (each must be multiples of 4)
//        L1, L2, L3 = DFT box size
//        n1, n2, n3 = block sizes.
//        N1/2 must be a multiple of n1
//        N2/2 must be a multiple of n2
//        N3/2 must be a multiple of n3
//        R0[3] is the box center default = (0, 0, 0)
//output: Charge2Block[nCharge][nBlockPerCharge[]]    = charge to block indexes
//        Charge2BlockInd[nCharge][nBlockPerCharge[]] = local index of the 
//                                                      charge in the block
//        Block2Charge[nBlock][nChargePerBlock[]]     = block to charge indexes
//        nBlock = M1*M2*M3
//        M1 = N1/2/n1
//        M2 = N2/2/n2
//        M3 = N3/2/n3


void NPME_FindBlockPerChg1D (size_t& nBlockPerChg1, 
  long int BlockPerChg1[NPME_Bspline_MaxOrder], 
  const long int N1,      const long int n1, 
  const long int mStart1, const long int BnOrder);
//input:  N1 = DFT array size
//        n1 = Qpme block size
//        mStart1 <= m1 <= mStart1 + BnOrder - 1
//output: BlockPerChg1[nBlockPerChg1] = block indexes in x direction
//                                      associrated with charge
}//end namespace NPME_Library


#endif // NPME_REC_SUM_Q_H



