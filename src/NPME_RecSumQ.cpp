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

#include <cstdlib> 
#include <cstring> 
#include <cmath>
#include <cstdio>

#include <iostream> 
#include <vector>
#include <algorithm>



#include "NPME_Constant.h"
#include "NPME_SupportFunctions.h"
#include "NPME_RecSumQ.h"
#include "NPME_ExtLibrary.h"
#include "NPME_Bspline.h"
#include "NPME_RecSumSupportFunctions.h"

namespace
{
void NPME_CopyDoubleArray (const long int N, const int nProc, 
  double *A, const double *B, const long int blockSize)
{
  const long int r = N%blockSize;
  const long int n = (N-r)/blockSize;

  long int n1;
  #pragma omp parallel shared(A, B) private(n1) num_threads(nProc) 
  {
    #pragma omp for schedule(static) nowait
    for (n1 = 0; n1 < n; n1++)
      memcpy(&A[n1*blockSize], &B[n1*blockSize], blockSize*sizeof(double));
    #pragma omp single
    {
      memcpy(&A[n*blockSize], &B[n*blockSize], r*sizeof(double));
    }
  }
}
}//end empty namespace

namespace NPME_Library
{

void NPME_RecSumQ_CalcBx (int& nBx, long int *p1Vec, long int *c1Vec, 
  double *Bx, const double wX, const long int BnOrder, const long int N1,
  const long int n1, const long int b1)
//input: wX, BnOrder, N1, n1 = blockSize, b1 = blockIndex
//output: nBx <= BnOrder = # of x component contributions to Q due to block b1
//        p1Vec[nBx] = DFT indexes          for X inside block b1 only
//        c1Vec[nBx] = local block indexes  for X inside block b1 only
//        Bx[nBx]    = B-spline values      for X inside block b1 only
{
  const long int mStart1 = NPME_GetQpme_DFT_StartIndex (wX);
  const long int p1Start = n1*b1;
  const long int p1End   = n1*(b1+1);
  const int BnOrderInt   = (int) BnOrder;

  nBx = 0;
  for (long int m = mStart1; m < mStart1 + BnOrder; m++)
  {
    long int p = NPME_DFT2Array_Index (m, N1/2);
    if ( (p >= p1Start) && (p < p1End) )
    {
      p1Vec[nBx] = p;
      c1Vec[nBx] = p%n1;
      Bx[nBx]    = NPME_Bspline_B (BnOrderInt, wX + m);
      nBx++;
    }
  }
}

//******************************************************************************
//******************************************************************************
//******************************************************************************
//**************** Basic Slow Q1 code = single charge input*********************
//******************************************************************************
//******************************************************************************
//******************************************************************************
void NPME_RecSumQ_CalcQ1_Basic (_Complex double *Q, const size_t nCharge, 
  const _Complex double *charge, const double *coord, 
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int BnOrder, const double R0[3])
{
  const int BnOrderInt   = (int) BnOrder;

  memset(Q, 0, N1*N2*N3/8*sizeof(_Complex double));

  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r = &coord[3*i];
    const double wX = N1/L1*(r[0] - R0[0]);
    const double wY = N2/L2*(r[1] - R0[1]);
    const double wZ = N3/L3*(r[2] - R0[2]);

    const long int mStart1  = NPME_GetQpme_DFT_StartIndex (wX);
    const long int mStart2  = NPME_GetQpme_DFT_StartIndex (wY);
    const long int mStart3  = NPME_GetQpme_DFT_StartIndex (wZ);

    if (!NPME_CheckQpme_DFT_StartIndex (mStart1, N1, BnOrder))
    {
      std::cout << "Error in NPME_RecSumQ_CalcQ1_Basic for x\n";
      std::cout << "  NPME_CheckQpme_DFT_StartIndex failed for mStart1\n";
      exit(0);
    }
    if (!NPME_CheckQpme_DFT_StartIndex (mStart2, N2, BnOrder))
    {
      std::cout << "Error in NPME_RecSumQ_CalcQ1_Basic for y\n";
      std::cout << "  NPME_CheckQpme_DFT_StartIndex failed for mStart2\n";
      exit(0);
    }
    if (!NPME_CheckQpme_DFT_StartIndex (mStart3, N3, BnOrder))
    {
      std::cout << "Error in NPME_RecSumQ_CalcQ1_Basic for z\n";
      std::cout << "  NPME_CheckQpme_DFT_StartIndex failed for mStart3\n";
      exit(0);
    }

    double B1[NPME_Bspline_MaxOrder];
    double B2[NPME_Bspline_MaxOrder];
    double B3[NPME_Bspline_MaxOrder];


    for (long int n1 = 0; n1 < BnOrder; n1++)
    {
      B1[n1] = NPME_Bspline_B (BnOrderInt, wX + mStart1 + n1);
      B2[n1] = NPME_Bspline_B (BnOrderInt, wY + mStart2 + n1);
      B3[n1] = NPME_Bspline_B (BnOrderInt, wZ + mStart3 + n1);
    }

    for (long int n1 = 0; n1 < BnOrder; n1++)
    {
      const long int m1         = mStart1 + n1;
      const long int p1         = NPME_DFT2Array_Index (m1, N1/2);
      const _Complex double C1  = conj(charge[i])*B1[n1];

      for (long int n2 = 0; n2 < BnOrder; n2++)
      {
        const long int m2         = mStart2 + n2;
        const long int p2         = NPME_DFT2Array_Index (m2, N2/2);
        const _Complex double C2  = C1*B2[n2];

        for (long int n3 = 0; n3 < BnOrder; n3++)
        {
          const long int m3 = mStart3 + n3;
          const long int p3 = NPME_DFT2Array_Index (m3, N3/2);

          long int Qindex   = NPME_ind3D (p1, p2, p3, N2/2, N3/2);
          Q[Qindex]        += C2*B3[n3];
        }
      }
    }
  }
}

//******************************************************************************
//******************************************************************************
//******************************************************************************
//*********************Q1 code = single charge input****************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
void NPME_RecSumQ_CalcQ1_UpdateRow (_Complex double *Q,
  const double C, const int N, const double *B3)
{
  for (int i = 0; i < N; i++)
    Q[i] += C*B3[i];
}

void NPME_RecSumQ_CalcQ1_Block (_Complex double *Q, 
  const double   L1, const double   L2, const double   L3, 
  const long int N1, const long int N2, const long int N3, 
  const long int n1, const long int n2, const long int n3, 
  const long int b1, const long int b2, const long int b3, 
  const long int BnOrder, const size_t nCharge, const double *charge, 
  const double *coord, const double R0[3])
{
  for (size_t n = 0; n < nCharge; n++)
  {
    const double *r   = &coord[3*n];
    const double wX   = N1/L1*(r[0] - R0[0]);
    const double wY   = N2/L2*(r[1] - R0[1]);
    const double wZ   = N3/L3*(r[2] - R0[2]);
    const double qLoc = charge[n];

    int nB1, nB2, nB3;
    long int p1Vec[NPME_Bspline_MaxOrder];
    long int p2Vec[NPME_Bspline_MaxOrder];
    long int p3Vec[NPME_Bspline_MaxOrder];
    long int c1Vec[NPME_Bspline_MaxOrder];
    long int c2Vec[NPME_Bspline_MaxOrder];
    long int c3Vec[NPME_Bspline_MaxOrder];
    double B1[NPME_Bspline_MaxOrder];
    double B2[NPME_Bspline_MaxOrder];
    double B3[NPME_Bspline_MaxOrder];


    NPME_RecSumQ_CalcBx (nB1, p1Vec, c1Vec, B1, wX, BnOrder, N1, n1, b1);
    NPME_RecSumQ_CalcBx (nB2, p2Vec, c2Vec, B2, wY, BnOrder, N2, n2, b2);
    NPME_RecSumQ_CalcBx (nB3, p3Vec, c3Vec, B3, wZ, BnOrder, N3, n3, b3);

    if (nB3 > 0)
    {
      for (int i1 = 0; i1 < nB1; i1++)
      {
        const double C1       = qLoc*B1[i1];
        const long int iTerm1 = c1Vec[i1]*n2*n3;
        
        for (int i2 = 0; i2 < nB2; i2++)
        {
          const double C2       = C1*B2[i2];
          const long int iTerm2 = iTerm1 + c2Vec[i2]*n3 + c3Vec[0];
          NPME_RecSumQ_CalcQ1_UpdateRow (&Q[iTerm2], C2, nB3, B3);
        }
      }
    }
  }
}

void NPME_RecSumQ_CalcQ1 (_Complex double *Q, 
  const size_t nCharge, const double *charge, const double *coord, 
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int n1, const long int n2, const long int n3,
  const long int BnOrder, const size_t nRecBlockTot, 
  const long int *RecBlock2BlockIndex, const size_t *nRecBlock2Charge, 
  const size_t **RecBlock2Charge, const double R0[3], 
  const int nProc, bool PRINT, std::ostream& os)
//input:  charge[nCharge], coord[3*nCharge]
//output: Q[N1*N2*N3/8] (in block form)
{
  double time0, time;
  char str[2000];

//const long int M1 = N1/2/n1;
  const long int M2 = N2/2/n2;
  const long int M3 = N3/2/n3;


  const long int blockSize = n1*n2*n3;

  std::vector<double> coordLoc;
  std::vector<double> chargeLoc;

  if (PRINT)
    os << "\n      NPME_RecSumQ_CalcQ\n";

  time0 = NPME_GetTime ();
  NPME_ZeroArray (N1*N2*N3/8, nProc, Q, blockSize);
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, "      time to zero Q = %le\n", time);
    os << str;
    os.flush();
  }

  size_t n;
  time0 = NPME_GetTime ();
  #pragma omp parallel shared(Q, charge, coord, RecBlock2BlockIndex, nRecBlock2Charge, RecBlock2Charge) private(n, coordLoc, chargeLoc) num_threads(nProc)
  {
    #pragma omp for schedule(dynamic)
    for (n = 0; n < nRecBlockTot; n++)
    {
      coordLoc.resize(3*nRecBlock2Charge[n]);
      chargeLoc.resize(nRecBlock2Charge[n]);

      const size_t *index = RecBlock2Charge[n];

      for (size_t i = 0; i < nRecBlock2Charge[n]; i++)
      {
        const size_t chargeNum = index[i];
        coordLoc[3*i+0] = coord[3*chargeNum  ];
        coordLoc[3*i+1] = coord[3*chargeNum+1];
        coordLoc[3*i+2] = coord[3*chargeNum+2];
        chargeLoc[i]    = charge[chargeNum];
      }

      const long int b1     = RecBlock2BlockIndex[3*n  ];
      const long int b2     = RecBlock2BlockIndex[3*n+1];
      const long int b3     = RecBlock2BlockIndex[3*n+2];
      const long int nIndex = NPME_ind3D (b1, b2, b3, M2, M3);

      NPME_RecSumQ_CalcQ1_Block (&Q[nIndex*blockSize], 
        L1, L2, L3, 
        N1, N2, N3, 
        n1, n2, n3, 
        b1, b2, b3, 
        BnOrder, nRecBlock2Charge[n], &chargeLoc[0], &coordLoc[0], 
        R0);
    }
  }
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, "      time Q calc    = %le\n", time);
    os << str;
    os.flush();
  }
}

void NPME_RecSumQ_CalcQ1_UpdateRow (_Complex double *Q,
  const _Complex double C, const int N, const double *B3)
{
  for (int i = 0; i < N; i++)
    Q[i] += C*B3[i];
}

void NPME_RecSumQ_CalcQ1_Block (_Complex double *Q, 
  const double   L1, const double   L2, const double   L3, 
  const long int N1, const long int N2, const long int N3, 
  const long int n1, const long int n2, const long int n3, 
  const long int b1, const long int b2, const long int b3, 
  const long int BnOrder, const size_t nCharge, const _Complex double *charge, 
  const double *coord, const double R0[3])
{
  for (size_t n = 0; n < nCharge; n++)
  {
    const double *r             = &coord[3*n];
    const double wX             = N1/L1*(r[0] - R0[0]);
    const double wY             = N2/L2*(r[1] - R0[1]);
    const double wZ             = N3/L3*(r[2] - R0[2]);
    const _Complex double qLoc  = conj(charge[n]);

    int nB1, nB2, nB3;
    long int p1Vec[NPME_Bspline_MaxOrder];
    long int p2Vec[NPME_Bspline_MaxOrder];
    long int p3Vec[NPME_Bspline_MaxOrder];
    long int c1Vec[NPME_Bspline_MaxOrder];
    long int c2Vec[NPME_Bspline_MaxOrder];
    long int c3Vec[NPME_Bspline_MaxOrder];
    double B1[NPME_Bspline_MaxOrder];
    double B2[NPME_Bspline_MaxOrder];
    double B3[NPME_Bspline_MaxOrder];


    NPME_RecSumQ_CalcBx (nB1, p1Vec, c1Vec, B1, wX, BnOrder, N1, n1, b1);
    NPME_RecSumQ_CalcBx (nB2, p2Vec, c2Vec, B2, wY, BnOrder, N2, n2, b2);
    NPME_RecSumQ_CalcBx (nB3, p3Vec, c3Vec, B3, wZ, BnOrder, N3, n3, b3);

    if (nB3 > 0)
    {
      for (int i1 = 0; i1 < nB1; i1++)
      {
        const _Complex double C1      = qLoc*B1[i1];
        const long int        iTerm1  = c1Vec[i1]*n2*n3;
        
        for (int i2 = 0; i2 < nB2; i2++)
        {
          const _Complex double C2      = C1*B2[i2];
          const long int        iTerm2  = iTerm1 + c2Vec[i2]*n3 + c3Vec[0];
          NPME_RecSumQ_CalcQ1_UpdateRow (&Q[iTerm2], C2, nB3, B3);
        }
      }
    }
  }
}

void NPME_RecSumQ_CalcQ1 (_Complex double *Q, 
  const size_t nCharge, const _Complex double *charge, const double *coord, 
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int n1, const long int n2, const long int n3,
  const long int BnOrder, const size_t nRecBlockTot, 
  const long int *RecBlock2BlockIndex, const size_t *nRecBlock2Charge, 
  const size_t **RecBlock2Charge, const double R0[3], 
  const int nProc, bool PRINT, std::ostream& os)
//input:  charge[nCharge], coord[3*nCharge]
//output: Q[N1*N2*N3/8] (in block form)
{
  double time0, time;
  char str[2000];

//const long int M1 = N1/2/n1;
  const long int M2 = N2/2/n2;
  const long int M3 = N3/2/n3;
  

  const long int blockSize = n1*n2*n3;

  std::vector<double> coordLoc;
  std::vector<_Complex double> chargeLoc;

  if (PRINT)
    os << "\n      NPME_RecSumQ_CalcQ\n";

  time0 = NPME_GetTime ();
  NPME_ZeroArray (N1*N2*N3/8, nProc, Q, blockSize);
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, "      time to zero Q = %le\n", time);
    os << str;
    os.flush();
  }

  size_t n;
  time0 = NPME_GetTime ();
  #pragma omp parallel shared(Q, charge, coord, RecBlock2BlockIndex, nRecBlock2Charge, RecBlock2Charge) private(n, coordLoc, chargeLoc) num_threads(nProc)
  {
    #pragma omp for schedule(dynamic)
    for (n = 0; n < nRecBlockTot; n++)
    {
      coordLoc.resize(3*nRecBlock2Charge[n]);
      chargeLoc.resize(nRecBlock2Charge[n]);

      const size_t *index = RecBlock2Charge[n];

      for (size_t i = 0; i < nRecBlock2Charge[n]; i++)
      {
        const size_t chargeNum = index[i];
        coordLoc[3*i+0] = coord[3*chargeNum  ];
        coordLoc[3*i+1] = coord[3*chargeNum+1];
        coordLoc[3*i+2] = coord[3*chargeNum+2];
        chargeLoc[i]    = charge[chargeNum];
      }

      const long int b1     = RecBlock2BlockIndex[3*n  ];
      const long int b2     = RecBlock2BlockIndex[3*n+1];
      const long int b3     = RecBlock2BlockIndex[3*n+2];
      const long int nIndex = NPME_ind3D (b1, b2, b3, M2, M3);

      NPME_RecSumQ_CalcQ1_Block (&Q[nIndex*blockSize], 
        L1, L2, L3, 
        N1, N2, N3, 
        n1, n2, n3, 
        b1, b2, b3, 
        BnOrder, nRecBlock2Charge[n], &chargeLoc[0], &coordLoc[0], 
        R0);
    }
  }
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, "      time Q calc    = %le\n", time);
    os << str;
    os.flush();
  }
}


void NPME_RecSumQ_Calc_C1 (_Complex double *C1, const long int N1)
//C1[N1/2] = cexp(-2*NPME_Pi*I/N1*m1)  for -N1/4 <= m1 <= N1/4 - 1
{
  for (long int n1 = 0; n1 < N1/2; n1++)
  {
    long int m1 = NPME_Array2DFT_Index (n1, N1/2);
    C1[n1]      = cexp(-2.0*NPME_Pi*I/N1*m1);
  }
}

void NPME_RecSumQ_CalcQfft (_Complex double *Qf, 
  const _Complex double *Qr,
  const long int N1, const long int N2, const long int N3, 
  const long int a1, const long int a2, const long int a3,
  const _Complex double *C1, 
  const _Complex double *C2, 
  const _Complex double *C3,
  const int nProc, bool PRINT, std::ostream& os)
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
{
  const long int M1 = N1/2;
  const long int M2 = N2/2;
  const long int M3 = N3/2;

  long int k;
  #pragma omp parallel for schedule(dynamic) shared(Qf, Qr, C1, C2, C3) private(k) num_threads(nProc) default(none)
  for (k = 0; k < M1*M2; k++)
  {
    long int n1, n2;
    NPME_ind2D_2_n1_n2 (k, M2, n1, n2);

    _Complex double C  = 1.0;
    if (a1 == 1)    C  = C1[n1];
    if (a2 == 1)    C *= C2[n2];

    long int count = n1*M2*M3 + n2*M3;
    if (a3 == 0)
    {
      for (long int n3 = 0; n3 < M3; n3++)
      {
        Qf[count] = Qr[count]*C;
        count++;
      }
    }
    else
    {
      for (long int n3 = 0; n3 < M3; n3++)
      {
        Qf[count] = Qr[count]*C*C3[n3];
        count++;
      }
    }
  }

  mkl_set_num_threads (nProc);
  NPME_3D_FFT_NoNorm (Qf, N1/2, N2/2, N3/2);
  mkl_set_num_threads (1);
}




void NPME_RecSumQ_CreateBlock (size_t& nNonZeroBlock, 
  std::vector<long int>& NonZeroBlock2BlockIndex_XYZ, 
  std::vector<size_t>& nNonZeroBlock2Charge, 
  std::vector<size_t*>& NonZeroBlock2Charge, 
  std::vector<size_t>&  NonZeroBlock2Charge1D, 
  const size_t nCharge, const double *coord, const long int BnOrder,
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int n1, const long int n2, const long int n3,
  const double R0[3], bool PRINT, std::ostream& os)
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
{

  if (!NPME_CheckFFTParm (BnOrder,
    N1, N2, N3,
    n1, n2, n3))
  {
    std::cout << "Error in NPME_RecSumQ_CreateBlock.  NPME_CheckFFTParm failed\n";
    exit(0);
  }

  std::vector< std::vector <long int> > Charge2Block;
  std::vector< std::vector <size_t> > Charge2BlockInd;
  std::vector< std::vector <size_t> > Block2Charge;
  NPME_RecSumQ_FindChargeInBlock (nCharge, &coord[0], BnOrder,
    N1, N2, N3, L1, L2, L3, n1, n2, n3,
    Charge2Block, Charge2BlockInd, Block2Charge, R0);


  NPME_RecSumQ_FindNonZeroBlock (nNonZeroBlock, NonZeroBlock2BlockIndex_XYZ, 
    nNonZeroBlock2Charge, NonZeroBlock2Charge, NonZeroBlock2Charge1D, 
    N1, N2, N3, 
    n1, n2, n3, 
    Block2Charge, PRINT, os);
}








//******************************************************************************
//******************************************************************************
//******************************Support Functions*******************************
//******************************************************************************
//******************************************************************************

struct NPME_RSQBT_SortBlock_Data
{
  size_t nCharge;
  size_t index;
};

bool NPME_RSQBT_SortBlock_Data_Criteria (
  const NPME_RSQBT_SortBlock_Data& i1, 
  const NPME_RSQBT_SortBlock_Data& i2)
{
  if (i1.nCharge > i2.nCharge)
    return true;
  else 
    return false;
}

void NPME_RecSumQ_FindNonZeroBlock (size_t& nNonZeroBlock, 
  std::vector<long int>& NonZeroBlock2BlockIndex_XYZ, 
  std::vector<size_t>& nNonZeroBlock2Charge, 
  std::vector<size_t*>& NonZeroBlock2Charge, 
  std::vector<size_t>&  NonZeroBlock2Charge1D, 
  const long int N1, const long int N2, const long int N3, 
  const long int n1, const long int n2, const long int n3, 
  const std::vector< std::vector <size_t> >& Block2Charge,
  bool PRINT, std::ostream& os)
{
  char str[2000];
  size_t size_1D;
  const long int M1 = N1/2/n1;
  const long int M2 = N2/2/n2;
  const long int M3 = N3/2/n3;

  const long int nBlockTot = M1*M2*M3;
  if (nBlockTot != (long int) Block2Charge.size())
  {
    std::cout << "Error in NPME_RecSumQ_FindNonZeroBlock\n";
    sprintf(str, "  nBlockTot = %ld != %ld = Block2Charge.size()\n",
      nBlockTot, Block2Charge.size() );
    std::cout << str;
    exit(0);
  }

  //sort block indexes based on number of charges per block
  std::vector<NPME_RSQBT_SortBlock_Data> NPME_RSQBT_SortBlock (nBlockTot);
  for (long int n = 0; n < nBlockTot; n++)
  {
    NPME_RSQBT_SortBlock[n].nCharge = Block2Charge[n].size();
    NPME_RSQBT_SortBlock[n].index   = n;
  }
  std::sort (NPME_RSQBT_SortBlock.begin(), 
             NPME_RSQBT_SortBlock.end(), 
             NPME_RSQBT_SortBlock_Data_Criteria);





  nNonZeroBlock = 0;
  size_1D       = 0;
  for (long int n = 0; n < nBlockTot; n++)
  {
    const size_t totBlockIndex  = NPME_RSQBT_SortBlock[n].index;
    const size_t nCharge        = Block2Charge[totBlockIndex].size();
    if (nCharge > 0)
    {
      nNonZeroBlock++;
      size_1D += Block2Charge[totBlockIndex].size();
    }
  }

  NonZeroBlock2BlockIndex_XYZ.resize(3*nNonZeroBlock);
  nNonZeroBlock2Charge.resize(nNonZeroBlock);
  NonZeroBlock2Charge.resize(nNonZeroBlock);
  NonZeroBlock2Charge1D.resize(size_1D);

  nNonZeroBlock = 0;
  size_1D       = 0;
  for (long int n = 0; n < nBlockTot; n++)
  {
    const size_t totBlockIndex  = NPME_RSQBT_SortBlock[n].index;
    const size_t nCharge        = Block2Charge[totBlockIndex].size();

    if (nCharge > 0)
    {
      const size_t *chgIndex = &(Block2Charge[totBlockIndex][0]);

      long int b1, b2, b3;
      NPME_ind3D_2_n1_n2_n3 (totBlockIndex, M2, M3, b1, b2, b3);

      NonZeroBlock2BlockIndex_XYZ[3*nNonZeroBlock  ] = b1;
      NonZeroBlock2BlockIndex_XYZ[3*nNonZeroBlock+1] = b2;
      NonZeroBlock2BlockIndex_XYZ[3*nNonZeroBlock+2] = b3;

      nNonZeroBlock2Charge[nNonZeroBlock] = nCharge;
      NonZeroBlock2Charge[nNonZeroBlock]  = &NonZeroBlock2Charge1D[size_1D];

      memcpy(NonZeroBlock2Charge[nNonZeroBlock], 
        chgIndex, nCharge*sizeof(size_t));

      nNonZeroBlock++;
      size_1D += nCharge;
    }
  }
      
  if (PRINT)
  {
    os << "\n\nNPME_RecSumQ_FindNonZeroBlock\n";
    sprintf(str, "nNonZeroBlock = %lu (non-zero blocks) out of nBlockTot = %ld\n", 
      nNonZeroBlock, nBlockTot);
    os << str;
    for (size_t i = 0; i < nNonZeroBlock; i++)
    {
      sprintf(str, "  non-zero block %4lu  (original block index = %4lu) %3ld %3ld %3ld  nCharge = %lu\n", 
        i, NPME_RSQBT_SortBlock[i].index,
        NonZeroBlock2BlockIndex_XYZ[3*i  ], 
        NonZeroBlock2BlockIndex_XYZ[3*i+1], 
        NonZeroBlock2BlockIndex_XYZ[3*i+2], nNonZeroBlock2Charge[i]);
      os << str;
    }
  }
}

void NPME_RecSumQ_FindChargeInBlock (const size_t nCharge, 
  const double *coord, const long int BnOrder,
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int n1, const long int n2, const long int n3,
  std::vector< std::vector <long int> >& Charge2Block,
  std::vector< std::vector <size_t> >& Charge2BlockInd,
  std::vector< std::vector <size_t> >& Block2Charge,
  const double R0[3])
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
{
  char str[2000];

  if (BnOrder > (long int) NPME_Bspline_MaxOrder)
  {
    std::cout << "Error in NPME_RecSumQ_FindChargeInBlock.\n";
    sprintf(str, "  BnOrder = %ld > %ld NPME_Bspline_MaxOrder\n", 
      BnOrder, (long int) NPME_Bspline_MaxOrder);
    std::cout << str;
    exit(0);
  }
  if (N1%(2*n1) != 0)
  {
    std::cout << "Error in NPME_RecSumQ_FindChargeInBlock.\n";
    sprintf(str, "  N1 = %ld n1 = %ld N1 mod (2*n1) = %ld != 0\n", 
      N1, n1, N1%(2*n1));
    std::cout << str;
    exit(0);
  }
  if (N2%(2*n2) != 0)
  {
    std::cout << "Error in NPME_RecSumQ_FindChargeInBlock.\n";
    sprintf(str, "  N2 = %ld n2 = %ld N2 mod (2*n2) = %ld != 0\n", 
      N2, n2, N2%(2*n2));
    std::cout << str;
    exit(0);
  }
  if (N3%(2*n3) != 0)
  {
    std::cout << "Error in NPME_RecSumQ_FindChargeInBlock.\n";
    sprintf(str, "  N3 = %ld n3 = %ld N3 mod (2*n3) = %ld != 0\n", 
      N3, n3, N3%(2*n3));
    std::cout << str;
    exit(0);
  }

  //Qpme (real space) is stored on a N1*N2*N3/8 grid whose DFT indexes satisfy
  //      -N1/4 <= m1 <= N1/4 - 1
  //      -N2/4 <= m2 <= N2/4 - 1
  //      -N3/4 <= m3 <= N3/4 - 1
  //and array indexes (p1, p2, p3) satisfy
  //      0 <= p1 <= N1/2 - 1
  //      0 <= p2 <= N2/2 - 1
  //      0 <= p3 <= N3/2 - 1

  const long int M1 = N1/2/n1;    //M1 = # of blocks in x direction
  const long int M2 = N2/2/n2;    //M2 = # of blocks in y direction
  const long int M3 = N3/2/n3;    //M3 = # of blocks in z direction

  //consider a block defined by indexes (b1, b2, b3).  this block contains
  //array indexes (p1, p2, p3) defined by
  //  b1*n1 <= p1 < (b1+1)*n1
  //  b2*n2 <= p2 < (b2+1)*n2
  //  b3*n3 <= p3 < (b3+1)*n3



  //1) find blocks associated with each charge coordinates 
  Charge2Block.resize(nCharge);
  for (size_t chgInd = 0; chgInd < nCharge; chgInd++)
  {
    const double *r         = &coord[3*chgInd];
    const double wX         = N1/L1*(r[0] - R0[0]);
    const double wY         = N2/L2*(r[1] - R0[1]);
    const double wZ         = N3/L3*(r[2] - R0[2]);

    const long int mStart1  = NPME_GetQpme_DFT_StartIndex (wX);
    const long int mStart2  = NPME_GetQpme_DFT_StartIndex (wY);
    const long int mStart3  = NPME_GetQpme_DFT_StartIndex (wZ);

    if (!NPME_CheckQpme_DFT_StartIndex (mStart1, N1, BnOrder))
    {
      std::cout << "Error in NPME_RecSumQ_FindChargeInBlock for x\n";
      std::cout << "  NPME_CheckQpme_DFT_StartIndex failed for mStart1\n";
      exit(0);
    }
    if (!NPME_CheckQpme_DFT_StartIndex (mStart2, N2, BnOrder))
    {
      std::cout << "Error in NPME_RecSumQ_FindChargeInBlock for y\n";
      std::cout << "  NPME_CheckQpme_DFT_StartIndex failed for mStart2\n";
      exit(0);
    }
    if (!NPME_CheckQpme_DFT_StartIndex (mStart3, N3, BnOrder))
    {
      std::cout << "Error in NPME_RecSumQ_FindChargeInBlock for z\n";
      std::cout << "  NPME_CheckQpme_DFT_StartIndex failed for mStart3\n";
      exit(0);
    }

    //The DFT indexes associated with coord[3*chgInd] are defined by
    //      mStart1 <= m1 < mStart1 + BnOrder
    //      mStart2 <= m2 < mStart2 + BnOrder
    //      mStart3 <= m3 < mStart3 + BnOrder

    //By construction, (m1, m2, m3) satisfy
    //      -N1/4 <= m1 <= N1/4 - 1
    //      -N2/4 <= m2 <= N2/4 - 1
    //      -N3/4 <= m3 <= N3/4 - 1

    //As Qpme (real space) is stored on a N1*N2*N3/8 grid, note the array
    //   indexes (p1, p2, p3) are calculated from (m1, m2, m3) and satisfy
    //      0 <= p1 <= N1/2 - 1
    //      0 <= p2 <= N2/2 - 1
    //      0 <= p3 <= N3/2 - 1

    size_t nBlkPerChg1;
    size_t nBlkPerChg2;
    size_t nBlkPerChg3;
    long int BlkPerChg1[NPME_Bspline_MaxOrder];
    long int BlkPerChg2[NPME_Bspline_MaxOrder];
    long int BlkPerChg3[NPME_Bspline_MaxOrder];

    NPME_FindBlockPerChg1D (nBlkPerChg1, BlkPerChg1, N1, n1, mStart1, BnOrder);
    NPME_FindBlockPerChg1D (nBlkPerChg2, BlkPerChg2, N2, n2, mStart2, BnOrder);
    NPME_FindBlockPerChg1D (nBlkPerChg3, BlkPerChg3, N3, n3, mStart3, BnOrder);

    Charge2Block[chgInd].resize(nBlkPerChg1*nBlkPerChg2*nBlkPerChg3);
    size_t count = 0;
    for (size_t i1 = 0; i1 < nBlkPerChg1; i1++)
    for (size_t i2 = 0; i2 < nBlkPerChg2; i2++)
    for (size_t i3 = 0; i3 < nBlkPerChg3; i3++)
    {
      long int b1 = BlkPerChg1[i1];
      long int b2 = BlkPerChg2[i2];
      long int b3 = BlkPerChg3[i3];

      Charge2Block[chgInd][count] = NPME_ind3D (b1, b2, b3, M2, M3);
      count++;
    }
  }

  //find block2charge data
  Block2Charge.resize(M1*M2*M3);
  for (long int i = 0; i < M1*M2*M3; i++)
    Block2Charge[i].clear();

  Charge2BlockInd.resize(nCharge);
  for (size_t i = 0; i < nCharge; i++)
  {
    Charge2BlockInd[i].resize(Charge2Block[i].size());
    for (size_t j = 0; j < Charge2Block[i].size(); j++)
    {
      const long int blockNum         = Charge2Block[i][j];
      const size_t locChargeIndex     = Block2Charge[blockNum].size();
      Charge2BlockInd[i][j]           = locChargeIndex;

      if (blockNum >= M1*M2*M3)
      {
        std::cout << "Error in NPME_RecSumQ_FindChargeInBlock.\n";
        sprintf(str, "  blockNum = %4ld != %4ld = M1*M2*M3\n",
          blockNum, M1*M2*M3);
        std::cout << str;
        exit(0);
      }
      Block2Charge[blockNum].push_back(i);
    }
  }
}

void NPME_FindBlockPerChg1D (size_t& nBlockPerChg1, 
  long int BlockPerChg1[NPME_Bspline_MaxOrder], 
  const long int N1,      const long int n1, 
  const long int mStart1, const long int BnOrder)
//input:  N1 = DFT array size
//        n1 = Qpme block size
//        mStart1 <= m1 <= mStart1 + BnOrder - 1
//output: BlockPerChg1[nBlockPerChg1] = block indexes in x direction
//                                      associrated with charge
{
  nBlockPerChg1 = 0;
  for (long int n = 0; n < BnOrder; n++)
  {
    long int m1 = mStart1 + n;
    long int p1 = NPME_DFT2Array_Index (m1, N1/2);
    //  -N1/4 <= m1 <= N1/4 - 1
    //  0     <= p1 <= N1/2 - 1

    long int r1 = p1%n1;
    long int b1 = (p1 - r1)/n1;
    
    if (n > 0)
    {
      if (b1 != BlockPerChg1[nBlockPerChg1 - 1])
      {
        BlockPerChg1[nBlockPerChg1] = b1;
        nBlockPerChg1++;
      }
    }
    else
    {
      BlockPerChg1[nBlockPerChg1] = b1;
      nBlockPerChg1++;
    }
  }
}




void NPME_TransformBlock_Q_2_Block (_Complex double *Q, _Complex double *Qtmp,
  const long int N1, const long int N2, const long int N3, 
  const long int n1, const long int n2, const long int n3,
  const int nProc)
//input: N1, N2, N3 and Q[N1*N2*N3] (original form)
//       n1 = block size with N1/n1 = integer
//       n2 = block size with N2/n2 = integer
//       n3 = block size with N3/n3 = integer
//output: Q[N1*N2*N3] in block form
//temporary array of size Qtmp(n1*N2*N3)
{
  char str[2000];

  if (N1%n1 != 0)
  {
    std::cout << "Error in NPME_TransformBlock_Q_2_Block.\n";
    sprintf(str, "  N1 = %ld is not divisible by block size n1 = %ld\n", 
      N1, n1);
    std::cout << str;
    exit(0);
  }
  if (N2%n2 != 0)
  {
    std::cout << "Error in NPME_TransformBlock_Q_2_Block.\n";
    sprintf(str, "  N2 = %ld is not divisible by block size n2 = %ld\n", 
      N2, n2);
    std::cout << str;
    exit(0);
  }
  if (N3%n3 != 0)
  {
    std::cout << "Error in NPME_TransformBlock_Q_2_Block.\n";
    sprintf(str, "  N3 = %ld is not divisible by block size n3 = %ld\n", 
      N3, n3);
    std::cout << str;
    exit(0);
  }

  //M1, M2, M3 = #of blocks in each direction
  const long int M1 = N1/n1;
  const long int M2 = N2/n2;
  const long int M3 = N3/n3;

  const long int arrayBlockSize = N2*N3;
  for (long int b1 = 0; b1 < M1; b1++)
  {
  //memcpy(&Qtmp[0], &Q[b1*n1*N2*N3], n1*N2*N3*sizeof(_Complex double));
    NPME_CopyDoubleArray (2*n1*N2*N3, nProc, (double*) &Qtmp[0], 
      (const double*) &Q[b1*n1*N2*N3], arrayBlockSize);

    long int b23;
    #pragma omp parallel shared(Q, Qtmp) private(b23) num_threads(nProc)
    {
      #pragma omp for schedule(static)
      for (b23 = 0; b23 < M2*M3; b23++)
      {
        long int b2, b3;
        NPME_ind2D_2_n1_n2 (b23, M3, b2, b3);

        long int j              = b1*M2*M3+b23;
        _Complex double *Qblock = &Q[j*n1*n2*n3];

        for (long int c1 = 0; c1 < n1; c1++)
        for (long int c2 = 0; c2 < n2; c2++)
        {
          const long int p1       = n1*b1+c1;
          const long int p2       = n2*b2+c2;
          const long int p3Start  = n3*b3;
          const long int kStart   = NPME_ind3D (c1, c2, (long int) 0, n2, n3);

          //i = original Q index
          const long int iStart = NPME_ind3D (p1, p2, p3Start, N2, N3)  
                                    - b1*n1*N2*N3;
          memcpy(&Qblock[kStart], &Qtmp[iStart], n3*sizeof(_Complex double));
        }
      }
    }
  }
}

void NPME_TransformBlock_Block_2_Q_CopyBlock (_Complex double *Q, _Complex double *Qtmp, 
  const long int b1, const long int b2, const long int b3, 
  const long int n1, const long int n2, const long int n3,
                     const long int N2, const long int N3)
{
  const long int n1_n2  = n1*n2;
  long int kStart       = 0;
  long int iStart0      = n1*b1*N2*N3 + n2*b2*N3 + n3*b3;
  for (long int c12 = 0; c12 < n1_n2; c12++)
  {
    long int c1, c2;
    NPME_ind2D_2_n1_n2 (c12, n2, c1, c2);

  //long int iStart   = (n1*b1+c1)*N2*N3 + (n2*b2+c2)*N3 + n3*b3;
    long int iStart   = iStart0 + (c1*N2 + c2)*N3;


    memcpy(&Q[iStart], &Qtmp[kStart], n3*sizeof(_Complex double));
    kStart += n3;
  }

}



void NPME_TransformBlock_Block_2_Q (_Complex double *Q, _Complex double *Qtmp,
  const long int N1, const long int N2, const long int N3, 
  const long int n1, const long int n2, const long int n3,
  int nProc)
//input: N1, N2, N3 and Q[N1*N2*N3] (block form)
//       n1 = block size with N1/n1 = integer
//       n2 = block size with N2/n2 = integer
//       n3 = block size with N3/n3 = integer
//output: Q[N1*N2*N3] in original form
//temporary array of size Qtmp(n1*N2*N3)
{
  char str[2000];

  if (N1%n1 != 0)
  {
    std::cout << "Error in NPME_TransformBlock_Block_2_Q.\n";
    sprintf(str, "  N1 = %ld is not divisible by block size n1 = %ld\n", 
      N1, n1);
    std::cout << str;
    exit(0);
  }
  if (N2%n2 != 0)
  {
    std::cout << "Error in NPME_TransformBlock_Block_2_Q.\n";
    sprintf(str, "  N2 = %ld is not divisible by block size n2 = %ld\n", 
      N2, n2);
    std::cout << str;
    exit(0);
  }
  if (N3%n3 != 0)
  {
    std::cout << "Error in NPME_TransformBlock_Block_2_Q.\n";
    sprintf(str, "  N3 = %ld is not divisible by block size n3 = %ld\n", 
      N3, n3);
    std::cout << str;
    exit(0);
  }

  //M1, M2, M3 = #of blocks in each direction
  const long int M1 = N1/n1;
  const long int M2 = N2/n2;
  const long int M3 = N3/n3;
  const long int blockSize = n1*n2*n3;


  const long int arrayBlockSize = N2*N3;
  for (long int b1 = 0; b1 < M1; b1++)
  {
  //memcpy(&Qtmp[0], &Q[b1*n1*N2*N3], n1*N2*N3*sizeof(_Complex double));
    NPME_CopyDoubleArray (2*n1*N2*N3, nProc, (double*) &Qtmp[0], 
      (const double*) &Q[b1*n1*N2*N3], arrayBlockSize);

    long int b23;
    #pragma omp parallel shared(Q, Qtmp) private(b23) num_threads(nProc)
    {
      #pragma omp for schedule(static)
      for (b23 = 0; b23 < M2*M3; b23++)
      {
        long int b2, b3;
        NPME_ind2D_2_n1_n2 (b23, M3, b2, b3);
        NPME_TransformBlock_Block_2_Q_CopyBlock (Q, &Qtmp[blockSize*b23], 
          b1, b2, b3, n1, n2, n3, N2, N3);
      }
    }
  }
}

}//end namespace NPME_Library



