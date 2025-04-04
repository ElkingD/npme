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



#include "NPME_Constant.h"
#include "NPME_SupportFunctions.h"
#include "NPME_RecSumQ.h"
#include "NPME_RecSumV.h"
#include "NPME_RecSumGrid.h"
#include "NPME_RecSumSupportFunctions.h"
#include "NPME_Bspline.h"
#include "NPME_ExtLibrary.h"


namespace NPME_Library
{
//******************************************************************************
//******************************************************************************
//******************************************************************************
//**************** Basic Slow V1 code = single charge input*********************
//******************************************************************************
//******************************************************************************
//******************************************************************************
void NPME_RecSumV_CalcV1_Basic (_Complex double *V1, const size_t nCharge, 
  const double *coord, const _Complex double *theta,
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int BnOrder, const double R0[3])
//input:  coord[3*nCharge]
//        theta[N1*N2*N3/8]
//output: V1[4*nCharge] = {V[0], dVdx[0], dVdy[0], dVdz[0], 
//                         V[1], dVdx[1], dVdy[1], dVdz[1],..}
{
  const int BnOrderInt   = (int) BnOrder;

  memset(V1, 0, 4*nCharge*sizeof(_Complex double));

  const double C1 = N1/L1;
  const double C2 = N2/L2;
  const double C3 = N3/L3;

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
      printf("Error in NPME_RecSumV_CalcV1_Basic for x\n");
      printf("  NPME_CheckQpme_DFT_StartIndex failed for mStart1\n");
      exit(0);
    }
    if (!NPME_CheckQpme_DFT_StartIndex (mStart2, N2, BnOrder))
    {
      printf("Error in NPME_RecSumV_CalcV1_Basic for y\n");
      printf("  NPME_CheckQpme_DFT_StartIndex failed for mStart2\n");
      exit(0);
    }
    if (!NPME_CheckQpme_DFT_StartIndex (mStart3, N3, BnOrder))
    {
      printf("Error in NPME_RecSumV_CalcV1_Basic for z\n");
      printf("  NPME_CheckQpme_DFT_StartIndex failed for mStart3\n");
      exit(0);
    }

    double B1[NPME_Bspline_MaxOrder];
    double B2[NPME_Bspline_MaxOrder];
    double B3[NPME_Bspline_MaxOrder];

    double dB1dwX[NPME_Bspline_MaxOrder];
    double dB2dwY[NPME_Bspline_MaxOrder];
    double dB3dwZ[NPME_Bspline_MaxOrder];

    for (long int n1 = 0; n1 < BnOrder; n1++)
    {
      B1[n1] = NPME_Bspline_dBdw (dB1dwX[n1], BnOrderInt, wX + mStart1 + n1);
      B2[n1] = NPME_Bspline_dBdw (dB2dwY[n1], BnOrderInt, wY + mStart2 + n1);
      B3[n1] = NPME_Bspline_dBdw (dB3dwZ[n1], BnOrderInt, wZ + mStart3 + n1);
    }

    _Complex double V1loc[4] = {0, 0, 0, 0};
    for (long int n1 = 0; n1 < BnOrder; n1++)
    {
      const long int m1 = mStart1 + n1;
      const long int p1 = NPME_DFT2Array_Index (m1, N1/2);

      for (long int n2 = 0; n2 < BnOrder; n2++)
      {
        const long int m2 = mStart2 + n2;
        const long int p2 = NPME_DFT2Array_Index (m2, N2/2);

        const double A1   =     B1[n1]*    B2[n2];
        const double A2   = dB1dwX[n1]*    B2[n2];
        const double A3   =     B1[n1]*dB2dwY[n2];

        _Complex double theta_B3 = 0;
        _Complex double theta_dBdZ3 = 0;
        for (long int n3 = 0; n3 < BnOrder; n3++)
        {
          const long int m3   = mStart3 + n3;
          const long int p3   = NPME_DFT2Array_Index (m3, N3/2);
          const long int tInd = NPME_ind3D (p1, p2, p3, N2/2, N3/2);

          theta_B3 += theta[tInd]*B3[n3];
          theta_dBdZ3 += theta[tInd]*dB3dwZ[n3];
        }

        V1loc[0] += A1*theta_B3;
        V1loc[1] += A2*theta_B3;
        V1loc[2] += A3*theta_B3;
        V1loc[3] += A1*theta_dBdZ3;
      }
    }
    V1loc[1] *= C1;
    V1loc[2] *= C2;
    V1loc[3] *= C3;

    V1[4*i  ] = V1loc[0];
    V1[4*i+1] = V1loc[1];
    V1[4*i+2] = V1loc[2];
    V1[4*i+3] = V1loc[3];
  }
}

//******************************************************************************
//******************************************************************************
//******************************************************************************
//**************** Block Q1 code = single charge input**************************
//******************************************************************************
//******************************************************************************
//******************************************************************************
void NPME_RecSumV_CalcBx_dBdx (int& nBx, long int *p1Vec, long int *c1Vec, 
  double *Bx, double *dBxdX, const double wX, const long int BnOrder, 
  const long int N1, const long int n1, const long int b1)
//input: wX, BnOrder, N1, n1 = blockSize, b1 = blockIndex
//output: nBx <= BnOrder = # of x component contributions to Q due to block b1
//        p1Vec[nBx] = DFT indexes          for X inside block b1 only
//        c1Vec[nBx] = local block indexes  for X inside block b1 only
//        Bx   [nBx] = B-spline values      for X inside block b1 only
//        dBxdX[nBx] = B-spline derivatives for X inside block b1 only
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
      Bx[nBx]    = NPME_Bspline_dBdw (dBxdX[nBx], BnOrderInt, wX + m);
      nBx++;
    }
  }
}

void NPME_RecSumV_CalcV1_ContractTheta_B3 (
  _Complex double& theta_B3, _Complex double& theta_dBdZ3,
  const int nB3, const _Complex double *theta, 
  const double *B3, const double *dB3dZ)
{
  theta_B3    = 0.0;
  theta_dBdZ3 = 0.0;
  for (int i = 0; i < nB3; i++)
  {
    theta_B3    += theta[i]*B3[i];
    theta_dBdZ3 += theta[i]*dB3dZ[i];
  }
}

void NPME_RecSumV_CalcV1_Block_UpdateV1 (_Complex double V1[4], 
  const _Complex double *thetaBlock, 
  const int nB1,       const int nB2,       const int nB3, 
  const double *B1,    const double *B2,    const double *B3,
  const double *dB1dX, const double *dB2dY, const double *dB3dZ,
  const long int *c1,  const long int *c2,  const long int *c3,
  const long int n1,   const long int n2,   const long int n3,
  const double C1,     const double C2,     const double C3)
//calculates V1[4] due to a single charge for theta contributions coming
//from inside the block
//input:  thetaBlock[n1*n2*n3]
//        nB1, nB2, nB3 = number of B-spline contributions inside block
//           B1[nB1],    B2[nB2],    B3[nB3],   = B-splines
//        dB1dX[nB1], dB2dY[nB2], dB3dZ[nB3],   = B-spline derivatives
//           c1[nB1],    c2[nB2],    c3[nB3],   = local indexes within block
//        C1 = N1/L1,    C2 = N2/L2  C3 = N3/L3 = chain rule constants
{
  V1[0] = 0.0;
  V1[1] = 0.0;
  V1[2] = 0.0;
  V1[3] = 0.0;

  for (int i1 = 0; i1 < nB1; i1++)
  {
    const long int iTerm1 = c1[i1]*n2*n3 + c3[0];
    for (int i2 = 0; i2 < nB2; i2++)
    {
      const long int iStart = iTerm1 + c2[i2]*n3;

      const double A1 =    B1[i1]*   B2[i2];
      const double A2 = dB1dX[i1]*   B2[i2];
      const double A3 =    B1[i1]*dB2dY[i2];

      _Complex double theta_B3, theta_dBdZ3;
      NPME_RecSumV_CalcV1_ContractTheta_B3 (theta_B3, theta_dBdZ3, 
        nB3, &thetaBlock[iStart], B3, dB3dZ);

      V1[0] += A1*theta_B3;
      V1[1] += A2*theta_B3;
      V1[2] += A3*theta_B3;
      V1[3] += A1*theta_dBdZ3;
    }
  }

  V1[1] *= C1;
  V1[2] *= C2;
  V1[3] *= C3;
}



void NPME_RecSumV_CalcV1_Block (_Complex double *V1, 
  const _Complex double *thetaBlock, const size_t nCharge, const double *coord,
  const double   L1, const double   L2, const double   L3, 
  const long int N1, const long int N2, const long int N3, 
  const long int n1, const long int n2, const long int n3, 
  const long int b1, const long int b2, const long int b3, 
  const long int BnOrder, const double R0[3])
{
  const double C1 = N1/L1;
  const double C2 = N2/L2;
  const double C3 = N3/L3;

  for (size_t n = 0; n < nCharge; n++)
  {
    const double *r           = &coord[3*n];
    const double X            = C1*(r[0]-R0[0]);
    const double Y            = C2*(r[1]-R0[1]);
    const double Z            = C3*(r[2]-R0[2]);

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
    double dB1dX[NPME_Bspline_MaxOrder];
    double dB2dY[NPME_Bspline_MaxOrder];
    double dB3dZ[NPME_Bspline_MaxOrder];

    NPME_RecSumV_CalcBx_dBdx (nB1, p1Vec, c1Vec, B1, dB1dX, 
      X, BnOrder, N1, n1, b1);
    NPME_RecSumV_CalcBx_dBdx (nB2, p2Vec, c2Vec, B2, dB2dY, 
      Y, BnOrder, N2, n2, b2);
    NPME_RecSumV_CalcBx_dBdx (nB3, p3Vec, c3Vec, B3, dB3dZ, 
      Z, BnOrder, N3, n3, b3);

    _Complex double V1loc[4];
    NPME_RecSumV_CalcV1_Block_UpdateV1 (V1loc, thetaBlock,
      nB1,   nB2,   nB3,
      B1,    B2,    B3,
      dB1dX, dB2dY, dB3dZ,
      c1Vec, c2Vec, c3Vec, 
      n1,    n2,    n3,
      C1,    C2,    C3);

    V1[4*n  ] = V1loc[0];
    V1[4*n+1] = V1loc[1];
    V1[4*n+2] = V1loc[2];
    V1[4*n+3] = V1loc[3];
  }
}

void NPME_RecSumV_CalcV1 (double *V1, _Complex double *theta, 
  const size_t nCharge, const double *coord, 
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int n1, const long int n2, const long int n3,
  const long int BnOrder, const int nProc,
  const size_t nRecBlockTot, const long int *RecBlock2BlockIndex,
  const size_t *nRecBlock2Charge, const size_t **RecBlock2Charge,
  const double R0[3], int outputFormat, bool PRINT, std::ostream& os)
//V1[nCharge][4] is not initialized to zero
//input:  nCharge, coord[3*nCharge], BnOrder = Bspline order
//        N1, N2, N3 = DFT Sizes
//        L1, L2, L3 = DFT box size
//        n1, n2, n3 = block sizes
//        n1, n2, n3 >= BnOrder
//        theta[N1*N2*N3/8] = theta potential assumed already calculated
//        nBlock2Charge[nBlock], Block2Charge[nBlock][], Block2V[nBlock]
//        nBlock = M1*M2*M3 (M1 = N1/2/n1, M2 = N2/2/n2, M3 = N3/2/n3)
//        R0[3] = center of coord
//output: V1[nCharge][4]  (outputFormat = 1)
//        V1[4][nCharge]  (outputFormat = 2)
//for a given charge, there are 4 components for (V, dVdr[3])
{
  double time0, time;
  char str[2000];

//const long int M1 = N1/2/n1;
  const long int M2 = N2/2/n2;
  const long int M3 = N3/2/n3;


  const long int blockSize = n1*n2*n3;


  if (PRINT)
  {
    os << "\n      NPME_RecSumV_CalcV1\n";
    os.flush();
  }


  size_t maxNumCharge = 0;
  for (size_t n = 0; n < nRecBlockTot; n++)
    if (maxNumCharge < nRecBlock2Charge[n])
      maxNumCharge = nRecBlock2Charge[n];
  std::vector<_Complex double> V1_loc;
  std::vector<double> coord_loc;

  size_t n;
  time0 = NPME_GetTime ();
  #pragma omp parallel shared(V1, theta, coord, RecBlock2BlockIndex, nRecBlock2Charge, RecBlock2Charge) private(n, coord_loc, V1_loc) num_threads(nProc)
  {
    #pragma omp for schedule(dynamic)
    for (n = 0; n < nRecBlockTot; n++)
    {
      const size_t nChargeLoc = nRecBlock2Charge[n];
      V1_loc.resize(4*maxNumCharge);
      coord_loc.resize(3*maxNumCharge);

      const size_t *chargeIndexLoc = RecBlock2Charge[n];
      for (size_t i = 0; i < nChargeLoc; i++)
      {
        const size_t index  = chargeIndexLoc[i];
        coord_loc[3*i  ]    = coord[3*index  ];
        coord_loc[3*i+1]    = coord[3*index+1];
        coord_loc[3*i+2]    = coord[3*index+2];
      }

      const long int b1     = RecBlock2BlockIndex[3*n  ];
      const long int b2     = RecBlock2BlockIndex[3*n+1];
      const long int b3     = RecBlock2BlockIndex[3*n+2];
      const long int nIndex = NPME_ind3D (b1, b2, b3, M2, M3);

      NPME_RecSumV_CalcV1_Block (&V1_loc[0], &theta[nIndex*blockSize], 
        nChargeLoc, &coord_loc[0], 
        L1, L2, L3, 
        N1, N2, N3, 
        n1, n2, n3, 
        b1, b2, b3, 
        BnOrder, R0);

      #pragma omp critical (update_NPME_RecSumV_CalcV1)
      if (outputFormat == 1)
      {
        for (size_t i = 0; i < nChargeLoc; i++)
        {
          size_t index = 4*chargeIndexLoc[i];
          V1[index  ] += creal(V1_loc[4*i  ]);
          V1[index+1] += creal(V1_loc[4*i+1]);
          V1[index+2] += creal(V1_loc[4*i+2]);
          V1[index+3] += creal(V1_loc[4*i+3]);
        }
      }
      else if (outputFormat == 2)
      {
        double *V0 = &V1[0];
        double *VX = &V1[nCharge];
        double *VY = &V1[nCharge*2];
        double *VZ = &V1[nCharge*3];

        for (size_t i = 0; i < nChargeLoc; i++)
        {
          size_t index = chargeIndexLoc[i];
          V0[index] += creal(V1_loc[4*i  ]);
          VX[index] += creal(V1_loc[4*i+1]);
          VY[index] += creal(V1_loc[4*i+2]);
          VZ[index] += creal(V1_loc[4*i+3]);
        }
      }
      else
      {
        sprintf(str, "Error in NPME_RecSumV_CalcV1.  outputFormat = %d must be 1 or 2\n",
          outputFormat);
        os << str;
        exit(0);
      }
    }
  }

  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, "      time V calc          = %6.2le\n", time);
    os << str;
    os.flush();
  }
}
void NPME_RecSumV_CalcV1 (_Complex double *V1, _Complex double *theta, 
  const size_t nCharge, const double *coord, 
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int n1, const long int n2, const long int n3,
  const long int BnOrder, const int nProc,
  const size_t nRecBlockTot, const long int *RecBlock2BlockIndex,
  const size_t *nRecBlock2Charge, const size_t **RecBlock2Charge,
  const double R0[3], int outputFormat, bool PRINT, std::ostream& os)
//V1[nCharge][4] is not initialized to zero
//input:  nCharge, coord[3*nCharge], BnOrder = Bspline order
//        N1, N2, N3 = DFT Sizes
//        L1, L2, L3 = DFT box size
//        n1, n2, n3 = block sizes
//        n1, n2, n3 >= BnOrder
//        theta[N1*N2*N3/8] = theta potential assumed already calculated
//        nBlock2Charge[nBlock], Block2Charge[nBlock][], Block2V[nBlock]
//        nBlock = M1*M2*M3 (M1 = N1/2/n1, M2 = N2/2/n2, M3 = N3/2/n3)
//        R0[3] = center of coord
//output: V1[nCharge][4]  (outputFormat = 1)
//        V1[4][nCharge]  (outputFormat = 2)
//for a given charge, there are 4 components for (V, dVdr[3])
{
  double time0, time;
  char str[2000];

//const long int M1 = N1/2/n1;
  const long int M2 = N2/2/n2;
  const long int M3 = N3/2/n3;


  const long int blockSize = n1*n2*n3;


  if (PRINT)
  {
    os << "\n      NPME_RecSumV_CalcV1\n";
    os.flush();
  }


  size_t maxNumCharge = 0;
  for (size_t n = 0; n < nRecBlockTot; n++)
    if (maxNumCharge < nRecBlock2Charge[n])
      maxNumCharge = nRecBlock2Charge[n];
  std::vector<_Complex double> V1_loc;
  std::vector<double> coord_loc;

  size_t n;
  time0 = NPME_GetTime ();
  #pragma omp parallel shared(V1, theta, coord, RecBlock2BlockIndex, nRecBlock2Charge, RecBlock2Charge) private(n, coord_loc, V1_loc) num_threads(nProc)
  {
    #pragma omp for schedule(dynamic)
    for (n = 0; n < nRecBlockTot; n++)
    {
      const size_t nChargeLoc = nRecBlock2Charge[n];
      V1_loc.resize(4*maxNumCharge);
      coord_loc.resize(3*maxNumCharge);

      const size_t *chargeIndexLoc = RecBlock2Charge[n];
      for (size_t i = 0; i < nChargeLoc; i++)
      {
        const size_t index  = chargeIndexLoc[i];
        coord_loc[3*i  ]    = coord[3*index  ];
        coord_loc[3*i+1]    = coord[3*index+1];
        coord_loc[3*i+2]    = coord[3*index+2];
      }

      const long int b1     = RecBlock2BlockIndex[3*n  ];
      const long int b2     = RecBlock2BlockIndex[3*n+1];
      const long int b3     = RecBlock2BlockIndex[3*n+2];
      const long int nIndex = NPME_ind3D (b1, b2, b3, M2, M3);

      NPME_RecSumV_CalcV1_Block (&V1_loc[0], &theta[nIndex*blockSize], 
        nChargeLoc, &coord_loc[0], 
        L1, L2, L3, 
        N1, N2, N3, 
        n1, n2, n3, 
        b1, b2, b3, 
        BnOrder, R0);

      #pragma omp critical (update_NPME_RecSumV_CalcV1)
      if (outputFormat == 1)
      {
        for (size_t i = 0; i < nChargeLoc; i++)
        {
          size_t index = 4*chargeIndexLoc[i];
          V1[index  ] += V1_loc[4*i  ];
          V1[index+1] += V1_loc[4*i+1];
          V1[index+2] += V1_loc[4*i+2];
          V1[index+3] += V1_loc[4*i+3];
        }
      }
      else if (outputFormat == 2)
      {
        _Complex double *V0 = &V1[0];
        _Complex double *VX = &V1[nCharge];
        _Complex double *VY = &V1[nCharge*2];
        _Complex double *VZ = &V1[nCharge*3];

        for (size_t i = 0; i < nChargeLoc; i++)
        {
          size_t index = chargeIndexLoc[i];
          V0[index] += V1_loc[4*i  ];
          VX[index] += V1_loc[4*i+1];
          VY[index] += V1_loc[4*i+2];
          VZ[index] += V1_loc[4*i+3];
        }
      }
      else
      {
        sprintf(str, "Error in NPME_RecSumV_CalcV1.  outputFormat = %d must be 1 or 2\n",
          outputFormat);
        std::cout << str;
        exit(0);
      }
    }
  }

  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, "      time V calc          = %6.2le\n", time);
    os << str;
    os.flush();
  }
}



void NPME_RSTP_CalcLamda (const long int N1, const long int BnOrder, 
  _Complex double *lamda1, double *lamda1_2)
{
  for (long int nIndex = 0; nIndex < N1; nIndex++)
  {
    const long int n  = NPME_Array2DFT_Index (nIndex, N1);
    const double z    = (double) n / (double) N1;

    _Complex double denom = 0.0 + 0.0*I;
    for (int p = 0; p <= (int) BnOrder; p++)
      denom += NPME_Bspline_B ( (int) BnOrder, p)*cexp(-2.0*NPME_Pi*I*z*p);

    _Complex double l = 1.0/denom;
    lamda1[nIndex]   = l;
    lamda1_2[nIndex] = creal( l*conj(l) );
  }
}

void NPME_RSTP_CalcT_FullF_Contribution (
  _Complex double *X, const _Complex double *F, 
  const long int N1,      const long int N2,      const long int N3,
  const long int a1,      const long int a2,      const long int a3,
  const _Complex double *C1, 
  const _Complex double *C2, 
  const _Complex double *C3,
  const double *lamda1_2, 
  const double *lamda2_2, 
  const double *lamda3_2,
  int nProc)
//input:  X [N1*N2*N3/8]  = initialized with compact even/odd FFT(Q) contrib.
//        F [N1*N2*N3]    = full FFT of F
//        a1, a2, a3      = 0, 1 (0 for even, 1 for odd)
//        C1[N1/2]        = cexp(-2*NPME_Pi*I/N1*m1)  for -N1/4 <= m1 <= N1/4 - 1
//        C2[N2/2]        = cexp(-2*NPME_Pi*I/N2*m2)  for -N2/4 <= m2 <= N2/4 - 1
//        C3[N3/2]        = cexp(-2*NPME_Pi*I/N3*m3)  for -N3/4 <= m3 <= N3/4 - 1
//        lamda1_2[N1]
//        lamda2_2[N2]
//        lamda3_2[N3]
//output: X [N1*N2*N3/8]  = compact even/odd contribution to theta
{
  const long int M1 = N1/2;
  const long int M2 = N2/2;
  const long int M3 = N3/2;

  long int k;
  #pragma omp parallel for schedule(dynamic) shared(X, F, lamda1_2, lamda2_2, lamda3_2) private(k) num_threads(nProc) default(none)
  for (k = 0; k < M1*M2; k++)
  {
    long int n1, n2;
    NPME_ind2D_2_n1_n2 (k, M2, n1, n2);

    long int m1       = NPME_Array2DFT_Index (n1, M1);
    long int m2       = NPME_Array2DFT_Index (n2, M2);

    long int mTot1    = 2*m1+a1;
    long int mTot2    = 2*m2+a2;

    long int nTot1    = NPME_DFT2Array_Index (mTot1, N1);
    long int nTot2    = NPME_DFT2Array_Index (mTot2, N2);

    const double C    = lamda1_2[nTot1]*lamda2_2[nTot2]/(N1*N2*N3);

    long int Xindex   = NPME_ind3D (n1, n2, 0, N2/2, N3/2);
    long int Findex0  = nTot1*N2*N3 + nTot2*N3;
    long int nTot3    = a3;
    for (long int n3 = 0; n3 < M3; n3++)
    {
      X[Xindex] = C*conj(X[Xindex])*F[Findex0+nTot3]*lamda3_2[nTot3];
      Xindex++;
      nTot3  += 2;
    }
  }
}

void NPME_RSTP_CalcT_CompactSphereSymmF_Contribution (
  _Complex double *X, const _Complex double *F, 
  const long int N1,      const long int N2,      const long int N3,
  const long int a1,      const long int a2,      const long int a3,
  const _Complex double *C1, 
  const _Complex double *C2, 
  const _Complex double *C3,
  const double *lamda1_2, 
  const double *lamda2_2, 
  const double *lamda3_2,
  int nProc)
//input:  X [N1*N2*N3/8]  = initialized with compact even/odd FFT(Q) contrib.
//        F [sizeF]       = full FFT of F, sizeF = (N1/2+1)*(N2/2+1)*(N3/2+1)
//        a1, a2, a3      = 0, 1 (0 for even, 1 for odd)
//        C1[N1/2]        = cexp(-2*NPME_Pi*I/N1*m1)  for -N1/4 <= m1 <= N1/4 - 1
//        C2[N2/2]        = cexp(-2*NPME_Pi*I/N2*m2)  for -N2/4 <= m2 <= N2/4 - 1
//        C3[N3/2]        = cexp(-2*NPME_Pi*I/N3*m3)  for -N3/4 <= m3 <= N3/4 - 1
//        lamda1_2[N1]
//        lamda2_2[N2]
//        lamda3_2[N3]
//output: X [N1*N2*N3/8]  = compact even/odd contribution to theta
{
  const long int M1 = N1/2;
  const long int M2 = N2/2;
  const long int M3 = N3/2;

  long int k;
  #pragma omp parallel for schedule(dynamic) shared(X, F, lamda1_2, lamda2_2, lamda3_2) private(k) num_threads(nProc) default(none)
  for (k = 0; k < M1*M2; k++)
  {
    long int n1, n2;
    NPME_ind2D_2_n1_n2 (k, M2, n1, n2);

    long int m1       = NPME_Array2DFT_Index (n1, M1);
    long int m2       = NPME_Array2DFT_Index (n2, M2);

    long int mTot1    = 2*m1+a1;
    long int mTot2    = 2*m2+a2;

    long int nTot1    = NPME_CompactGrid_mIndex_2_arrayIndex (mTot1, N1);
    long int nTot2    = NPME_CompactGrid_mIndex_2_arrayIndex (mTot2, N2);

    long int kTot1    = NPME_DFT2Array_Index (mTot1, N1);
    long int kTot2    = NPME_DFT2Array_Index (mTot2, N2);

    const double C    = lamda1_2[kTot1]*lamda2_2[kTot2]/(N1*N2*N3);

    long int Xindex   = NPME_ind3D (n1, n2, 0, N2/2, N3/2);
    long int Findex0  = NPME_ind3D (nTot1, nTot2, 0, N2/2+1, N3/2+1);
    long int kTot3    = a3;

    for (long int n3 = 0; n3 < M3/2; n3++)
    {
      X[Xindex] = C*conj(X[Xindex])*F[Findex0+kTot3]*lamda3_2[kTot3];
      Xindex++;
      kTot3 += 2;
    }

    Findex0 = NPME_ind3D (nTot1, nTot2, M3-a3, N2/2+1, N3/2+1);
    long int iTemp = 0;
    for (long int n3 = 0; n3 < M3/2; n3++)
    {
      X[Xindex] = C*conj(X[Xindex])*F[Findex0-iTemp]*lamda3_2[kTot3];
      Xindex++;
      iTemp += 2;
      kTot3 += 2;
    }
  }
}

void NPME_RSTP_CalcT_UpdateThetaContribution (_Complex double *T,
  const _Complex double *X, 
  const long int N1,      const long int N2,      const long int N3,
  const long int a1,      const long int a2,      const long int a3,
  const _Complex double *C1, 
  const _Complex double *C2, 
  const _Complex double *C3,
  int nProc)
//input:  X [N1*N2*N3/8]  = initialized with compact even/odd FFT(Q) contrib.
//        F [N1*N2*N3]    = full FFT of F
//        a1, a2, a3      = 0, 1 (0 for even, 1 for odd)
//        C1[N1/2]        = cexp(-2*NPME_Pi*I/N1*m1)  for -N1/4 <= m1 <= N1/4 - 1
//        C2[N2/2]        = cexp(-2*NPME_Pi*I/N2*m2)  for -N2/4 <= m2 <= N2/4 - 1
//        C3[N3/2]        = cexp(-2*NPME_Pi*I/N3*m3)  for -N3/4 <= m3 <= N3/4 - 1
//        lamda1_2[N1]
//        lamda2_2[N2]
//        lamda3_2[N3]
//output: X [N1*N2*N3/8]  = compact even/odd contribution to theta
{
  const long int M1 = N1/2;
  const long int M2 = N2/2;
  const long int M3 = N3/2;

  long int k;
  #pragma omp parallel for schedule(dynamic) shared(T, X, C1, C2, C3) private(k) num_threads(nProc) default(none)
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
        T[count] += X[count]*C;
        count++;
      }
    }
    else
    {
      for (long int n3 = 0; n3 < M3; n3++)
      {
        T[count] += X[count]*C*C3[n3];
        count++;
      }
    }
  }
}




void NPME_RSTP_CalcTheta (
  _Complex double *T, _Complex double *X, 
  const _Complex double *F, const _Complex double *Qr,
  const long int N1, const long int N2, const long int N3,
  const _Complex double *C1, 
  const _Complex double *C2, 
  const _Complex double *C3,
  const double *lamda1_2, 
  const double *lamda2_2, 
  const double *lamda3_2,
  bool useSphereSymmCompactF,
  int nProc, bool PRINT, std::ostream& os)
//input:  Qr[N1*N2*N3/8]  = real compact Q
//        if useSphereSymmCompactF == 1
//          F [sizeF]     = spherically symmetric compact F
//                          sizeF = (N1/2+1)*(N2/2+1)*(N3/2+1)
//        else
//          F [N1*N2*N3]  = full FFT of F
//
//        a1, a2, a3      = 0, 1 (0 for even, 1 for odd)
//        X [N1*N2*N3/8]  = compact temp array
//
//output: T [N1*N2*N3/8]  = compact FFT of even/odd contribution to theta
{
  double time0, time;
  char str[2000];

  //1) initialize T[N1*N2*N3/8] to zero
  time0 = NPME_GetTime ();
  memset(T, 0, (size_t) (N1*N2*N3/8)*sizeof(_Complex double));
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    os << "\n    NPME_RSTP_CalcT_FullF\n";
    sprintf(str, "      time to zero T = %le\n", time);
    os << str;
    os.flush();
  }


  //2) even/odd contributions to FFT of theta
  time0 = NPME_GetTime ();
  for (long int a1 = 0; a1 <= 1; a1++)
  for (long int a2 = 0; a2 <= 1; a2++)
  for (long int a3 = 0; a3 <= 1; a3++)
  {
    //3) even/odd of Qfft
    NPME_RecSumQ_CalcQfft (X, Qr, N1, N2, N3, a1, a2, a3,
      C1, C2, C3, nProc, PRINT, os);

    //4) calculate even/odd contribution to theta
    if (useSphereSymmCompactF)
      NPME_RSTP_CalcT_CompactSphereSymmF_Contribution (X, F, N1, N2, N3, 
        a1, a2, a3, C1, C2, C3, lamda1_2, lamda2_2, lamda3_2, nProc);
    else
      NPME_RSTP_CalcT_FullF_Contribution (X, F, N1, N2, N3, 
        a1, a2, a3, C1, C2, C3, lamda1_2, lamda2_2, lamda3_2, nProc);

    //5) FFT of theta
    mkl_set_num_threads (nProc);
    NPME_3D_FFT_NoNorm (X, N1/2, N2/2, N3/2);
    mkl_set_num_threads (1);

    //6) update compact FFT of theta
    NPME_RSTP_CalcT_UpdateThetaContribution (T, X, N1, N2, N3, a1, a2, a3,
      C1, C2, C3, nProc);
  }
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, "      time to calc T = %le\n", time);
    os << str;
    os.flush();
  }
}










void NPME_RSTP_CalcT_FullF_Contribution_ReverseSign (
  _Complex double *X, const _Complex double *F, 
  const long int N1,      const long int N2,      const long int N3,
  const long int a1,      const long int a2,      const long int a3,
  const _Complex double *C1, 
  const _Complex double *C2, 
  const _Complex double *C3,
  const double *lamda1_2, 
  const double *lamda2_2, 
  const double *lamda3_2,
  const bool reverseSignX, const bool reverseSignY, const bool reverseSignZ, 
  int nProc)
//input:  X [N1*N2*N3/8]  = initialized with compact even/odd FFT(Q) contrib.
//        F [N1*N2*N3]    = full FFT of F
//        a1, a2, a3      = 0, 1 (0 for even, 1 for odd)
//        C1[N1/2]        = cexp(-2*NPME_Pi*I/N1*m1)  for -N1/4 <= m1 <= N1/4 - 1
//        C2[N2/2]        = cexp(-2*NPME_Pi*I/N2*m2)  for -N2/4 <= m2 <= N2/4 - 1
//        C3[N3/2]        = cexp(-2*NPME_Pi*I/N3*m3)  for -N3/4 <= m3 <= N3/4 - 1
//        lamda1_2[N1]
//        lamda2_2[N2]
//        lamda3_2[N3]
//output: X [N1*N2*N3/8]  = compact even/odd contribution to theta
{
  const long int M1 = N1/2;
  const long int M2 = N2/2;
  const long int M3 = N3/2;

  long int k;
  #pragma omp parallel for schedule(dynamic) shared(X, F, lamda1_2, lamda2_2, lamda3_2) private(k) num_threads(nProc) default(none)
  for (k = 0; k < M1*M2; k++)
  {
    long int n1, n2;
    NPME_ind2D_2_n1_n2 (k, M2, n1, n2);

    long int m1       = NPME_Array2DFT_Index (n1, M1);
    long int m2       = NPME_Array2DFT_Index (n2, M2);

    long int mTot1    = 2*m1+a1;
    long int mTot2    = 2*m2+a2;

    if (reverseSignX) mTot1 = -mTot1;
    if (reverseSignY) mTot2 = -mTot2;

    long int nTot1    = NPME_DFT2Array_Index (mTot1, N1);
    long int nTot2    = NPME_DFT2Array_Index (mTot2, N2);
    const double C    = lamda1_2[nTot1]*lamda2_2[nTot2]/(N1*N2*N3);

    const long int Findex0  = nTot1*N2*N3 + nTot2*N3;
    long int Xindex         = NPME_ind3D (n1, n2, 0, N2/2, N3/2);
    if (!reverseSignZ)
    {
      long int nTot3 = a3;
      for (long int n3 = 0; n3 < M3; n3++)
      {
        X[Xindex] = C*conj(X[Xindex])*F[Findex0+nTot3]*lamda3_2[nTot3];
        Xindex++;
        nTot3  += 2;
      }
    }
    else
    {
      if (a3 == 0)
      //a3 == 0
      {
        //m3 == 0 term
        X[Xindex] = C*conj(X[Xindex])*F[Findex0]*lamda3_2[0];
        Xindex++;

        long int nTot3 = N3 - 2;
        for (long int n3 = 1; n3 < M3; n3++)
        {
          X[Xindex] = C*conj(X[Xindex])*F[Findex0 + nTot3]*lamda3_2[nTot3];
          Xindex++;
          nTot3 -= 2;
        }
      }
      else
      //a3 == 1
      {
        long int nTot3 = N3 - 1;
        for (long int n3 = 0; n3 < M3; n3++)
        {
          X[Xindex] = C*conj(X[Xindex])*F[Findex0 + nTot3]*lamda3_2[nTot3];
          Xindex++;
          nTot3  -= 2;
        }
      }
    }
  }
}


void NPME_RSTP_CalcTheta_FullF_ReverseSign (
  _Complex double *T, _Complex double *X, 
  const _Complex double *F, const _Complex double *Qr,
  const long int N1, const long int N2, const long int N3,
  const _Complex double *C1, 
  const _Complex double *C2, 
  const _Complex double *C3,
  const double *lamda1_2, 
  const double *lamda2_2, 
  const double *lamda3_2,
  bool reverseSignX, bool reverseSignY, bool reverseSignZ, 
  int nProc, bool PRINT, std::ostream& os)
//input:  Qr[N1*N2*N3/8]  = real compact Q
//          F [N1*N2*N3]  = full FFT of F for translated helm potential for 
//                          R0[3] = {X0, Y0, Z0}
//        reverseSignX    = 1 uses F corresponding to R0[3] = {-X0, Y0, Z0}
//        X [N1*N2*N3/8]  = compact temp array
//output: T [N1*N2*N3/8]  = compact FFT of even/odd contribution to theta
{
  double time0, time;
  char str[2000];

  //1) initialize T[N1*N2*N3/8] to zero
  time0 = NPME_GetTime ();
  memset(T, 0, (size_t) (N1*N2*N3/8)*sizeof(_Complex double));
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    os << "\n    NPME_RSTP_CalcT_FullF\n";
    sprintf(str, "      time to zero T = %le\n", time);
    os << str;
    os.flush();
  }


  //2) even/odd contributions to FFT of theta
  time0 = NPME_GetTime ();
  for (long int a1 = 0; a1 <= 1; a1++)
  for (long int a2 = 0; a2 <= 1; a2++)
  for (long int a3 = 0; a3 <= 1; a3++)
  {
    //3) even/odd of Qfft
    NPME_RecSumQ_CalcQfft (X, Qr, N1, N2, N3, a1, a2, a3,
      C1, C2, C3, nProc, PRINT, os);

    //4) calculate even/odd contribution to theta
    NPME_RSTP_CalcT_FullF_Contribution_ReverseSign (X, F, N1, N2, N3, 
        a1, a2, a3, C1, C2, C3, lamda1_2, lamda2_2, lamda3_2, 
        reverseSignX, reverseSignY, reverseSignZ, nProc);

    //5) FFT of theta
    mkl_set_num_threads (nProc);
    NPME_3D_FFT_NoNorm (X, N1/2, N2/2, N3/2);
    mkl_set_num_threads (1);

    //6) update compact FFT of theta
    NPME_RSTP_CalcT_UpdateThetaContribution (T, X, N1, N2, N3, a1, a2, a3,
      C1, C2, C3, nProc);
  }
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, "      time to calc T = %le\n", time);
    os << str;
    os.flush();
  }
}

}//end namespace NPME_Library




