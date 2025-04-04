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
#include "NPME_RecSumGrid.h"
#include "NPME_KernelFunction.h"
#include "NPME_SupportFunctions.h"
#include "NPME_ExtLibrary.h"


namespace NPME_Library
{



//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
//****************************Main Grid Functions******************************
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

bool NPME_RecSumGrid_CalcGenFunc (_Complex double *f,
  const long int N1, const double a1, const double del1, double *x1,
  const long int N2, const double a2, const double del2, double *y1,
  const long int N3, const double a3, const double del3, double *z1,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncReal& func, 
  const double R0[3], int nProc)
//calculates a generic func(x, y, z) on a pre-defined rectangular grid defined 
//by x1[N1], y1[N2], z1[N3] which need not be the full DFT grid 
//note that x1, y1, z1 are modified
{
  std::vector<double> x2(N1);
  std::vector<double> y2(N2);
  std::vector<double> z2(N3);

  std::vector<double> Tx1(N1);
  std::vector<double> Ty1(N2);
  std::vector<double> Tz1(N3);

  std::vector<double> Tx2(N1);
  std::vector<double> Ty2(N2);
  std::vector<double> Tz2(N3);

  const double Xmax = X + del1;
  const double Ymax = Y + del2;
  const double Zmax = Z + del3;


  NPME_CalcOverlapSmoothFunc(&Tx1[0], &Tx2[0], &x2[0], N1, &x1[0], X, Xmax, a1);
  NPME_CalcOverlapSmoothFunc(&Ty1[0], &Ty2[0], &y2[0], N2, &y1[0], Y, Ymax, a2);
  NPME_CalcOverlapSmoothFunc(&Tz1[0], &Tz2[0], &z2[0], N3, &z1[0], Z, Zmax, a3);

  //the T1 and T2 functions are already calculated.  
  //the x1 and x2 coordinate arrays are only needed to calculate f0
  //and can be translated by R0[3]

  //x -> x^2
  for (long int i = 0; i < N1; i++)
  {
    x1[i] += R0[0];
    x2[i] += R0[0];
  }

  //y -> y^2
  for (long int i = 0; i < N2; i++)
  {
    y1[i] += R0[1];
    y2[i] += R0[1];
  }

  //z -> z^2
  for (long int i = 0; i < N3; i++)
  {
    z1[i] += R0[2];
    z2[i] += R0[2];
  }



  long int k;
  #pragma omp parallel for shared (f, x1, x2, y1, y2, z1, z2, Tx1, Tx2, Ty1, Ty2, Tz1, Tz2, func) private(k) default(none) num_threads(nProc)
  for (k = 0; k < N1*N2; k++)
  {
    long int i1, i2;
    NPME_ind2D_2_n1_n2 (k, N2, i1, i2);
    long int count = k*N3;
    double r2, r;
    _Complex double f0;
    for (long int i3 = 0; i3 < N3; i3++)
    {
      f[count] = 0.0;

      double x_f0[8];
      double y[8];
      double z[8];

      x_f0[0] = x1[i1];   y[0] = y1[i2];    z[0] = z1[i3];
      x_f0[1] = x1[i1];   y[1] = y1[i2];    z[1] = z2[i3];
      x_f0[2] = x1[i1];   y[2] = y2[i2];    z[2] = z1[i3];
      x_f0[3] = x1[i1];   y[3] = y2[i2];    z[3] = z2[i3];

      x_f0[4] = x2[i1];   y[4] = y1[i2];    z[4] = z1[i3];
      x_f0[5] = x2[i1];   y[5] = y1[i2];    z[5] = z2[i3];
      x_f0[6] = x2[i1];   y[6] = y2[i2];    z[6] = z1[i3];
      x_f0[7] = x2[i1];   y[7] = y2[i2];    z[7] = z2[i3];

      func.Calc (8, x_f0, y, z);

      //(1,1,1)
      f[count] += Tx1[i1]*Ty1[i2]*Tz1[i3]*x_f0[0];

      //(1,1,2)
      f[count] += Tx1[i1]*Ty1[i2]*Tz2[i3]*x_f0[1];

      //(1,2,1)
      f[count] += Tx1[i1]*Ty2[i2]*Tz1[i3]*x_f0[2];

      //(1,2,2)
      f[count] += Tx1[i1]*Ty2[i2]*Tz2[i3]*x_f0[3];

      //(2,1,1)
      f[count] += Tx2[i1]*Ty1[i2]*Tz1[i3]*x_f0[4];

      //(2,1,2)
      f[count] += Tx2[i1]*Ty1[i2]*Tz2[i3]*x_f0[5];

      //(2,2,1)
      f[count] += Tx2[i1]*Ty2[i2]*Tz1[i3]*x_f0[6];

      //(2,2,2)
      f[count] += Tx2[i1]*Ty2[i2]*Tz2[i3]*x_f0[7];

      count++;
    }
  }

  return true;
}

bool NPME_RecSumGrid_CalcGenFunc (_Complex double *f,
  const long int N1, const double a1, const double del1, double *x1,
  const long int N2, const double a2, const double del2, double *y1,
  const long int N3, const double a3, const double del3, double *z1,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncComplex& func, 
  const double R0[3], int nProc)
//calculates a generic func(x, y, z) on a pre-defined rectangular grid defined 
//by x1[N1], y1[N2], z1[N3] which need not be the full DFT grid 
//note that x1, y1, z1 are modified
{
  std::vector<double> x2(N1);
  std::vector<double> y2(N2);
  std::vector<double> z2(N3);

  std::vector<double> Tx1(N1);
  std::vector<double> Ty1(N2);
  std::vector<double> Tz1(N3);

  std::vector<double> Tx2(N1);
  std::vector<double> Ty2(N2);
  std::vector<double> Tz2(N3);

  const double Xmax = X + del1;
  const double Ymax = Y + del2;
  const double Zmax = Z + del3;


  NPME_CalcOverlapSmoothFunc(&Tx1[0], &Tx2[0], &x2[0], N1, &x1[0], X, Xmax, a1);
  NPME_CalcOverlapSmoothFunc(&Ty1[0], &Ty2[0], &y2[0], N2, &y1[0], Y, Ymax, a2);
  NPME_CalcOverlapSmoothFunc(&Tz1[0], &Tz2[0], &z2[0], N3, &z1[0], Z, Zmax, a3);

  //the T1 and T2 functions are already calculated.  
  //the x1 and x2 coordinate arrays are only needed to calculate f0
  //and can be translated by R0[3]

  //x -> x^2
  for (long int i = 0; i < N1; i++)
  {
    x1[i] += R0[0];
    x2[i] += R0[0];
  }

  //y -> y^2
  for (long int i = 0; i < N2; i++)
  {
    y1[i] += R0[1];
    y2[i] += R0[1];
  }

  //z -> z^2
  for (long int i = 0; i < N3; i++)
  {
    z1[i] += R0[2];
    z2[i] += R0[2];
  }


  long int k;
  #pragma omp parallel for shared (f, x1, x2, y1, y2, z1, z2, Tx1, Tx2, Ty1, Ty2, Tz1, Tz2, func) private(k) default(none) num_threads(nProc)
  for (k = 0; k < N1*N2; k++)
  {
    long int i1, i2;
    NPME_ind2D_2_n1_n2 (k, N2, i1, i2);
    long int count = k*N3;
    double r2, r;
    _Complex double f0;
    for (long int i3 = 0; i3 < N3; i3++)
    {
      f[count] = 0.0;

      double x_f0_r[8];
      double   f0_i[8];
      double y[8];
      double z[8];

      x_f0_r[0] = x1[i1];   y[0] = y1[i2];    z[0] = z1[i3];
      x_f0_r[1] = x1[i1];   y[1] = y1[i2];    z[1] = z2[i3];
      x_f0_r[2] = x1[i1];   y[2] = y2[i2];    z[2] = z1[i3];
      x_f0_r[3] = x1[i1];   y[3] = y2[i2];    z[3] = z2[i3];

      x_f0_r[4] = x2[i1];   y[4] = y1[i2];    z[4] = z1[i3];
      x_f0_r[5] = x2[i1];   y[5] = y1[i2];    z[5] = z2[i3];
      x_f0_r[6] = x2[i1];   y[6] = y2[i2];    z[6] = z1[i3];
      x_f0_r[7] = x2[i1];   y[7] = y2[i2];    z[7] = z2[i3];

      func.Calc (8, x_f0_r, f0_i, y, z);

      //(1,1,1)
      f[count] += Tx1[i1]*Ty1[i2]*Tz1[i3]*(x_f0_r[0] + I*f0_i[0]);

      //(1,1,2)
      f[count] += Tx1[i1]*Ty1[i2]*Tz2[i3]*(x_f0_r[1] + I*f0_i[1]);

      //(1,2,1)
      f[count] += Tx1[i1]*Ty2[i2]*Tz1[i3]*(x_f0_r[2] + I*f0_i[2]);

      //(1,2,2)
      f[count] += Tx1[i1]*Ty2[i2]*Tz2[i3]*(x_f0_r[3] + I*f0_i[3]);

      //(2,1,1)
      f[count] += Tx2[i1]*Ty1[i2]*Tz1[i3]*(x_f0_r[4] + I*f0_i[4]);

      //(2,1,2)
      f[count] += Tx2[i1]*Ty1[i2]*Tz2[i3]*(x_f0_r[5] + I*f0_i[5]);

      //(2,2,1)
      f[count] += Tx2[i1]*Ty2[i2]*Tz1[i3]*(x_f0_r[6] + I*f0_i[6]);

      //(2,2,2)
      f[count] += Tx2[i1]*Ty2[i2]*Tz2[i3]*(x_f0_r[7] + I*f0_i[7]);

      count++;
    }
  }


  return true;
}


//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
//****************************Full Grid Functions******************************
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

bool NPME_RecSumGrid_Full_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncReal& func, int nProc,
  bool PRINT, std::ostream& os)
{
  double R0[3] = {0, 0, 0};

  std::vector<double> x1(N1);
  std::vector<double> y1(N2);
  std::vector<double> z1(N3);

  const double Xmax = X + del1;
  const double Ymax = Y + del2;
  const double Zmax = Z + del3;
  const double L1   = 2*Xmax;
  const double L2   = 2*Ymax;
  const double L3   = 2*Zmax;

  NPME_Set_xArray (&x1[0], N1, L1);
  NPME_Set_xArray (&y1[0], N2, L2);
  NPME_Set_xArray (&z1[0], N3, L3);

  if (PRINT)
  {
    os << "\nNPME_RecSumGrid_Full_Func\n";
  }
  if (!NPME_RecSumGrid_CalcGenFunc (f,
    N1, a1, del1, &x1[0],
    N2, a2, del2, &y1[0],
    N3, a3, del3, &z1[0],
    X,  Y,  Z,
    func, R0, nProc))
  {
    std::cout << "Error in NPME_RecSumGrid_Full_Func\n";
    std::cout << "NPME_RecSumGrid_CalcGenFunc failed\n";
    return false;
  }

  return true;
}
bool NPME_RecSumGrid_Full_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncComplex& func, int nProc,
  bool PRINT, std::ostream& os)
{
  double R0[3] = {0, 0, 0};

  std::vector<double> x1(N1);
  std::vector<double> y1(N2);
  std::vector<double> z1(N3);

  const double Xmax = X + del1;
  const double Ymax = Y + del2;
  const double Zmax = Z + del3;
  const double L1   = 2*Xmax;
  const double L2   = 2*Ymax;
  const double L3   = 2*Zmax;

  NPME_Set_xArray (&x1[0], N1, L1);
  NPME_Set_xArray (&y1[0], N2, L2);
  NPME_Set_xArray (&z1[0], N3, L3);

  if (PRINT)
  {
    os << "\nNPME_RecSumGrid_Full_Func\n";
  }
  if (!NPME_RecSumGrid_CalcGenFunc (f,
    N1, a1, del1, &x1[0],
    N2, a2, del2, &y1[0],
    N3, a3, del3, &z1[0],
    X,  Y,  Z,
    func, R0, nProc))
  {
    std::cout << "Error in NPME_RecSumGrid_Full_Func\n";
    std::cout << "NPME_RecSumGrid_CalcGenFunc failed\n";
    return false;
  }


  return true;
}
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
//*****************Compact Radially Symmetric Grid Functions*******************
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

bool NPME_RecSumGrid_CompactReal_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncReal& func, int nProc,
  bool PRINT, std::ostream& os)
//calculates compact real space grid for spherically symmetric function
{
  const double R0[3]  = {0, 0, 0};
  const long int M1   = N1/2+1;
  const long int M2   = N2/2+1;
  const long int M3   = N3/2+1;

  const double Xmax   = X + del1;
  const double Ymax   = Y + del2;
  const double Zmax   = Z + del3;
  const double L1     = 2*Xmax;
  const double L2     = 2*Ymax;
  const double L3     = 2*Zmax;

  std::vector<double> x1(N1);
  std::vector<double> y1(N2);
  std::vector<double> z1(N3);

  NPME_CompactGrid_Set_xArray (&x1[0], N1, L1);
  NPME_CompactGrid_Set_xArray (&y1[0], N2, L2);
  NPME_CompactGrid_Set_xArray (&z1[0], N3, L3);

  if (PRINT)
  {
    os << "\nNPME_RecSumGrid_CompactReal_Func\n";
  }

  if (!NPME_RecSumGrid_CalcGenFunc (f,
    M1, a1, del1, &x1[0],
    M2, a2, del2, &y1[0],
    M3, a3, del3, &z1[0],
    X, Y, Z, func, R0, nProc))
  {
    std::cout << "Error in NPME_RecSumGrid_CompactReal_Func\n";
    std::cout << "NPME_RecSumGrid_CalcGenFunc failed\n";
    return false;
  }


  return true;
}

bool NPME_RecSumGrid_CompactReal_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncComplex& func, int nProc,
  bool PRINT, std::ostream& os)
//calculates compact real space grid for spherically symmetric function
{
  const double R0[3]  = {0, 0, 0};
  const long int M1   = N1/2+1;
  const long int M2   = N2/2+1;
  const long int M3   = N3/2+1;

  const double Xmax   = X + del1;
  const double Ymax   = Y + del2;
  const double Zmax   = Z + del3;
  const double L1     = 2*Xmax;
  const double L2     = 2*Ymax;
  const double L3     = 2*Zmax;

  std::vector<double> x1(N1);
  std::vector<double> y1(N2);
  std::vector<double> z1(N3);

  NPME_CompactGrid_Set_xArray (&x1[0], N1, L1);
  NPME_CompactGrid_Set_xArray (&y1[0], N2, L2);
  NPME_CompactGrid_Set_xArray (&z1[0], N3, L3);

  if (PRINT)
  {
    os << "\nNPME_RecSumGrid_CompactReal_Func\n";
  }

  if (!NPME_RecSumGrid_CalcGenFunc (f,
    M1, a1, del1, &x1[0],
    M2, a2, del2, &y1[0],
    M3, a3, del3, &z1[0],
    X, Y, Z, func, R0, nProc))
  {
    std::cout << "Error in NPME_RecSumGrid_CompactReal_Func\n";
    std::cout << "NPME_RecSumGrid_CalcGenFunc failed\n";
    return false;
  }


  return true;
}
void NPME_Compact2FullGrid1D (_Complex double *xF,
  const _Complex double *xC, const long int N)
//transforms xC[N/2+1] -> xF[N]
//xC[N/2+1] = {0, 1, 2, ... N/2}
//xF[N]     = {0, 1, 2, ... N/2-1, -N/2, -N/2+1, .. -1}
{
  const long int N_2 = N/2;
  for (long int i = 0; i < N_2; i++)
    xF[i] = xC[i];
  for (long int i = N_2; i < N; i++)
    xF[i] = xC[N - i];
}

void NPME_Full2CompactGrid1D (const _Complex double *xF,
  _Complex double *xC, const long int N)
//transforms xF[N] -> xC[N/2+1] 
//xC[N/2+1] = {0, 1, 2, ... N/2}
//xF[N]     = {0, 1, 2, ... N/2-1, -N/2, -N/2+1, .. -1}
{
  const long int N_2 = N/2;
  xC[0]   = xF[0];
  xC[N_2] = xF[N_2];

  for (long int i = 1; i < N_2; i++)
    xC[i] = 0.5*(xF[i]+xF[N-i]);
}



void NPME_RecSumGrid_CompactTransformReal2FourierGrid (_Complex double *f,
  const long int N1, const long int N2, const long int N3, 
  _Complex double *fTmp, int nProc)
//f   [M1*M2*M3]
//fTmp[M1*M2*M3]
{
  const long int M1 = N1/2+1;
  const long int M2 = N2/2+1;
  const long int M3 = N3/2+1;

  long int n;

//1) z-component
  mkl_set_num_threads (1);
  #pragma omp parallel for schedule(static) shared (f) private(n) default(none) num_threads(nProc)
  for (n = 0; n < M1*M2; n++)
  {
    _Complex double *xC = &f[n*M3];
    std::vector<_Complex double> xF(N3);
    NPME_Compact2FullGrid1D (&xF[0], xC, N3);
    NPME_1D_FFT_NoNorm  (&xF[0], N3);
    NPME_Full2CompactGrid1D (&xF[0], xC, N3);
  }

//2) y-component
  //transpose f[M1][M2][M3] -> f[M1][M3][M2]
  {
    mkl_set_num_threads (nProc);
    for (long int n1 = 0; n1 < M1; n1++)
      NPME_TransposeComplex ((size_t) M2, (size_t) M3, &f[n1*M2*M3]);
  }

  mkl_set_num_threads (1);
  #pragma omp parallel for schedule(static) shared (f) private(n) default(none) num_threads(nProc)
  for (n = 0; n < M1*M3; n++)
  {
    _Complex double *xC = &f[n*M2];
    std::vector<_Complex double> xF(N2);
    NPME_Compact2FullGrid1D (&xF[0], xC, N2);
    NPME_1D_FFT_NoNorm  (&xF[0], N2);
    NPME_Full2CompactGrid1D (&xF[0], xC, N2);
  }

  //transpose f[M1][M3][M2] -> f[M1][M2][M3]
  {
    mkl_set_num_threads (nProc);
    for (long int n1 = 0; n1 < M1; n1++)
      NPME_TransposeComplex ((size_t) M3, (size_t) M2, &f[n1*M2*M3]);
  }

//3) x-component
  //transpose f[M1][M2][M3] -> f[M2][M3][M1], i.e.
  //f[M1][M2*M3] -> fTmp[M2*M3][M1]
  NPME_TransposeSimple ((size_t) M1, (size_t) M2*M3, fTmp, f);

  mkl_set_num_threads (1);
  #pragma omp parallel for schedule(static) shared (fTmp) private(n) default(none) num_threads(nProc)
  for (n = 0; n < M2*M3; n++)
  {
    _Complex double *xC = &fTmp[n*M1];
    std::vector<_Complex double> xF(N1);
    NPME_Compact2FullGrid1D (&xF[0], xC, N1);
    NPME_1D_FFT_NoNorm  (&xF[0], N1);
    NPME_Full2CompactGrid1D (&xF[0], xC, N1);
  }

  //fTmp[M2*M3][M1] -> f[M1][M2*M3]
  NPME_TransposeSimple ( (size_t) M2*M3, (size_t) M1, f, fTmp);
}









bool NPME_RecSumGrid_CompactFourier_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncReal& func, int nProc,
  _Complex double *fTmp, bool PRINT, std::ostream& os)
//f   [M1*M2*M3]
//fTmp[M1*M2*M3]
{
  if (PRINT)
  {
    os << "\nNPME_RecSumGrid_CompactFourier_Func\n";
  }
  if (!NPME_RecSumGrid_CompactReal_Func (&f[0], 
    N1, a1, del1, 
    N2, a2, del2, 
    N3, a3, del3,
    X, Y, Z, func,
    nProc, PRINT, os))
  {
    std::cout << "Error in NPME_RecSumGrid_CompactFourier_Func\n";
    std::cout << "NPME_RecSumGrid_CompactReal_Func failed\n";
    return false;
  }

  NPME_RecSumGrid_CompactTransformReal2FourierGrid (f, N1, N2, N3, fTmp, nProc);

  return true;
}
bool NPME_RecSumGrid_CompactFourier_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncComplex& func, int nProc,
  _Complex double *fTmp, bool PRINT, std::ostream& os)
//f   [M1*M2*M3]
//fTmp[M1*M2*M3]
{
  if (PRINT)
  {
    os << "\nNPME_RecSumGrid_CompactFourier_Func\n";
  }
  if (!NPME_RecSumGrid_CompactReal_Func (&f[0], 
    N1, a1, del1, 
    N2, a2, del2, 
    N3, a3, del3,
    X, Y, Z, func,
    nProc, PRINT, os))
  {
    std::cout << "Error in NPME_RecSumGrid_CompactFourier_Func\n";
    std::cout << "NPME_RecSumGrid_CompactReal_Func failed\n";
    return false;
  }

  NPME_RecSumGrid_CompactTransformReal2FourierGrid (f, N1, N2, N3, fTmp, nProc);

  return true;
}






//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
//******************Double Box Translation Grid Functions**********************
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

bool NPME_RecSumGrid_FullTrans_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncReal& func, const double R0[3], 
  int nProc, bool PRINT, std::ostream& os)
//calculates f(r) = func(|r+R0|) on a DFT grid 
//for r = (x, y, z) given by
//-X <= x <= X
//-Y <= y <= Y
//-Z <= z <= Z
{
  double *x1  = (double *) _mm_malloc (N1*sizeof(double), 64);
  double *y1  = (double *) _mm_malloc (N2*sizeof(double), 64);
  double *z1  = (double *) _mm_malloc (N3*sizeof(double), 64);

  const double Xmax = X + del1;
  const double Ymax = Y + del2;
  const double Zmax = Z + del3;
  const double L1   = 2*Xmax;
  const double L2   = 2*Ymax;
  const double L3   = 2*Zmax;

  NPME_Set_xArray (&x1[0], N1, L1);
  NPME_Set_xArray (&y1[0], N2, L2);
  NPME_Set_xArray (&z1[0], N3, L3);

  double rMin = NPME_CalcMinDistance_R0_r (R0, L1, L2, L3);
  if (rMin < 1.0E-6)
  {
    char str[500];
    std::cout << "Error in NPME_RecSumGrid_FullTrans_Func.\n";
    sprintf(str, "rMin = %le < 1.0E-6.\n", rMin);
    std::cout << str;
    sprintf(str, "box-box separation R0 = %f %f %f should be increased\n",
      R0[0], R0[1], R0[2]);
    std::cout << str;
    exit(0);
  }

  if (PRINT)
  {
    os << "\nNPME_RecSumGrid_FullTrans_Func\n";
  }

  if (!NPME_RecSumGrid_CalcGenFunc (f,
    N1, a1, del1, x1,
    N2, a2, del2, y1,
    N3, a3, del3, z1,
    X, Y, Z, func, R0, nProc))
  {
    std::cout << "Error in NPME_RecSumGrid_FullTrans_Func\n";
    std::cout << "NPME_RecSumGrid_CalcGenFunc failed\n";
    return false;
  }


  _mm_free(x1);
  _mm_free(y1);
  _mm_free(z1);

  return true;
}
bool NPME_RecSumGrid_FullTrans_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncComplex& func, const double R0[3], 
  int nProc, bool PRINT, std::ostream& os)
//calculates f(r) = func(|r+R0|) on a DFT grid 
//for r = (x, y, z) given by
//-X <= x <= X
//-Y <= y <= Y
//-Z <= z <= Z
{
  char str[500];

  double *x1  = (double *) _mm_malloc (N1*sizeof(double), 64);
  double *y1  = (double *) _mm_malloc (N2*sizeof(double), 64);
  double *z1  = (double *) _mm_malloc (N3*sizeof(double), 64);

  const double Xmax = X + del1;
  const double Ymax = Y + del2;
  const double Zmax = Z + del3;
  const double L1   = 2*Xmax;
  const double L2   = 2*Ymax;
  const double L3   = 2*Zmax;

  NPME_Set_xArray (&x1[0], N1, L1);
  NPME_Set_xArray (&y1[0], N2, L2);
  NPME_Set_xArray (&z1[0], N3, L3);

  double rMin = NPME_CalcMinDistance_R0_r (R0, L1, L2, L3);
  if (rMin < 1.0E-6)
  {
    std::cout << "Error in NPME_RecSumGrid_FullTrans_Func.\n";
    sprintf(str, "rMin = %le < 1.0E-6.\n", rMin);
    std::cout << str;
    sprintf(str, "box-box separation R0 = %f %f %f should be increased\n",
      R0[0], R0[1], R0[2]);
    std::cout << str;
    exit(0);
  }

  if (PRINT)
  {
    os << "\nNPME_RecSumGrid_FullTrans_Func\n";
  }

  if (!NPME_RecSumGrid_CalcGenFunc (f,
    N1, a1, del1, x1,
    N2, a2, del2, y1,
    N3, a3, del3, z1,
    X, Y, Z, func, R0, nProc))
  {
    std::cout << "Error in NPME_RecSumGrid_FullTrans_Func\n";
    std::cout << "NPME_RecSumGrid_CalcGenFunc failed\n";
    return false;
  }


  _mm_free(x1);
  _mm_free(y1);
  _mm_free(z1);

  return true;
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
//********************Full Grid Interpolation Function*************************
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
void NPME_RecSumGrid_Full_Interpolate_XYZexpArray (
  const size_t nRandom, const double *r,
  _Complex double *Xexp, _Complex double *Yexp, _Complex double *Zexp,
  const long int N1,     const long int N2,     const long int N3,
  const double L1,       const double L2,       const double L3)
{
  {
    const long int N1_2 = N1/2;
    
    size_t count = 0;
    for (long int n = 0; n < N1; n++)
    for (size_t i = 0; i < nRandom; i++)
    {
      const double C = 2*NPME_Pi/L1*r[3*i+0];
      long int m1 = n;
      if (m1 >= N1_2)  m1 = m1 - N1;
      Xexp[count] = cexp(I*C*m1)/N1;
      count++;
    }
  }

  {
    const long int N2_2 = N2/2;
    
    size_t count = 0;
    for (long int n = 0; n < N2; n++)
    for (size_t i = 0; i < nRandom; i++)
    {
      const double C = 2*NPME_Pi/L2*r[3*i+1];
      long int m2 = n;
      if (m2 >= N2_2)  m2 = m2 - N2;
      Yexp[count] = cexp(I*C*m2)/N2;
      count++;
    }
  }


  {
    const long int N3_2 = N3/2;
    
    size_t count = 0;
    for (size_t i = 0; i < nRandom; i++)
    for (long int n = 0; n < N3; n++)
    {
      const double C = 2*NPME_Pi/L3*r[3*i+2];
      long int m3 = n;
      if (m3 >= N3_2)  m3 = m3 - N3;
      Zexp[count] = cexp(I*C*m3)/N3;
      count++;
    }
  }
}





void NPME_RecSumGrid_Full_Interpolate (const size_t nRandom, 
  _Complex double *f, const double *r,
  const long int N1,  const long int N2,  const long int N3,
  const double L1,    const double L2,    const double L3,
  const _Complex double *Fk, int nProc)
//input  r[3*nRandom], Fk[N1*N2*N3] = un-normalized FFT of grid
//output f[nRandom]
{
  memset(f, 0, nRandom*sizeof(_Complex double));

  std::vector<_Complex double> Xexp(nRandom*N1);
  std::vector<_Complex double> Yexp(nRandom*N2);
  std::vector<_Complex double> Zexp(nRandom*N3);

  NPME_RecSumGrid_Full_Interpolate_XYZexpArray (nRandom, r, 
    &Xexp[0], &Yexp[0], &Zexp[0], 
    N1,       N2,       N3, 
    L1,       L2,       L3);

  std::vector<_Complex double> fTmp;
  long int n;
  #pragma omp parallel for schedule(static) shared (f, Fk, Xexp, Yexp, Zexp) private(n, fTmp) default(none) num_threads(nProc)
  for (n = 0; n < N1*N2; n++)
  {
    fTmp.resize(nRandom);

    long int n1, n2;
    NPME_ind2D_2_n1_n2 (n, N2, n1, n2);

    const _Complex double *Fk_loc   = &Fk[n*N3];
    const _Complex double *Xexp_loc = &Xexp[n1*nRandom];
    const _Complex double *Yexp_loc = &Yexp[n2*nRandom];

    for (size_t m = 0; m < nRandom; m++)
      fTmp[m] = Xexp_loc[m]*Yexp_loc[m]*NPME_N_DotProd (N3, Fk_loc, &Zexp[N3*m]);


    #pragma omp critical (update_NPME_RecSumGrid_Full_Interpolate)
    {
      for (size_t m = 0; m < nRandom; m++)
        f[m] += fTmp[m];
    }
  }
}

//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
//******************Compact Grid Interpolation Function************************
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
void NPME_RecSumGrid_CompactGrid_Interpolate_XexpArray (const size_t nRandom, 
  const double *x, _Complex double *Xexp, const long int N1, const double L1)
//calculates Xexp[M1][nRandom]
{
  const long int M1   = N1/2+1;
  const long int N1_2 = N1/2;
  
  size_t count = 0;
  //m = 0
  for (size_t i = 0; i < nRandom; i++)
  {
    Xexp[count] = 1.0/N1;
    count++;
  }

  //m = 1, 2, .. N1/2-1
  for (long int m = 1; m < N1_2; m++)
  {
    for (size_t i = 0; i < nRandom; i++)
    {
      const double C = 2*NPME_Pi/L1*x[i];
      Xexp[count]    = (cexp(I*C*m) + cexp(-I*C*m))/N1;
      count++;
    }
  }

  //m = N1/2
  for (size_t i = 0; i < nRandom; i++)
  {
    const double C = 2*NPME_Pi/L1*x[i];
    Xexp[count]    = cexp(I*C*N1_2)/N1;
    count++;
  }
}




void NPME_RecSumGrid_CompactGrid_Interpolate (const size_t nRandom, 
  _Complex double *f, const double *r,
  const long int N1,  const long int N2,  const long int N3,
  const double L1,    const double L2,    const double L3,
  const _Complex double *Fk, int nProc)
//input  r[3*nRandom], Fk[N1*N2*N3] = un-normalized FFT of grid
//output f[nRandom]
{
  const long int M1 = N1/2+1;
  const long int M2 = N2/2+1;
  const long int M3 = N3/2+1;

  memset(f, 0, nRandom*sizeof(_Complex double));

  std::vector<_Complex double> Xexp(M1*nRandom); //Xexp[M1][nRandom]
  std::vector<_Complex double> Yexp(M2*nRandom); //Yexp[M2][nRandom]
  std::vector<_Complex double> Zexp(M3*nRandom); //Zexp[M3][nRandom]

  std::vector<double> xRandom(nRandom);
  std::vector<double> yRandom(nRandom);
  std::vector<double> zRandom(nRandom);
  for (size_t i = 0; i < nRandom; i++)
  {
    xRandom[i] = r[3*i  ];
    yRandom[i] = r[3*i+1];
    zRandom[i] = r[3*i+2];
  }


  NPME_RecSumGrid_CompactGrid_Interpolate_XexpArray (nRandom, 
    &xRandom[0], &Xexp[0], N1, L1);
  NPME_RecSumGrid_CompactGrid_Interpolate_XexpArray (nRandom, 
    &yRandom[0], &Yexp[0], N2, L2);
  NPME_RecSumGrid_CompactGrid_Interpolate_XexpArray (nRandom, 
    &zRandom[0], &Zexp[0], N3, L3);

  //transpose Zexp[M3][nRandom] -> Zexp[nRandom][M3]
  {
    mkl_set_num_threads (nProc);
    NPME_TransposeComplex ( (size_t) M3, nRandom, &Zexp[0]);
  }

  std::vector<_Complex double> fTmp;
  long int n;
  #pragma omp parallel for schedule(static) shared (f, Fk, Xexp, Yexp, Zexp) private(n, fTmp) default(none) num_threads(nProc)
  for (n = 0; n < M1*M2; n++)
  {
    fTmp.resize(nRandom);

    long int n1, n2;
    NPME_ind2D_2_n1_n2 (n, M2, n1, n2);

    const _Complex double *Fk_loc   = &Fk[n*M3];
    const _Complex double *Xexp_loc = &Xexp[n1*nRandom];
    const _Complex double *Yexp_loc = &Yexp[n2*nRandom];

    for (size_t m = 0; m < nRandom; m++)
      fTmp[m] = Xexp_loc[m]*Yexp_loc[m]*NPME_N_DotProd (M3, Fk_loc, &Zexp[M3*m]);

    #pragma omp critical (update_FE2_CGSS_Interpolate)
    {
      for (size_t m = 0; m < nRandom; m++)
        f[m] += fTmp[m];
    }
  }
}


//*****************************************************************************
//*****************************************************************************
//****************************Index/Array Functions****************************
//*****************************************************************************
//*****************************************************************************

void NPME_Set_xArray (double *x1, const long int N, const double L)
{
  if (N%2 != 0)
  {
    char str[500];
    sprintf(str, "Error in NPME_Set_xArray.  N = %ld must be even\n", N);
    std::cout << str;
    exit(0);
  }

  const double del   = L/N;
  const long int N_2 = N/2;
  for (long int n = 0; n < N; n++)
  {
    long int s = n;
    if (s >= N_2)  s = s - N;
    x1[n] = s*del;
  }
}
long int NPME_ArrayIndex_2_mIndex (long int arrayIndex, const long int N)
//input:  arrayIndex = (0, 1, .. N-1)
//output: mIndex     = (0, 1, 2, .. N/2-1, -N/2, -N/2+1, .. -1}
{
  if (N%2 != 0)
  {
    char str[500];
    sprintf(str, "Error in NPME_ArrayIndex_2_mIndex.  N = %ld must be even\n", 
      N);
    std::cout << str;
    exit(0);
  }

  if (arrayIndex < N/2) return arrayIndex;
  else                  return arrayIndex - N;
}
long int NPME_mIndex_2_ArrayIndex (long int mIndex, const long int N)
//input:  mIndex     = (0, 1, 2, .. N/2-1, -N/2, -N/2+1, .. -1}
//output: arrayIndex = (0, 1, .. N-1)
{
  if (N%2 != 0)
  {
    char str[500];
    sprintf(str, "Error in NPME_mIndex_2_ArrayIndex.  N = %ld must be even\n", 
      N);
    std::cout << str;
    exit(0);
  }

  if (mIndex < 0)
    return mIndex + N;
  else
    return mIndex;
}


void NPME_CompactGrid_Set_xArray (double *x1, const long int N, const double L)
//L  = 2*Xmax
//dx = L/N
//x1[N/2+1] = (0, dx, 2*dx, .. N/2*dx)
{
  if (N%2 != 0)
  {
    char str[500];
    sprintf(str, "Error in FE2_CGSS_Set_xArray.  N = %ld must be even\n", N);
    std::cout << str;
    exit(0);
  }

  const double del   = L/N;
  const long int N_2 = N/2;
  for (long int n = 0; n <= N_2; n++)
    x1[n] = n*del;
}

long int NPME_CompactGrid_mIndex_2_arrayIndex (const long int m, 
  const long int N)
{
  long int m_abs = abs(m);
  if (m_abs > N/2)
  {
    char str[500];
    std::cout << "Error in NPME_CompactGrid_mIndex_2_arrayIndex.\n";
    sprintf(str, "  m = %ld N = %ld   |m| > N/2\n", m, N);
    std::cout << str;
    exit(0);
  }

  return m_abs;
}





//*****************************************************************************
//*****************************************************************************
//****************************Smooth Functions*********************************
//*****************************************************************************
//*****************************************************************************

double NPME_Hramp (const double x, const double a)
{
  const double erf_max = 6.0;
  if (x < -1.0)
    return 0.0;
  else if (x > 1.0)
    return 1.0;
  else
  //-1 <= x <= 1
  {
    const double z    = erf_max/a;
    const double xMax = z/sqrt(1.0+z*z);
    if (x >= xMax)
      return 1.0;
    else if (x <= -xMax)
      return 0.0;
    else
      return 0.5*(1.0+erf(a*x/sqrt(1.0-x*x)));
  }
}

double NPME_Tsmooth (const double x, const double X, 
  const double Xmax, const double a)
//top hat smoothing function
//T(x) =  1.0 for -X <= x <= X
//T(x) -> 0.0 for  X <= x <= Xmax
//     -> 0.0 for -Xmax <= x <= -X
{
  char str[500];

  const double eps = 1.0E-12*Xmax;
  if (x < -Xmax-eps)
  {
    sprintf(str, "Error in NPME_Tsmooth.  x = %f < -Xmax = %f\n", x, -Xmax);
    std::cout << str;
    exit(0);
  }
  else if (x < -X)
  {
    const double del = (Xmax-X)*0.5;
    const double y   = (x + X + del)/del;
    return NPME_Hramp (y, a);
  }
  else if (x <= X)
    return 1.0;
  else if (x < Xmax+eps)
  {
    const double del = (Xmax-X)*0.5;
    const double y   = (-x + X + del)/del;
    return NPME_Hramp (y, a);

  }
  else
  {
    sprintf(str, "Error in NPME_Tsmooth.  x = %f > Xmax = %f\n", x, Xmax);
    std::cout << str;
    exit(0);
  }
}


void NPME_CalcNonOverlapSmoothFunc (double *T1, 
  const long int N, const double *x1,
  const double X, const double Xmax, const double a)
{
  for (long int i = 0; i < N; i++)
    T1[i] = NPME_Tsmooth (x1[i], X, Xmax, a);
}

void NPME_CalcOverlapSmoothFunc (double *T1, double *T2, 
  double *x2, const long int N, const double *x1,
  const double X, const double Xmax, const double a)
{
  const double eps = 1.0E-12*Xmax;
  for (long int i = 0; i < N; i++)
  {
    if ( (x1[i] < -Xmax-eps) || (x1[i] > Xmax+eps) )
    {
      char str[500];
      std::cout << "Error in NPME_CalcOverlapSmoothFunc.\n";
      sprintf(str, "  x1[%ld] = %f.  |x1| > Xmax = %f\n", i, x1[i], Xmax);
      std::cout << str;
      exit(0);
    }

    if (x1[i] > X)
    //X < x <= Xmax
    {
      T1[i] = NPME_Tsmooth (x1[i], X, 2*Xmax - X, a);
      x2[i] = x1[i] - 2*Xmax;
      T2[i] = NPME_Tsmooth (x2[i], X, 2*Xmax - X, a);
    }
    else if (x1[i] > -X)
    //-X < x <= X
    {
      T1[i] = 1.0;
      x2[i] = 1.0;  //can be arbitrary
      T2[i] = 0.0;
    }
    else 
    //-Xmax <= x <= -X
    {
      T1[i] = NPME_Tsmooth (x1[i], X, 2*Xmax - X, a);
      x2[i] = x1[i] + 2*Xmax;
      T2[i] = NPME_Tsmooth (x2[i], X, 2*Xmax - X, a);
    }
  }
}
void NPME_CalcOverlapSmoothFunc (_Complex double *T1, _Complex double *T2, 
  double *x2, const long int N, const double *x1,
  const double X, const double Xmax, const double a)
{
  const double eps = 1.0E-12*Xmax;
  for (long int i = 0; i < N; i++)
  {
    if ( (x1[i] < -Xmax-eps) || (x1[i] > Xmax+eps) )
    {
      char str[500];
      std::cout << "Error in NPME_CalcOverlapSmoothFunc.\n";
      sprintf(str, "  x1[%ld] = %f.  |x1| > Xmax = %f\n", i, x1[i], Xmax);
      std::cout << str;
      exit(0);
    }

    if (x1[i] > X)
    //X < x <= Xmax
    {
      T1[i] = NPME_Tsmooth (x1[i], X, 2*Xmax - X, a);
      x2[i] = x1[i] - 2*Xmax;
      T2[i] = NPME_Tsmooth (x2[i], X, 2*Xmax - X, a);
    }
    else if (x1[i] > -X)
    //-X < x <= X
    {
      T1[i] = 1.0;
      x2[i] = 1.0;  //can be arbitrary
      T2[i] = 0.0;
    }
    else 
    //-Xmax <= x <= -X
    {
      T1[i] = NPME_Tsmooth (x1[i], X, 2*Xmax - X, a);
      x2[i] = x1[i] + 2*Xmax;
      T2[i] = NPME_Tsmooth (x2[i], X, 2*Xmax - X, a);
    }
  }
}

double NPME_CalcShortRangeSmoothFunc (const double r,
  const double r0, const double del_r, const double a_r)
{
  if (r < r0)
    return 0;
  else
    return NPME_Hramp (2.0/del_r*(r - del_r/2 - r0), a_r);
}


}//end namespace NPME_Library



