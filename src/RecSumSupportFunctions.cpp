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



#include "Constant.h"
#include "RecSumSupportFunctions.h"
#include "SupportFunctions.h"
#include "RecSumQ.h"
#include "ExtLibrary.h"
#include "Bspline.h"


namespace NPME_Library
{


bool NPME_CheckFFTParm (long int Bn,
  long int N1, long int N2, long int N3,
  long int n1, long int n2, long int n3)
{
  using std::cout;
  const char *errorStr = "Error in NPME_CheckFFTParm.\n";

  //check FFT sizes are multiples of 4
  if (N1%4 != 0)
  {
    cout << errorStr << "  N1 = " << N1 << " is not a multiple of 4\n";
    return false;
  }
  if (N2%4 != 0)
  {
    cout << errorStr << "  N2 = " << N2 << " is not a multiple of 4\n";
    return false;
  }
  if (N3%4 != 0)
  {
    cout << errorStr << "  N3 = " << N3 << " is not a multiple of 4\n";
    return false;
  }

  //check B-spline order is smaller than the +/- interval of interest
  if (Bn >= N1/4)
  {
    cout << errorStr;
    cout << "  BsplineOrder = " << Bn << " >= " << N1/4 << " = N1/4\n";
    return false;
  }
  if (Bn >= N2/4)
  {
    cout << errorStr;
    cout << "  BsplineOrder = " << Bn << " >= " << N2/4 << " = N2/4\n";
    return false;
  }
  if (Bn >= N3/4)
  {
    cout << errorStr;
    cout << "  BsplineOrder = " << Bn << " >= " << N3/4 << " = N3/4\n";
    return false;
  }

  //check B-spline order is not bigger than the maximum B-spline order
  if (Bn > (long int) NPME_Bspline_MaxOrder)
  {
    cout << errorStr;
    cout << "  BsplineOrder = " << Bn << " > " << NPME_Bspline_MaxOrder;
    cout << " = NPME_Bspline_MaxOrder\n";
    return false;
  }

  //check the FFT block sizes are smaller than the +/- interval of interest
  if (n1 > N1/4)
  {
    cout << errorStr;
    cout << "  block size n1 = " << n1 << " > " << N1/4 << " = N1/4\n";
    return false;
  }
  if (n2 > N2/4)
  {
    cout << errorStr;
    cout << "  block size n2 = " << n2 << " > " << N2/4 << " = N2/4\n";
    return false;
  }
  if (n3 > N3/4)
  {
    cout << errorStr;
    cout << "  block size n3 = " << n3 << " > " << N3/4 << " = N3/4\n";
    return false;
  }

  //check block sizes (n1, n2, n3) are multiples of 1/2 FFT (N1/2, N2/2, N3/2)
  if (N1%(2*n1) != 0)
  {
    cout << errorStr;
    cout << "  N1 = " << N1 << " n1 = " << n1;
    cout << " N1 must be a multiple of 2*n1\n";
    return false;
  }
  if (N2%(2*n2) != 0)
  {
    cout << errorStr;
    cout << "  N2 = " << N2 << " n2 = " << n2;
    cout << " N2 must be a multiple of 2*n2\n";
    return false;
  }
  if (N3%(2*n3) != 0)
  {
    cout << errorStr;
    cout << "  N3 = " << N3 << " n3 = " << n3;
    cout << " N3 must be a multiple of 2*n3\n";
    return false;
  }


  return true;
}

double NPME_CalcMinX_Compact (double X0, double del, 
  long int BnOrder, long int N)
{
  const double a1 = 1.0 - 2.0*(BnOrder+1)/N;
  const double a2 = 1.0 - 4.0*(BnOrder+1)/N;

  double X = X0/a1 - del/2.0*a2/a1;
  if (X0 > X)
    X = X0;
  return X;
}

void NPME_CalcBoxSize (const size_t nCharge, const double *coord, 
  double& Xmax, double& Ymax, double &Zmax,
  double& Xmin, double& Ymin, double &Zmin)
{
  Xmax = -1.0E12;
  Ymax = -1.0E12;
  Zmax = -1.0E12;

  Xmin = 1.0E12;
  Ymin = 1.0E12;
  Zmin = 1.0E12;

  for (size_t i = 0; i < nCharge; i++)
  {
    double x = coord[3*i  ];
    double y = coord[3*i+1];
    double z = coord[3*i+2];

    if (Xmax < x) Xmax = x;
    if (Ymax < y) Ymax = y;
    if (Zmax < z) Zmax = z;

    if (Xmin > x) Xmin = x;
    if (Ymin > y) Ymin = y;
    if (Zmin > z) Zmin = z;
  }
}

void NPME_CalcBoxDimensionCenter (const size_t nCharge, const double *coord,
  double& X0, double& Y0, double &Z0, double R0[3])
//input:  coord[3*nCharge]
//output: X0, Y0, Z0 = box dimensions
//        R0[3]      = center of box
{
  //determine box size
  double Xmax, Ymax, Zmax;
  double Xmin, Ymin, Zmin;
  NPME_CalcBoxSize (nCharge, coord, 
    Xmax, Ymax, Zmax,
    Xmin, Ymin, Zmin);

  ///physical volume dimensions
  X0 = Xmax - Xmin;
  Y0 = Ymax - Ymin;
  Z0 = Zmax - Zmin;

  R0[0] = 0.5*(Xmax+Xmin);
  R0[1] = 0.5*(Ymax+Ymin);
  R0[2] = 0.5*(Zmax+Zmin);
}
  


bool NPME_CheckCoordInsideBox (const size_t nCharge, const double *coord,
  const double X0, const double Y0, const double Z0, const double R0[3])
{
  const double boxSize[3] = {X0, Y0, Z0};

  for (size_t i = 0; i < nCharge; i++)
  {
    const double *r = &coord[3*i];
    for (size_t p = 0; p < 3; p++)
    {
      if (fabs(r[p] - R0[p]) > boxSize[p]/2 + 1.0E-12)
      {
        double x[3] = {r[0] - R0[0], r[1] - R0[1], r[2] - R0[2]};
        using std::cout;
        using std::endl;
        cout << "Error in NPME_CheckCoordInsideBox for coord " << i << endl;
        NPME_PrintVec3 (cout, "r      ", r);
        NPME_PrintVec3 (cout, "R0     ", R0);
        NPME_PrintVec3 (cout, "r - R0 ", x);
        return false;
      }
    }
  }

  return true;
}


void NPME_TranslateBox (const size_t nCharge, double *coord, 
  double& X0, double& Y0, double &Z0,
  double& X,  double& Y,  double &Z,
  const long int N1, const long int N2, const long int N3,
  const double del1, const double del2, const double del3,
  const long int BnOrder)
//X0 = Xmax - Xmin (X physical dimension)
//Y0 = Ymax - Ymin (Y physical dimension)
//Z0 = Zmax - Zmin (Z physical dimension)
//X >= X0 (symmetric interval [-X/2, X/2])
//Y >= Y0 (symmetric interval [-Y/2, Y/2])
//Z >= Z0 (symmetric interval [-Z/2, Z/2])
//coord are translated with new max/min XYZ 
//s.t. X/2 = Xmax, Y/2 = Ymax, Z/2 = Zmax
{
  //find physical box size
  double Xmax, Xmin;
  double Ymax, Ymin;
  double Zmax, Zmin;

  NPME_CalcBoxSize (nCharge, coord, 
    Xmax, Ymax, Zmax,
    Xmin, Ymin, Zmin);

  X0 = Xmax - Xmin;
  Y0 = Ymax - Ymin;
  Z0 = Zmax - Zmin;

  X = NPME_CalcMinX_Compact (X0, del1, BnOrder, N1);
  Y = NPME_CalcMinX_Compact (Y0, del2, BnOrder, N2);
  Z = NPME_CalcMinX_Compact (Z0, del3, BnOrder, N3);

  const double trans[3] = {X/2 - Xmax, Y/2 - Ymax, Z/2 - Zmax};

  for (size_t i = 0; i < nCharge; i++)
  {
    double *r = &coord[3*i];
    r[0]     += trans[0];
    r[1]     += trans[1];
    r[2]     += trans[2];
  }
}




void NPME_RecSumInterface_GetPMECorrBox (
        double& X,         double& Y,         double& Z,        double  R[3],
  const double  X0,  const double  Y0,  const double  Z0, const double R0[3],
  const long int N1, const long int N2, const long int N3,
  const double del1, const double del2, const double del3,  
  const long int BnOrder)
//input:  X0, Y0, Z0      = point charge box size
//        R0[3]           = center of point charge box
//        N1 N2 N3        = FFT sizes
//        del1 del2 del3  = Fourier extension smoothing length
//        BnOrder         = B-spline order
{
  //pme corrected volume dimensions
  X  = NPME_CalcMinX_Compact (X0, del1, BnOrder, N1);
  Y  = NPME_CalcMinX_Compact (Y0, del2, BnOrder, N2);
  Z  = NPME_CalcMinX_Compact (Z0, del3, BnOrder, N3);

  R[0] = R0[0] + 0.5*(X0 - X);
  R[1] = R0[1] + 0.5*(Y0 - Y);
  R[2] = R0[2] + 0.5*(Z0 - Z);
}

void NPME_RecSumInterface_GetPMECorrBox (
        double& X,         double& Y,         double& Z,        double  R[3],
  const double  X0,  const double  Y0,  const double  Z0, const double R0[3],
  const long int N1, const long int N2, const long int N3,
  const double del,  const long int BnOrder)
//input:  X0, Y0, Z0  = point charge box size
//        R0[3]       = center of point charge box
//        N1 N2 N3    = FFT sizes
//        del         = Fourier extension smoothing length
//        BnOrder     = B-spline order
{
  //pme corrected volume dimensions
  X  = NPME_CalcMinX_Compact (X0, del, BnOrder, N1);
  Y  = NPME_CalcMinX_Compact (Y0, del, BnOrder, N2);
  Z  = NPME_CalcMinX_Compact (Z0, del, BnOrder, N3);

  R[0] = R0[0] + 0.5*(X0 - X);
  R[1] = R0[1] + 0.5*(Y0 - Y);
  R[2] = R0[2] + 0.5*(Z0 - Z);
}

double NPME_CalcMinX0_Compact (double X, double del, 
  long int BnOrder, long int N)
//X = max(X0/a1 - del/2.0*a2/a1, X0)
//if (X < X0)
//  X = max(X0/a1 - del/2.0*a2/a1, X0)
//  (X + del/2.0*a2/a1)*a1 = X0
//else
//  X0 = X
{
  const double a1 = 1.0 - 2.0*(BnOrder+1)/N;
  const double a2 = 1.0 - 4.0*(BnOrder+1)/N;

  double X0 = (X + del/2.0*a2/a1)*a1;
  if (X0 > X)
    X0 = X;
  return X0;
}
void NPME_RecSumInterface_GetPhysicalVolume (
        double& X0,  double& Y0,       double& Z0,      double  R0[3],
  const double  X,   const double  Y,  const double  Z, const double R[3],
  const long int N1, const long int N2, const long int N3,
  const double del1, const double del2, const double del3,  
  const long int BnOrder)
//input:  X, Y, Z     = pme corrected volume
//        R[3]        = center of pme corrected volume
//        N1 N2 N3    = FFT sizes
//        del         = Fourier extension smoothing length
//        BnOrder     = B-spline order
//output: X0, Y0, Z0  = physical volume
//        R0[3]       = center of physical volume
{
  //pme corrected volume dimensions
  X0  = NPME_CalcMinX0_Compact (X, del1, BnOrder, N1);
  Y0  = NPME_CalcMinX0_Compact (Y, del2, BnOrder, N2);
  Z0  = NPME_CalcMinX0_Compact (Z, del3, BnOrder, N3);

  R0[0] = R[0] + 0.5*(X - X0);
  R0[1] = R[1] + 0.5*(Y - Y0);
  R0[2] = R[2] + 0.5*(Z - Z0);
}



void NPME_Compact2FullQ (_Complex double *Qf, const _Complex double *Qc,
  const long int N1, const long int N2, const long int N3)
//Zero pads compact Q to full Q
//Qc[N1*N2*N3/8] -> Qf[N1*N2*N3]
{
  memset(Qf, 0, N1*N2*N3*sizeof(_Complex double));

  for (long int m1 = -N1/4; m1 <= N1/4-1; m1++)
  for (long int m2 = -N2/4; m2 <= N2/4-1; m2++)
  for (long int m3 = -N3/4; m3 <= N3/4-1; m3++)
  {
    long int fIndex, cIndex;
    {
      const long int p1 = NPME_DFT2Array_Index (m1, N1);
      const long int p2 = NPME_DFT2Array_Index (m2, N2);
      const long int p3 = NPME_DFT2Array_Index (m3, N3);
      fIndex            = NPME_ind3D (p1, p2, p3, N2, N3);
    }

    {
      const long int p1 = NPME_DFT2Array_Index (m1, N1/2);
      const long int p2 = NPME_DFT2Array_Index (m2, N2/2);
      const long int p3 = NPME_DFT2Array_Index (m3, N3/2);
      cIndex            = NPME_ind3D (p1, p2, p3, N2/2, N3/2);
    }

    Qf[fIndex] = Qc[cIndex];
  }
}

void NPME_Full2CompactQ (double& maxError, _Complex double *Qc, 
  const _Complex double *Qf,
  const long int N1, const long int N2, const long int N3)
//truncates full Q to compact Q
//Qf[N1*N2*N3] -> Qc[N1*N2*N3/8]
{
  for (long int m1 = -N1/4; m1 <= N1/4-1; m1++)
  for (long int m2 = -N2/4; m2 <= N2/4-1; m2++)
  for (long int m3 = -N3/4; m3 <= N3/4-1; m3++)
  {
    long int fIndex, cIndex;
    {
      const long int p1 = NPME_DFT2Array_Index (m1, N1);
      const long int p2 = NPME_DFT2Array_Index (m2, N2);
      const long int p3 = NPME_DFT2Array_Index (m3, N3);
      fIndex            = NPME_ind3D (p1, p2, p3, N2, N3);
    }

    {
      const long int p1 = NPME_DFT2Array_Index (m1, N1/2);
      const long int p2 = NPME_DFT2Array_Index (m2, N2/2);
      const long int p3 = NPME_DFT2Array_Index (m3, N3/2);
      cIndex            = NPME_ind3D (p1, p2, p3, N2/2, N3/2);
    }

    Qc[cIndex] = Qf[fIndex];
  }

  maxError = 0.0;
  for (long int p1 = 0; p1 < N1; p1++)
  for (long int p2 = 0; p2 < N2; p2++)
  for (long int p3 = 0; p3 < N3; p3++)
  {
    const long int m1     = NPME_Array2DFT_Index (p1, N1);
    const long int m2     = NPME_Array2DFT_Index (p2, N2);
    const long int m3     = NPME_Array2DFT_Index (p3, N3);

    const long int fIndex = NPME_ind3D (p1, p2, p3, N2, N3);

    bool cond1 = 0;
    bool cond2 = 0;
    bool cond3 = 0;
    if ( (m1 <= -N1/4-1) || (m1 >= N1/4) )  cond1 = 1;
    if ( (m2 <= -N2/4-1) || (m2 >= N2/4) )  cond2 = 1;
    if ( (m3 <= -N3/4-1) || (m3 >= N3/4) )  cond3 = 1;

    if (cond1 && cond2 && cond3)
    {
      double error = cabs(Qf[fIndex]);
      if (maxError < error)
        maxError = error;
    }
  }
}




}//end namespace NPME_Library



