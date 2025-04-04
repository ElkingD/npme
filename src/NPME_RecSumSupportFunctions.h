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

#ifndef NPME_REC_SUM_SUPPORT_FUNCTIONS_H
#define NPME_REC_SUM_SUPPORT_FUNCTIONS_H

#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>

namespace NPME_Library
{
bool NPME_CheckFFTParm (long int Bn,
  long int N1, long int N2, long int N3,
  long int n1, long int n2, long int n3);

double NPME_CalcMinX_Compact (double X0, double del, 
  long int BnOrder, long int N);
//input:  X0, del, BnOrder, N = FFT size
//output: X >= X0, s.t. Qpme_real(p1, p2, p3) is non-zero for
//        -N1/4 <= p1 <= N1/4-1
//        -N2/4 <= p2 <= N2/4-1
//        -N3/4 <= p3 <= N3/4-1
//        does x component, but y, z components treated the same way


void NPME_CalcBoxSize (const size_t nCharge, const double *coord, 
  double& Xmax, double& Ymax, double &Zmax,
  double& Xmin, double& Ymin, double &Zmin);

void NPME_CalcBoxDimensionCenter (const size_t nCharge, const double *coord,
  double& X0, double& Y0, double &Z0, double R0[3]);
//input:  coord[3*nCharge]
//output: X0, Y0, Z0 = box dimensions
//        R0[3]      = center of box

bool NPME_CheckCoordInsideBox (const size_t nCharge, const double *coord,
  const double X0, const double Y0, const double Z0, const double R0[3]);

void NPME_TranslateBox (const size_t nCharge, double *coord, 
  double& X0, double& Y0, double &Z0,
  double& X,  double& Y,  double &Z,
  const long int N1, const long int N2, const long int N3,
  const double del1, const double del2, const double del3,
  const long int BnOrder);
//X0 = Xmax - Xmin (X physical dimension)
//Y0 = Ymax - Ymin (Y physical dimension)
//Z0 = Zmax - Zmin (Z physical dimension)
//X >= X0 (symmetric interval [-X/2, X/2])
//Y >= Y0 (symmetric interval [-Y/2, Y/2])
//Z >= Z0 (symmetric interval [-Z/2, Z/2])
//coord are translated with new max/min XYZ 
//s.t. X/2 = Xmax, Y/2 = Ymax, Z/2 = Zmax

void NPME_RecSumInterface_GetPMECorrBox (
        double& X,         double& Y,         double& Z,        double  R[3],
  const double  X0,  const double  Y0,  const double  Z0, const double R0[3],
  const long int N1, const long int N2, const long int N3,
  const double del,  const long int BnOrder);
//input:  X0, Y0, Z0  = point charge box size
//        R0[3]       = center of point charge box
//        N1 N2 N3    = FFT sizes
//        del         = Fourier extension smoothing length
//        BnOrder     = B-spline order

void NPME_RecSumInterface_GetPMECorrBox (
        double& X,         double& Y,         double& Z,        double  R[3],
  const double  X0,  const double  Y0,  const double  Z0, const double R0[3],
  const long int N1, const long int N2, const long int N3,
  const double del1, const double del2, const double del3,  
  const long int BnOrder);
//input:  X0, Y0, Z0      = point charge box size
//        R0[3]           = center of point charge box
//        N1 N2 N3        = FFT sizes
//        del1 del2 del3  = Fourier extension smoothing length
//        BnOrder         = B-spline order

void NPME_RecSumInterface_GetPhysicalVolume (
        double& X0,  double& Y0,       double& Z0,      double  R0[3],
  const double  X,   const double  Y,  const double  Z, const double R[3],
  const long int N1, const long int N2, const long int N3,
  const double del1, const double del2, const double del3,  
  const long int BnOrder);
//input:  X, Y, Z     = pme corrected volume
//        R[3]        = center of pme corrected volume
//        N1 N2 N3    = FFT sizes
//        del         = Fourier extension smoothing length
//        BnOrder     = B-spline order
//output: X0, Y0, Z0  = physical volume
//        R0[3]       = center of physical volume


void NPME_Compact2FullQ (_Complex double *Qf, const _Complex double *Qc,
  const long int N1, const long int N2, const long int N3);
//Zero pads compact Q to full Q
//Qc[N1*N2*N3/8] -> Qf[N1*N2*N3]

void NPME_Full2CompactQ (double& maxError, _Complex double *Qc, 
  const _Complex double *Qf,
  const long int N1, const long int N2, const long int N3);
//truncates full Q to compact Q
//Qf[N1*N2*N3] -> Qc[N1*N2*N3/8]


//******************************************************************************
//******************************************************************************
//******************************Index Functions*********************************
//******************************************************************************
//******************************************************************************

inline long int NPME_GetQpme_DFT_StartIndex (double wX)
{
  if (wX < 0)   return (long int) (-wX) + 1;
  else          return (long int) (-wX);
}
inline bool NPME_CheckQpme_DFT_StartIndex (long int mStart1, long int N1,
  long int BnOrder)
{
  if (mStart1 < -N1/4)
  {
    std::cout << "Error in NPME_CheckQpme_DFT_StartIndex\n";
    std::cout << "  mStart1 = " << mStart1 << std::endl;
    std::cout << "  N1      = " << N1      << std::endl;
    std::cout << "  BnOrder = " << BnOrder << std::endl;
    std::cout << "Error: mStart1 < -N1/4\n";

    return false;
  }
  if (mStart1 + BnOrder - 1 > N1/4 - 1)
  {
    std::cout << "Error in NPME_CheckQpme_DFT_StartIndex\n";
    std::cout << "  mStart1 = " << mStart1 << std::endl;
    std::cout << "  N1      = " << N1      << std::endl;
    std::cout << "  BnOrder = " << BnOrder << std::endl;
    std::cout << "Error: mStart1 + BnOrder - 1 > N1/4 - 1\n";
    return false;
  }
  return true;
}

inline long int NPME_Array2DFT_Index (const long int kIndex, const long int M)
//kIndex    = 0, 1, 2, .. M-1
//returns k = -M/2, -M/2+1, .. 0, 1, .. M/2-1
{
  if (kIndex >= M/2)
    return kIndex - M;
  else
    return kIndex;
}
inline long int NPME_DFT2Array_Index (const long int k, const long int M)
//intput: k   index = -N/2, -N/2+1, .. -1,    0,   1,     .. N/2-1
//output: DFT index =  N/2,  N/2+1, .. N-1,   0,   1,     .. N/2-1
{
  if (k < 0)
    return k + M;
  else
    return k;
}






}//end namespace NPME_Library


#endif // NPME_REC_SUM_SUPPORT_FUNCTIONS_H



