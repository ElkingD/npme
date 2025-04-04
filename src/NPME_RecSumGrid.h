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

#ifndef NPME_REC_SUM_GRID_H
#define NPME_REC_SUM_GRID_H


#include "NPME_KernelFunction.h"

namespace NPME_Library
{


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
  bool PRINT = 0, std::ostream& os = std::cout);
bool NPME_RecSumGrid_Full_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncComplex& func, int nProc,
  bool PRINT = 0, std::ostream& os = std::cout);


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
  bool PRINT = 0, std::ostream& os = std::cout);
bool NPME_RecSumGrid_CompactReal_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncComplex& func, int nProc,
  bool PRINT = 0, std::ostream& os = std::cout);
//calculates compact real space grid for spherically symmetric function


bool NPME_RecSumGrid_CompactFourier_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncReal& func, int nProc,
  _Complex double *fTmp, bool PRINT = 0, std::ostream& os = std::cout);
bool NPME_RecSumGrid_CompactFourier_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncComplex& func, int nProc,
  _Complex double *fTmp, bool PRINT = 0, std::ostream& os = std::cout);
//f   [M1*M2*M3]
//fTmp[M1*M2*M3]

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
  int nProc, bool PRINT = 0, std::ostream& os = std::cout);
bool NPME_RecSumGrid_FullTrans_Func (_Complex double *f,
  const long int N1, const double a1, const double del1,
  const long int N2, const double a2, const double del2,
  const long int N3, const double a3, const double del3,
  const double X, const double Y, const double Z,
  const NPME_Library::NPME_KfuncComplex& func, const double R0[3], 
  int nProc, bool PRINT = 0, std::ostream& os = std::cout);
//calculates f(r) = func(|r+R0|) on a DFT grid 
//for r = (x, y, z) given by
//-X <= x <= X
//-Y <= y <= Y
//-Z <= z <= Z


//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
//****************************Interpolation Function***************************
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
void NPME_RecSumGrid_Full_Interpolate (const size_t nRandom, 
  _Complex double *f, const double *r,
  const long int N1,  const long int N2,  const long int N3,
  const double L1,    const double L2,    const double L3,
  const _Complex double *Fk, int nProc);
//input  r[3*nRandom], Fk[N1*N2*N3] = un-normalized FFT of grid
//output f[nRandom]

void NPME_RecSumGrid_CompactGrid_Interpolate (const size_t nRandom, 
  _Complex double *f, const double *r,
  const long int N1,  const long int N2,  const long int N3,
  const double L1,    const double L2,    const double L3,
  const _Complex double *Fk, int nProc);
//input  r[3*nRandom], Fk[N1*N2*N3] = un-normalized FFT of grid
//output f[nRandom]

//*****************************************************************************
//*****************************************************************************
//****************************Index/Array Functions****************************
//*****************************************************************************
//*****************************************************************************

void NPME_Set_xArray (double *x1, const long int N, const double L);
long int NPME_ArrayIndex_2_mIndex (long int arrayIndex, const long int N);
//input:  arrayIndex = (0, 1, .. N-1)
//output: mIndex     = (0, 1, 2, .. N/2-1, -N/2, -N/2+1, .. -1}

long int NPME_mIndex_2_ArrayIndex (long int mIndex, const long int N);
//input:  mIndex     = (0, 1, 2, .. N/2-1, -N/2, -N/2+1, .. -1}
//output: arrayIndex = (0, 1, .. N-1)



void NPME_CompactGrid_Set_xArray (double *x1, const long int N, const double L);
//L  = 2*Xmax
//dx = L/N
//x1[N/2+1] = (0, dx, 2*dx, .. N/2*dx)

long int NPME_CompactGrid_mIndex_2_arrayIndex (const long int m, 
  const long int N);



//*****************************************************************************
//*****************************************************************************
//****************************Smooth Functions*********************************
//*****************************************************************************
//*****************************************************************************
double NPME_Hramp (const double x, const double a);
double NPME_Tsmooth (const double x, const double X, 
  const double Xmax, const double a);
void NPME_CalcNonOverlapSmoothFunc (double *T1, 
  const long int N, const double *x1,
  const double X, const double Xmax, const double a);
void NPME_CalcOverlapSmoothFunc (double *T1, double *T2, 
  double *x2, const long int N, const double *x1,
  const double X, const double Xmax, const double a);
void NPME_CalcOverlapSmoothFunc (_Complex double *T1, _Complex double *T2, 
  double *x2, const long int N, const double *x1,
  const double X, const double Xmax, const double a);

double NPME_CalcShortRangeSmoothFunc (const double r,
  const double r0, const double del_r, const double a_r);



}//end namespace NPME_Library


#endif // NPME_REC_SUM_GRID_H


