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

#ifndef NPME_SUPPORT_FUNCTIONS_H
#define NPME_SUPPORT_FUNCTIONS_H

#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream> 

#include "Constant.h"

namespace NPME_Library
{
int NPME_DetermineVecOptionFromCompileFlag ();

double NPME_GetTime ();

double NPME_EwaldSplitOrig_Rdir2Beta (const double Rdir, const double tol);

inline void NPME_PrintVec3 (std::ostream& os, const char *str, 
  const double A[3])
{
  os.precision(6);
  os.precision(6);
  os << str << " = ";
  os << A[0] << " ";
  os << A[1] << " ";
  os << A[2] << "\n";
}

//******************************************************************************
//******************************************************************************
//******************************Index Functions*********************************
//******************************************************************************
//******************************************************************************



inline void NPME_ind2D_2_n1_n2 (const long int index, const long int Nj, 
  long int& i, long int& j)
{
  j = index%Nj;
  i = (index - j)/Nj;
}

inline long int NPME_ind3D (const long int i, const long int j, 
  const long int k, const long int Nj, const long int Nk)
{
  return i*Nj*Nk+j*Nk+k;
}
inline void NPME_ind3D_2_n1_n2_n3 (const long int index, const long int Nj, 
  const long int Nk, long int& i, long int& j, long int& k)
{
  const long int Nj_Nk  = Nj*Nk;
  const long int r1     = index%(Nj_Nk);   // r1 = j*Nk+k
  k                     = r1%Nk;
  j                     = (r1 - k)/Nk;
  i                     = (index - r1)/Nj_Nk;
}
inline void NPME_ind2D_2_n1_n2 (const size_t index, const size_t Nj, 
  size_t& i, size_t& j)
{
  j = index%Nj;
  i = (index - j)/Nj;
}

inline size_t NPME_ind3D (const size_t i, const size_t j, 
  const size_t k, const size_t Nj, const size_t Nk)
{
  return i*Nj*Nk+j*Nk+k;
}
inline void NPME_ind3D_2_n1_n2_n3 (const size_t index, const size_t Nj, 
  const size_t Nk, size_t& i, size_t& j, size_t& k)
{
  const size_t Nj_Nk  = Nj*Nk;
  const size_t r1     = index%(Nj_Nk);   // r1 = j*Nk+k
  k                   = r1%Nk;
  j                   = (r1 - k)/Nk;
  i                   = (index - r1)/Nj_Nk;
}
inline void NPME_ind2D_symmetric2_index_2_pq (size_t& p, size_t& q, const size_t k)
//finds (p, q) p >= q given its index k
{
  const double pf = (sqrt(1.0+8.0*k) - 1.0)/2.0;
  p = (size_t) pf;
  q = k - (p*(p+1))/2;
}

inline double NPME_GetDoubleRand (const double a, const double b)
//Gets a random number between a and b
{
    return a + (b - a)* ( (double) rand()/ (double)RAND_MAX );
}
inline _Complex double NPME_GetDoubleRandComplex (const double a, 
                                                  const double b)
//Gets a random number between a and b
{
  return NPME_GetDoubleRand (a, b) + I*NPME_GetDoubleRand (a, b);
}
inline int NPME_GetIntRand (const int a, const int b)
{
  if (a > b)
    return NPME_GetIntRand (b, a);

  const int c = rand()%(b-a+1);
  return a+c;
}
inline void NPME_GetDoubleRandVec3 (double v[3], const double a, const double b)
//Gets a random number between a and b
{
  v[0] = NPME_GetDoubleRand (a, b);
  v[1] = NPME_GetDoubleRand (a, b);
  v[2] = NPME_GetDoubleRand (a, b);
}

//******************************************************************************
//******************************************************************************
//******************************Misc Functions**********************************
//******************************************************************************
//******************************************************************************

template <class T>
T NPME_Min (const T a, const T b)
{
  if (a < b)  return a;
  else        return b;
}
template <class T>
T NPME_Min (const T a, const T b, const T c)
{
  return NPME_Min (NPME_Min (a, b), c);
}
template <class T>
T NPME_Max (const T a, const T b)
{
  if (a > b)  return a;
  else        return b;
}
template <class T>
T NPME_Max (const T a, const T b, const T c)
{
  return NPME_Max (NPME_Max (a, b), c);
}



inline double NPME_CalcDistance2 (const double *r1, const double *r2)
{
  const double x = r1[0] - r2[0];
  const double y = r1[1] - r2[1];
  const double z = r1[2] - r2[2];

  return x*x + y*y + z*z;
}
inline double NPME_CalcDistance (const double *r1, const double *r2)
{
  return sqrt( NPME_CalcDistance2 (r1, r2) );
}

inline double NPME_DotProd3 (const double *a, const double *b)
{
  return a[ 0]*b[ 0] + a[ 1]*b[ 1] + a[ 2]*b[ 2];
}
_Complex double NPME_N_DotProd (long int N, 
  const _Complex double *A, const _Complex double *B);





void NPME_FindFFTSizeBlockSize (long int& N, long int& n, 
  const long int N_ideal, const long int n_ideal);

void NPME_GenerateRandomCoord (size_t nCharge, double *coord,
  const double X0, const double Y0, const double Z0, const double Rc[3],
  size_t seed = 1);
//input:  X0, Y0, Z0 = physical box length dimensions
//        Rc[3]      = box center
//output: coord[3*nCharge] = random coordinates inside box

void NPME_GenerateUniformCoord (size_t nCharge, double *coord,
  const double X0, const double Y0, const double Z0, const double Rc[3]);
//input:  X0, Y0, Z0 = physical box length dimensions
//        Rc[3]      = box center
//output: coord[3*nCharge] = coordinates inside box




void NPME_GetRandomPoints (const size_t nPoint, double *coord, 
  const double X, const double Y, const double Z);

void NPME_CalcRMSD (double& eps_V, double& eps_dVdr, const size_t nCharge,
  const double *V1, const double *V2);
void NPME_CalcRMSD (double& eps_V, double& eps_dVdr, const size_t nCharge,
  const _Complex double *V1, const _Complex double *V2);






void NPME_GeneratePrimeFactorization (int n, std::vector<int>& primeFactor);
//input:  n
//output: primeFactor[] = {a, b, c, .. }
//        such that n = a*b*c*..
int NPME_FindOptimalGridSize (int n);
//returns a number N close to n, which satisfies
//  1) N is a multiple of 4
//  2) N has prime factors 2, 3, 5, 7



double NPME_CalcMinDistance_R0_r (
  const double R0[3], const double lx, const double ly, const double lz);
//calculates minimum distance of |R0 + r| for r[3] = (x,y,z) contained inside 
//a rectangular volume with center at the origin and dimensions (lx,ly,lz)
//min(R0+r)^2 = min(X0+x)^2 + min(Y0+y)^2 + min(Z0+z)^2
//if  (|X0| < lx/2) min(X0+x)^2 = 0
//else              min(X0+x)^2 = (lx/2 - |X0|)^2
//
//in order to apply to 2 cubes using the Fourier extension procedure, consider 
//center R1[3] and dimensions l1x, l1y, l1z
//center R2[3] and dimensions l2x, l2y, l2z
//R0 = R1 - R2
//lx = l1x+l2x + 2*delta
//ly = l1y+l2y + 2*delta
//lz = l1z+l2z + 2*delta
double NPME_CalcMinDistanceRectVolumes (
  const double R1[3], const double lx1, const double ly1, const double lz1,
  const double R2[3], const double lx2, const double ly2, const double lz2);
//calculates the minimum distance between 2 rectangular volumes with centers
//R1[3] and R2[3] and dimensions (lx1, ly1, lz1) and (lx2, ly2, lz2)

void NPME_ZeroArray (const long int N, const int nProc, 
  double *A, const long int blockSize);
void NPME_ZeroArray (const long int N, const int nProc, 
  _Complex double *A, const long int blockSize);

void NPME_TransposeSimple (const size_t M, const size_t N, 
  double *At, const double *A);
void NPME_TransposeSimple (const size_t M, const size_t N, 
  _Complex double *At, const _Complex double *A);
//A is MxN and At is NxM
//for M ~ N,  the MKL function is ~2x faster
//for M >> N, this function is over 10x faster


double NPME_CalcMinDistance_R0_r (
  const double R0[3], const double lx, const double ly, const double lz);
//calculates minimum distance of |R0 + r| for r[3] = (x,y,z) contained inside 
//a rectangular volume with center at the origin and dimensions (lx,ly,lz)
//min(R0+r)^2 = min(X0+x)^2 + min(Y0+y)^2 + min(Z0+z)^2
//if  (|X0| < lx/2) min(X0+x)^2 = 0
//else              min(X0+x)^2 = (lx/2 - |X0|)^2
//
//in order to apply to 2 cubes using the Fourier extension procedure, consider 
//center R1[3] and dimensions l1x, l1y, l1z
//center R2[3] and dimensions l2x, l2y, l2z
//R0 = R1 - R2
//lx = l1x+l2x + 2*delta
//ly = l1y+l2y + 2*delta
//lz = l1z+l2z + 2*delta
double NPME_CalcMinDistanceRectVolumes (
  const double R1[3], const double lx1, const double ly1, const double lz1,
  const double R2[3], const double lx2, const double ly2, const double lz2);
//calculates the minimum distance between 2 rectangular volumes with centers
//R1[3] and R2[3] and dimensions (lx1, ly1, lz1) and (lx2, ly2, lz2)

//******************************************************************************
//******************************************************************************
//*********************Error Checking Functions*********************************
//******************************************************************************
//******************************************************************************
void NPME_CompareArrays (double& error, const size_t N, 
  const char *desc1, const double *A1, 
  const char *desc2, const double *A2, bool PRINT, std::ostream& os);
void NPME_CompareArrays (double& error, const size_t N, 
  const char *desc1, const _Complex double *A1, 
  const char *desc2, const _Complex double *A2, bool PRINT, std::ostream& os);

void NPME_CompareV1Arrays (double& error, const size_t nCharge, 
  const char *desc1, const double *V1, 
  const char *desc2, const double *V2, bool PRINT, std::ostream& os);
void NPME_CompareV1Arrays (double& error, const size_t nCharge, 
  const char *desc1, const _Complex double *V1, 
  const char *desc2, const _Complex double *V2, bool PRINT, std::ostream& os);
void NPME_CompareV1Arrays (double& errorV, double& errordVdr,
  const size_t nCharge, 
  const char *desc1, const double *V1, 
  const char *desc2, const double *V2, bool PRINT, std::ostream& os);
void NPME_CompareV1Arrays (double& errorV, double& errordVdr,
  const size_t nCharge, 
  const char *desc1, const _Complex double *V1, 
  const char *desc2, const _Complex double *V2, bool PRINT, std::ostream& os);
//compares V1[nCharge][4]
//with     V2[nCharge][4]


void NPME_CalcV1AvgVecMag (double& V_mag, double& dVdr_mag, 
  const size_t nCharge, const double *V1);
void NPME_CalcV1AvgVecMag (double& V_mag, double& dVdr_mag, 
  const size_t nCharge, const _Complex double *V1);



}//end namespace NPME_Library


#endif // NPME_SUPPORT_FUNCTIONS_H



