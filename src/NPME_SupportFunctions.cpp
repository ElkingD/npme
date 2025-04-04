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
#include <sys/time.h>
#include <iostream> 
#include <vector>





#include "NPME_Constant.h"
#include "NPME_SupportFunctions.h"
#include "NPME_ExtLibrary.h"
#include "NPME_MathFunctions.h"
//#include "NPME_ReadPrint.h"
//#include "NPME_Bspline.h"




namespace NPME_Library
{
//******************************************************************************
//******************************************************************************
//*******************vecOption = 0,1,2 from compile flags***********************
//******************************************************************************
//******************************************************************************
int NPME_DetermineVecOptionFromCompileFlag ()
{
  int vecOption = 0;  //no vectorization

  #if NPME_USE_AVX
    vecOption = 1;
  #endif

  #if NPME_USE_AVX_512
    vecOption = 2;
  #endif
  
  return vecOption;
}



//******************************************************************************
//******************************************************************************
//**************************Measure Time Functions******************************
//******************************************************************************
//******************************************************************************

double NPME_GetTime ()
{
  double tseconds = 0.0;
  struct timeval time1;
  gettimeofday (&time1, NULL);
  tseconds = (double) (time1.tv_sec + time1.tv_usec*1.0E-6);
  return tseconds;
}








//******************************************************************************
//******************************************************************************
//**************************Error Checking Functions****************************
//******************************************************************************
//******************************************************************************

void NPME_CalcV1AvgVecMag (double& V_mag, double& dVdr_mag, 
  const size_t nCharge, const double *V1)
{
  V_mag     = 0;
  dVdr_mag  = 0;
  for (size_t i = 0; i < nCharge; i++)
  {
    V_mag += fabs(V1[4*i]);
    for (size_t p = 0; p < 3; p++)
      dVdr_mag += fabs(V1[4*i+p]);
  }

  V_mag /= nCharge;
  dVdr_mag /= (3*nCharge);
}


void NPME_CalcV1AvgVecMag (double& V_mag, double& dVdr_mag, 
  const size_t nCharge, const _Complex double *V1)
{
  V_mag     = 0;
  dVdr_mag  = 0;
  for (size_t i = 0; i < nCharge; i++)
  {
    V_mag += cabs(V1[4*i]);
    for (size_t p = 0; p < 3; p++)
      dVdr_mag += cabs(V1[4*i+p]);
  }

  V_mag /= nCharge;
  dVdr_mag /= (3*nCharge);
}


void NPME_CompareArrays (double& error, const size_t N, 
  const char *desc1, const double *A1, 
  const char *desc2, const double *A2, bool PRINT, std::ostream& os)
{
  error = 0;
  for (size_t i = 0; i < N; i++)
  {
    double diff = fabs(A1[i] - A2[i]);
    error      += diff*diff;

    if (PRINT)
    {
      char str[500];
      sprintf(str, "%s[%4lu] = %15.12f\n", desc1, i, A1[i]);  os << str;
      sprintf(str, "%s[%4lu] = %15.12f\n", desc2, i, A2[i]);  os << str;
      sprintf(str, "error = %.2le\n\n", diff);                os << str;
    }
  }

  error /= N;
  error = sqrt(error);
}

void NPME_CompareArrays (double& error, const size_t N, 
  const char *desc1, const _Complex double *A1, 
  const char *desc2, const _Complex double *A2, bool PRINT, std::ostream& os)
{
  error = 0;
  for (size_t i = 0; i < N; i++)
  {
    double diff = cabs(A1[i] - A2[i]);
    error      += diff*diff;

    if (PRINT)
    {
      char str[500];
      sprintf(str, "%s[%4lu] = %15.12f + %15.12fi\n", 
        desc1, i, creal(A1[i]), cimag(A1[i]));
      os << str;
      sprintf(str, "%s[%4lu] = %15.12f + %15.12fi\n", 
        desc2, i, creal(A2[i]), cimag(A2[i]));
      os << str;
      sprintf(str, "error = %.2le\n\n", diff);
      os << str;
    }
  }

  error /= N;
  error = sqrt(error);
}




void NPME_CompareV1Arrays (double& errorV, double& errordVdr,
  const size_t nCharge, 
  const char *desc1, const double *V1, 
  const char *desc2, const double *V2, bool PRINT, std::ostream& os)
//compares V1[nCharge][4]
//with     V2[nCharge][4]
{
  errorV    = 0;
  errordVdr = 0.0;

  for (size_t i = 0; i < nCharge; i++)
  {
    double error_i = 0;
    for (size_t p = 0; p < 4; p++)
    {
      double diff = fabs(V1[4*i+p] - V2[4*i+p]);
      if (p == 0)
        errorV += diff*diff;
      else
        errordVdr += diff*diff;
      error_i    += diff*diff;
    }


    if (PRINT)
    {
      char str[500];
      sprintf(str, "%s_V0[%4lu] = %15.12f\n",   desc1, i, V1[4*i  ]); os << str;
      sprintf(str, "%s_V0[%4lu] = %15.12f\n\n", desc2, i, V2[4*i  ]); os << str;

      sprintf(str, "%s_VX[%4lu] = %15.12f\n",   desc1, i, V1[4*i+1]); os << str;
      sprintf(str, "%s_VX[%4lu] = %15.12f\n\n", desc2, i, V2[4*i+1]); os << str;

      sprintf(str, "%s_VY[%4lu] = %15.12f\n",   desc1, i, V1[4*i+2]); os << str;
      sprintf(str, "%s_VY[%4lu] = %15.12f\n\n", desc2, i, V2[4*i+2]); os << str;

      sprintf(str, "%s_VZ[%4lu] = %15.12f\n",   desc1, i, V1[4*i+3]); os << str;
      sprintf(str, "%s_VZ[%4lu] = %15.12f\n\n", desc2, i, V2[4*i+3]); os << str;

      sprintf(str, "error = %.2le\n\n", sqrt(error_i/4)); os << str;
    }
  }

  errorV /= (nCharge);
  errorV = sqrt(errorV);

  errordVdr /= (3*nCharge);
  errordVdr = sqrt(errordVdr);
}

void NPME_CompareV1Arrays (double& error, const size_t nCharge, 
  const char *desc1, const double *V1, 
  const char *desc2, const double *V2, bool PRINT, std::ostream& os)
//compares V1[nCharge][4]
//with     V2[nCharge][4]
{
  double errorV, errordVdr;
  NPME_CompareV1Arrays (errorV, errordVdr, nCharge, 
    desc1, V1, 
    desc2, V2, PRINT, os);
  error = errorV + errordVdr;
}


void NPME_CompareV1Arrays (double& errorV, double& errordVdr,
  const size_t nCharge, 
  const char *desc1, const _Complex double *V1, 
  const char *desc2, const _Complex double *V2, bool PRINT, std::ostream& os)
//compares V1[nCharge][4]
//with     V2[nCharge][4]
{
  errorV    = 0;
  errordVdr = 0.0;

  for (size_t i = 0; i < nCharge; i++)
  {
    double error_i = 0;
    for (size_t p = 0; p < 4; p++)
    {
      double diff = cabs(V1[4*i+p] - V2[4*i+p]);
      if (p == 0)
        errorV += diff*diff;
      else
        errordVdr += diff*diff;

      error_i    += diff*diff;
    }

    if (PRINT)
    {
      char str[500];
      sprintf(str, "%s_V0[%4lu] = %15.12f + %15.12fi\n", 
        desc1, i, creal(V1[4*i  ]), cimag(V1[4*i  ]));
      os << str;
      sprintf(str, "%s_V0[%4lu] = %15.12f + %15.12fi\n\n", 
        desc2, i, creal(V2[4*i  ]), cimag(V2[4*i  ]));
      os << str;

      sprintf(str, "%s_VX[%4lu] = %15.12f + %15.12fi\n", 
        desc1, i, creal(V1[4*i+1]), cimag(V1[4*i+1]));
      os << str;
      sprintf(str, "%s_VX[%4lu] = %15.12f + %15.12fi\n\n", 
        desc2, i, creal(V2[4*i+1]), cimag(V2[4*i+1]));
      os << str;

      sprintf(str, "%s_VY[%4lu] = %15.12f + %15.12fi\n", 
        desc1, i, creal(V1[4*i+2]), cimag(V1[4*i+2]));
      os << str;
      sprintf(str, "%s_VY[%4lu] = %15.12f + %15.12fi\n\n", 
        desc2, i, creal(V2[4*i+2]), cimag(V2[4*i+2]));
      os << str;

      sprintf(str, "%s_VZ[%4lu] = %15.12f + %15.12fi\n", 
        desc1, i, creal(V1[4*i+3]), cimag(V1[4*i+3]));
      os << str;
      sprintf(str, "%s_VZ[%4lu] = %15.12f + %15.12fi\n\n", 
        desc2, i, creal(V2[4*i+3]), cimag(V2[4*i+3]));
      os << str;

      sprintf(str, "error = %.2le\n\n", sqrt(error_i/4));
      os << str;
    }
  }

  errorV /= (nCharge);
  errorV = sqrt(errorV);

  errordVdr /= (3*nCharge);
  errordVdr = sqrt(errordVdr);
}

void NPME_CompareV1Arrays (double& error, const size_t nCharge, 
  const char *desc1, const _Complex double *V1, 
  const char *desc2, const _Complex double *V2, bool PRINT, std::ostream& os)
//compares V1[nCharge][4]
//with     V2[nCharge][4]
{
  double errorV, errordVdr;
  NPME_CompareV1Arrays (errorV, errordVdr, nCharge, 
    desc1, V1, 
    desc2, V2, PRINT, os);
  error = errorV + errordVdr;
}








//******************************************************************************
//******************************************************************************
//***************************Prime Factor Functions*****************************
//******************************************************************************
//******************************************************************************

bool NPME_IsPrime (int n)
{
  for (int i = 2; i*i <= n; i++)
    if (n%i == 0)
      return false;
  return true;
}

void NPME_GeneratePrimeList (const int N, std::vector<int>& primeList)
{
  primeList.clear();
  for (int i = 2; i <= N; i++)
    if (NPME_IsPrime(i))
      primeList.push_back(i);

}

void NPME_GeneratePrimeFactorization (int n, const int nPrimeList, 
  const int *primeList, std::vector<int>& primeFactor)
//input:  n, primeList[nPrimeList]
//output: primeFactor[] = {a, b, c, .. }
//        such that n = a*b*c*..
{
  const int nStart = n;
  primeFactor.clear();
  size_t nFactorPrev = 0;
  while (n > 1)
  {
    for (int i = 0; i < nPrimeList; i++)
    {
      if (n%primeList[i] == 0)
      {
        primeFactor.push_back(primeList[i]);
        n /= primeList[i];
        break;
      }
    }
  }
}
void NPME_GeneratePrimeFactorization (int n, std::vector<int>& primeFactor)
//input:  n
//output: primeFactor[] = {a, b, c, .. }
//        such that n = a*b*c*..
{
  std::vector<int> primeList;
  NPME_GeneratePrimeList (n, primeList);
  NPME_GeneratePrimeFactorization (n, (int) primeList.size(), 
    &primeList[0], primeFactor);

}

bool NPME_DoesPrimeFactorHaveSmallRadices (const int nFactor, 
  const int *factorList)
//the factors of N = product (factorList[i]) must all be either 2, 3, 5, 7
{
  for (int i = 0; i < nFactor; i++)
  {
    if ( (factorList[i] != 2)  && 
         (factorList[i] != 3)  && 
         (factorList[i] != 5)  && 
         (factorList[i] != 7) )
      return false;
  }
  return true;
}

void NPME_GenerateOptimalFFTGridSize (int Nmax, std::vector<int>& optGridSize)
{
  std::vector<int> primeList;
  NPME_GeneratePrimeList (Nmax, primeList);

  std::vector<int> primeFactor;
  optGridSize.clear();
  
  //optGridSize[] elements are multiples of 4
  for (int n = 4; n <= Nmax; n += 4)
  {
    NPME_GeneratePrimeFactorization (n, (int) primeList.size(), 
      &primeList[0], primeFactor);

    if (NPME_DoesPrimeFactorHaveSmallRadices ( (int) primeFactor.size(), 
      &primeFactor[0]) )
        optGridSize.push_back (n);
  }
}

int NPME_FindOptimalGridSize (int n)
//returns a number N close to n, which satisfies
//  1) N is a multiple of 4
//  2) N has prime factors 2, 3, 5, 7
{
  std::vector<int> optGridSize;
  NPME_GenerateOptimalFFTGridSize (2*n, optGridSize);
  for (size_t i = 0; i < optGridSize.size(); i++)
    if (optGridSize[i] >= n)
      return optGridSize[i];

  std::cout << "Error in NPME_FindOptimalGridSize for n = " << n << std::endl;
  exit(0);
}

void NPME_FindFFTSizeBlockSize (long int& N, long int& n, 
  const long int N_ideal, const long int n_ideal)
{
  //
  N = (long int) NPME_FindOptimalGridSize ( (int) N_ideal);
  long int minDiff  = N;
  long int min_n    = 1;

  for (long int m = 2; m < N; m++)
  {
    if ( (N%(2*m) == 0) && (m <= N/4) )
    {
      long int diff = abs(m - n_ideal);
      if (minDiff > diff)
      {
        minDiff = diff;
        min_n   = m;
      }
    }
  }

  n = min_n;
}











void NPME_GenerateRandomCoord (size_t nCharge, double *coord,
  const double X0, const double Y0, const double Z0, const double Rc[3],
  size_t seed)
//input:  X0, Y0, Z0 = physical box length dimensions
//        Rc[3]      = box center
//output: coord[3*nCharge] = random coordinates inside box
{
  //generate random numbers between 0 and 1
  NPME_RandomNumberArray (3*nCharge, coord, -1.0, 1.0, seed);

  for (size_t i = 0; i < nCharge; i++)
  {
    //scale by dimensions
    double x = 0.5*X0*coord[3*i  ];
    double y = 0.5*Y0*coord[3*i+1];
    double z = 0.5*Z0*coord[3*i+2];

    //translate by center
    coord[3*i  ] = Rc[0] + x;
    coord[3*i+1] = Rc[1] + y;
    coord[3*i+2] = Rc[2] + z;
  }
}

void NPME_GenerateUniformCoord (size_t nCharge, double *coord,
  const double X0, const double Y0, const double Z0, const double Rc[3])
//input:  X0, Y0, Z0 = physical box length dimensions
//        Rc[3]      = box center
//output: coord[3*nCharge] = coordinates inside box
{
  //approx spacing
  double l  = pow(X0*Y0*Z0/nCharge, 1.0/3.0);
  
  size_t NX = (size_t) (X0/l + 0.001);
  size_t NY = (size_t) (Y0/l + 0.001);
  size_t NZ = (size_t) (Z0/l + 0.001);


  if (NX*NY*NZ < nCharge)
  {
    NX++;
    NY++;
    NZ++;
  }

  if (NX*NY*NZ < nCharge)
  {
    std::cout << "Error in NPME_GenerateUniformCoord.\n";
    std::cout << "NX*NY*NZ = " << NX*NY*NZ;
    std::cout << " < " << nCharge << " = nCharge\n";
    exit(0);
  }

  double lx = X0/(NX-1);
  double ly = Y0/(NY-1);
  double lz = Z0/(NZ-1);

  size_t count = 0;
  for (size_t nX = 0; nX < NX; nX++)
  for (size_t nY = 0; nY < NY; nY++)
  for (size_t nZ = 0; nZ < NZ; nZ++)
  {
    double x = lx*nX - X0/2;
    double y = ly*nY - Y0/2;
    double z = lz*nZ - Z0/2;

    //translate by center
    coord[3*count  ] = Rc[0] + x;
    coord[3*count+1] = Rc[1] + y;
    coord[3*count+2] = Rc[2] + z;
      
    count++;
    if (count == nCharge)
      return;
  }
}




_Complex double NPME_N_DotProd (long int N, 
  const _Complex double *A, const _Complex double *B)
{
  _Complex double sum = 0.0;
  for (long int i = 0; i < N; i++)
    sum += A[i]*B[i];
  return sum;
}
double NPME_Distance (const double r1[3], const double r2[3])
{
  const double x = r1[0] - r2[0];
  const double y = r1[1] - r2[1];
  const double z = r1[2] - r2[2];

  return sqrt(x*x + y*y + z*z);
}

void NPME_GetRandomPoints (const size_t nPoint, double *coord, 
  const double X, const double Y, const double Z)
{
  NPME_RandomNumberArray (3*nPoint, coord, -1.0, 1.0);
  for (int i = 0; i < nPoint; i++)
  {
    coord[3*i  ] *= X;
    coord[3*i+1] *= Y;
    coord[3*i+2] *= Z;
  }
}

void NPME_CalcRMSD (double& eps_V, double& eps_dVdr, const size_t nCharge,
  const double *V1, const double *V2)
{
  eps_V = 0;
  eps_dVdr = 0;
  for (size_t i = 0; i < nCharge; i++)
  {
    double diff = 0;
    diff = fabs(V1[4*i  ]-V2[4*i  ]);   eps_V    += diff*diff;
    diff = fabs(V1[4*i+1]-V2[4*i+1]);   eps_dVdr += diff*diff;
    diff = fabs(V1[4*i+2]-V2[4*i+2]);   eps_dVdr += diff*diff;
    diff = fabs(V1[4*i+3]-V2[4*i+3]);   eps_dVdr += diff*diff;
  }
  eps_V    /= nCharge;
  eps_dVdr /= 3*nCharge;

  eps_V    = sqrt(eps_V);
  eps_dVdr = sqrt(eps_dVdr);
}
void NPME_CalcRMSD (double& eps_V, double& eps_dVdr, const size_t nCharge,
  const _Complex double *V1, const _Complex double *V2)
{
  eps_V = 0;
  eps_dVdr = 0;
  for (size_t i = 0; i < nCharge; i++)
  {
    double diff = 0;
    diff = cabs(V1[4*i  ]-V2[4*i  ]);   eps_V    += diff*diff;
    diff = cabs(V1[4*i+1]-V2[4*i+1]);   eps_dVdr += diff*diff;
    diff = cabs(V1[4*i+2]-V2[4*i+2]);   eps_dVdr += diff*diff;
    diff = cabs(V1[4*i+3]-V2[4*i+3]);   eps_dVdr += diff*diff;
  }
  eps_V    /= nCharge;
  eps_dVdr /= 3*nCharge;

  eps_V    = sqrt(eps_V);
  eps_dVdr = sqrt(eps_dVdr);
}







void NPME_ZeroArray (const long int N, const int nProc, 
  double *A, const long int blockSize)
{
  const long int r = N%blockSize;
  const long int n = (N-r)/blockSize;

  long int n1;
  #pragma omp parallel shared(A) private(n1) num_threads(nProc) 
  {
    #pragma omp for schedule(static) nowait
    for (n1 = 0; n1 < n; n1++)
      memset(&A[n1*blockSize], 0, blockSize*sizeof(double));
    #pragma omp single
    {
      memset(&A[n*blockSize], 0, r*sizeof(double));
    }
  }
}

void NPME_ZeroArray (const long int N, const int nProc, 
  _Complex double *A, const long int blockSize)
{
  const long int r = N%blockSize;
  const long int n = (N-r)/blockSize;

  long int n1;
  #pragma omp parallel shared(A) private(n1) num_threads(nProc) 
  {
    #pragma omp for schedule(static) nowait
    for (n1 = 0; n1 < n; n1++)
      memset(&A[n1*blockSize], 0, blockSize*sizeof(_Complex double));
    #pragma omp single
    {
      memset(&A[n*blockSize], 0, r*sizeof(_Complex double));
    }
  }
}

//************************************************
//************************************************
//***********Transpose****************************
//************************************************
//************************************************
void NPME_TransposeSimple (const size_t M, const size_t N, 
  double *At, const double *A)
//A is MxN and At is NxM
//for M ~ N,  the MKL function is ~2x faster
//for M >> N, this function is over 10x faster
{
  for (size_t j = 0; j < N; j++)
    for (size_t i = 0; i < M; i++)
      At[j*M+i] = A[i*N+j];
}
void NPME_TransposeSimple (const size_t M, const size_t N, 
  _Complex double *At, const _Complex double *A)
//A is MxN and At is NxM
//for M ~ N,  the MKL function is ~2x faster
//for M >> N, this function is over 10x faster
{
  for (size_t j = 0; j < N; j++)
    for (size_t i = 0; i < M; i++)
      At[j*M+i] = A[i*N+j];
}


double NPME_CalcMinDistance_R0_r (
  const double R0[3], const double lx, const double ly, const double lz)
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
{
  const double X0_mag = fabs(R0[0]);
  const double Y0_mag = fabs(R0[1]);
  const double Z0_mag = fabs(R0[2]);

  double Mx, My, Mz;
  
  if (X0_mag < lx/2)  Mx = 0;
  else                Mx = (lx/2 - X0_mag)*(lx/2 - X0_mag);

  if (Y0_mag < ly/2)  My = 0;
  else                My = (ly/2 - Y0_mag)*(ly/2 - Y0_mag);

  if (Z0_mag < lz/2)  Mz = 0;
  else                Mz = (lz/2 - Z0_mag)*(lz/2 - Z0_mag);

  return sqrt(Mx + My + Mz);
}


double NPME_CalcMinDistanceRectVolumes (
  const double R1[3], const double lx1, const double ly1, const double lz1,
  const double R2[3], const double lx2, const double ly2, const double lz2)
//calculates the minimum distance between 2 rectangular volumes with centers
//R1[3] and R2[3] and dimensions (lx1, ly1, lz1) and (lx2, ly2, lz2)
{
  const double R0[3] = {R1[0] - R2[0], R1[1] - R2[1], R1[2] - R2[2]};
  return NPME_CalcMinDistance_R0_r (R0, lx1 + lx2, ly1 + ly2, lz1 + lz2);
}

}//end namespace NPME_Library



