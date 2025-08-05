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
#include <cmath> 
#include <cstdio>

#include <iostream> 
#include <vector>

#include <immintrin.h>


#include "Constant.h"
#include "FunctionDerivMatch.h"
#include "SupportFunctions.h"
#include "MathFunctions.h"
#include "ExtLibrary.h"


namespace NPME_Library
{
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
//************Smooth Kernel Derivative Matching Functions**********************
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

void NPME_FunctionDerivMatch_HelmholtzRadialDeriv (_Complex double *f, 
  const int N, const double r, const _Complex double k0)
//uses Leibnitz product rule for 
//f[n] = (1/r d/dr)^n exp(I*k0*r)/r
{
  const double r2         = r*r;
  const _Complex double C = cexp(I*k0*r);
  const _Complex double A = k0*k0/r2;

  f[0] = C/r;
  f[1] = -C/r/r2 + I*k0*C/r2;
  for (int n = 2; n <= N; n++)
    f[n] = -(2.0*n-1.0)/r2*f[n-1] - A*f[n-2];
}

void NPME_FunctionDerivMatch_HelmholtzRadialDeriv_OLD (_Complex double *f, 
  const int N, const double r, const _Complex double k0)
//uses scaled spherical hankel functions
//calculates f[N+1] where
//f[0] = cexp(I*k0*r)/r
//f[1] = (1/r d/dr)   f[0]
//f[n] = (1/r d/dr)^n f[0]
{
  const _Complex double z = r*k0;
//const _Complex double C   = -k0*k0/z;
  const _Complex double C   = -k0/r;
  _Complex double     fact  = I*k0;

  NPME_SphereHankel (f, (int) N, z);
  for (int n = 0; n <= N; n++)
  {
    f[n] *= fact;
    fact *= C;
  }
}

void NPME_FunctionDerivMatch_RalphaRadialDeriv (double *f, const int N, 
  const double r, const double alpha)
//calculates f[N+1] where
//f[0] = r^alpha
//f[1] = (1/r d/dr)   f[0]
//f[n] = (1/r d/dr)^n f[0]
{
  const double r2 = r*r;

  f[0] = pow(r, alpha);
  for (int n = 1; n <= N; n++)
    f[n] = f[n-1]*(alpha - 2*n + 2)/r2;
}

/*
void NPME_FunctionDerivMatch_Xmatrix_OLD (_Complex double *X, 
  const int N, const double r)
//calculates X[ (N+1)*(N+1)]
{
  for (int n = 0; n <= N; n++)
  for (int k = 0; k <= N; k++)
  {
    if (k < n)
      X[n*(N+1) + k] = 0.0;
    else
    {
      const double C1 = NPME_IntPow (2.0, (int) n);
      const double C2 = NPME_IntPow (r*r, (int) (k-n));
      const double C3 = NPME_Factorial2 ( (int) k, (int) (k-n));
      
      X[n*(N+1) + k] = C1*C2*C3;
    }
  }
}

void NPME_FunctionDerivMatch_Xmatrix_OLD (double *X, 
  const int N, const double r)
//calculates X[ (N+1)*(N+1)]
//X[n][k] = 2^n*Rdir^(2*(k-n))*k!/(k-n)!
{
  for (int n = 0; n <= N; n++)
  for (int k = 0; k <= N; k++)
  {
    if (k < n)
      X[n*(N+1) + k] = 0.0;
    else
    {
      const double C1 = NPME_IntPow (2.0, (int) n);
      const double C2 = NPME_IntPow (r*r, (int) (k-n));
      const double C3 = NPME_Factorial2 ( (int) k, (int) (k-n));
      
      X[n*(N+1) + k] = C1*C2*C3;
    }
  }
}

bool NPME_FunctionDerivMatch_CalcEvenSeries_OLD (_Complex double *a, 
  _Complex double *b, _Complex double *fRad, 
  const int Nder, const double Rdir)
//input:  fRad[nDer+1] = (1/r d/dr)^n f0(r) at r = Rdir
//        n = 0, 1, .. Nder
//output: a[nDer+1]
//        b[nDer+1]
{
  if (Nder < 2)
  {
    std::cout << "Error in NPME_FunctionDerivMatch_CalcEvenSeries.\n";
    std::cout << "  Nder = " << Nder << " < 2.  Nder must be at least 2\n";
    return false;
  }


  std::vector<_Complex double> X( (Nder+1)*(Nder+1));
  NPME_FunctionDerivMatch_Xmatrix (&X[0], Nder, Rdir);
  NPME_SolveLinearSystemComplex (Nder+1, 1, &X[0], fRad, a);

  for (int i = 0; i <= Nder-1; i++)
    b[i] = a[i+1]*2*(i+1);
  b[Nder] = 0;

  return true;
}

bool NPME_FunctionDerivMatch_CalcEvenSeries_OLD (double *a, double *b, 
  double *fRad, const int Nder, const double Rdir)
//input:  fRad[nDer+1] = (1/r d/dr)^n f0(r) at r = Rdir
//        n = 0, 1, .. Nder
//output: a[nDer+1]
//        b[nDer+1]
{
  if (Nder < 2)
  {
    std::cout << "Error in NPME_FunctionDerivMatch_CalcEvenSeries.\n";
    std::cout << "  Nder = " << Nder << " < 2.  Nder must be at least 2\n";
    return false;
  }

  std::vector<double> X( (Nder+1)*(Nder+1));
  NPME_FunctionDerivMatch_Xmatrix (&X[0], Nder, Rdir);
  NPME_SolveLinearSystem (Nder+1, 1, &X[0], fRad, a);

  for (int i = 0; i <= Nder-1; i++)
    b[i] = a[i+1]*2*(i+1);
  b[Nder] = 0;

  return true;
}
*/

void NPME_FunctionDerivMatch_Xmatrix (double *X, 
  const int N, const double r)
//calculates X[ (N+1)*(N+1)]
{
  for (int n = 0; n <= N; n++)
  {
    const double C0 = NPME_IntPow (2.0/(r*r), n);
    for (int p = 0; p <= N; p++)
    {
      if (p < n)
        X[n*(N+1) + p] = 0.0;
      else
      {
        const double C3 = 1.0/NPME_Factorial (p-n);
        X[n*(N+1) + p] = C0*C3;
      }
    }
  }
}
void NPME_FunctionDerivMatch_Xmatrix (_Complex double *X, 
  const int N, const double r)
//calculates X[ (N+1)*(N+1)]
{
  for (int n = 0; n <= N; n++)
  {
    const double C0 = NPME_IntPow (2.0/(r*r), n);
    for (int p = 0; p <= N; p++)
    {
      if (p < n)
        X[n*(N+1) + p] = 0.0;
      else
      {
        const double C3 = 1.0/NPME_Factorial (p-n);
        X[n*(N+1) + p] = C0*C3;
      }
    }
  }
}

double NPME_FunctionDerivMatch_Calc_Cnp (int n, int p, double Rdir)
{
  if (n > p)
    return 0;

  double C0 = NPME_IntPow (Rdir*Rdir, p-n);
  double C1 = NPME_IntPow (2, n);
  double C2 = NPME_Factorial (p)/NPME_Factorial (p-n);

  return C0*C1*C2;
}

bool NPME_FunctionDerivMatch_CalcEvenSeries (double *a, 
  double *b, const double *fRad, const int Nder, const double Rdir)
//input:  fRad[nDer+1] = (1/r d/dr)^n f0(r) at r = Rdir
//        n = 0, 1, .. Nder
//output: a[nDer+1]
//        b[nDer+1]
{
  if (Nder < 2)
  {
    std::cout << "Error in NPME_FunctionDerivMatch_CalcEvenSeries.\n";
    std::cout << "  Nder = " << Nder << " < 2.  Nder must be at least 2\n";
    return false;
  }

  for (int p = 0; p <= Nder; p++)
  {
    a[Nder-p] = fRad[Nder-p];
    for (int q = 0; q < p; q++)
      a[Nder-p] -= NPME_FunctionDerivMatch_Calc_Cnp (Nder-p, Nder-q, Rdir)*
                   a[Nder-q];
    a[Nder-p] /= NPME_FunctionDerivMatch_Calc_Cnp (Nder-p, Nder-p, Rdir);
  }


  for (int i = 0; i <= Nder-1; i++)
    b[i] = a[i+1]*2*(i+1);
  b[Nder] = 0;

  return true;
}
bool NPME_FunctionDerivMatch_CalcEvenSeries (_Complex double *a, 
  _Complex double *b, const _Complex double *fRad, 
  const int Nder, const double Rdir)
//input:  fRad[nDer+1] = (1/r d/dr)^n f0(r) at r = Rdir
//        n = 0, 1, .. Nder
//output: a[nDer+1]
//        b[nDer+1]
{
  if (Nder < 2)
  {
    std::cout << "Error in NPME_FunctionDerivMatch_CalcEvenSeries.\n";
    std::cout << "  Nder = " << Nder << " < 2.  Nder must be at least 2\n";
    return false;
  }

  for (int p = 0; p <= Nder; p++)
  {
    a[Nder-p] = fRad[Nder-p];
    for (int q = 0; q < p; q++)
      a[Nder-p] -= NPME_FunctionDerivMatch_Calc_Cnp (Nder-p, Nder-q, Rdir)*
                   a[Nder-q];
    a[Nder-p] /= NPME_FunctionDerivMatch_Calc_Cnp (Nder-p, Nder-p, Rdir);
  }


  for (int i = 0; i <= Nder-1; i++)
    b[i] = a[i+1]*2*(i+1);
  b[Nder] = 0;

  return true;
}

bool NPME_FunctionDerivMatch_CalcEvenSeries_Solve (double *a, 
  double *b, double *fRad, const int Nder, const double Rdir)
//input:  fRad[nDer+1] = (1/r d/dr)^n f0(r) at r = Rdir
//        n = 0, 1, .. Nder
//output: a[nDer+1]
//        b[nDer+1]
{
  if (Nder < 2)
  {
    std::cout << "Error in NPME_FunctionDerivMatch_CalcEvenSeries_Solve.\n";
    std::cout << "  Nder = " << Nder << " < 2.  Nder must be at least 2\n";
    return false;
  }

//c_2p = c[p] = a[p]*p!*Rdir^2p


  std::vector<double> X( (Nder+1)*(Nder+1));
  NPME_FunctionDerivMatch_Xmatrix (&X[0], Nder, Rdir);
  NPME_SolveLinearSystem (Nder+1, 1, &X[0], fRad, a);

  for (int p = 0; p <= Nder; p++)
    a[p] /= (NPME_IntPow (Rdir*Rdir, p)*NPME_Factorial (p) );

  for (int i = 0; i <= Nder-1; i++)
    b[i] = a[i+1]*2*(i+1);
  b[Nder] = 0;

  return true;
}
bool NPME_FunctionDerivMatch_CalcEvenSeries_Solve (_Complex double *a, 
  _Complex double *b, _Complex double *fRad, 
  const int Nder, const double Rdir)
//input:  fRad[nDer+1] = (1/r d/dr)^n f0(r) at r = Rdir
//        n = 0, 1, .. Nder
//output: a[nDer+1]
//        b[nDer+1]
{
  if (Nder < 2)
  {
    std::cout << "Error in NPME_FunctionDerivMatch_CalcEvenSeries.\n";
    std::cout << "  Nder = " << Nder << " < 2.  Nder must be at least 2\n";
    return false;
  }

//c_2p = c[p] = a[p]*p!*Rdir^2p


  std::vector<_Complex double> X( (Nder+1)*(Nder+1));
  NPME_FunctionDerivMatch_Xmatrix (&X[0], Nder, Rdir);
  NPME_SolveLinearSystemComplex (Nder+1, 1, &X[0], fRad, a);

  for (int p = 0; p <= Nder; p++)
    a[p] /= (NPME_IntPow (Rdir*Rdir, p)*NPME_Factorial (p) );

  for (int i = 0; i <= Nder-1; i++)
    b[i] = a[i+1]*2*(i+1);
  b[Nder] = 0;

  return true;
}



double NPME_FunctionDerivMatch_EvenSeriesReal (const int N, 
  const double *a, const double r2)
//a[N+1], calculates 
//Sum{a[n]*r^2n } n = 0, .. N
{
  double sum = a[N-1] + a[N]*r2;
  for (int k = N - 2; k >= 0; k--)
  {
    sum *= r2;
    sum += a[k];
  }

  return sum;
}

double NPME_FunctionDerivMatch_EvenSeriesReal (double& f1, 
  const int N, const double *a, const double *b, 
  const double r2)
//a[N+1], calculates 
//f0 = Sum{a[n]*r^2n } n = 0, .. N
//f1 = 1/r d/dr f0(r)
//   = Sum{b[n]*r^2n } n = 0, .. N-1
{
  double f0 = a[N-1] + a[N]*r2;
  for (int k = N - 2; k >= 0; k--)
  {
    f0 *= r2;
    f0 += a[k];
  }

  f1 = b[N-2] + b[N-1]*r2;
  for (int k = N - 3; k >= 0; k--)
  {
    f1 *= r2;
    f1 += b[k];
  }

  return f0;
}

_Complex double NPME_FunctionDerivMatch_EvenSeriesComplex (const int N, 
  const _Complex double *a, const double r2)
//a[N+1], calculates 
//Sum{a[n]*r^2n } n = 0, .. N
{
  _Complex double sum = a[N-1] + a[N]*r2;
  for (int k = N - 2; k >= 0; k--)
  {
    sum *= r2;
    sum += a[k];
  }

  return sum;
}

_Complex double NPME_FunctionDerivMatch_EvenSeriesComplex (_Complex double& f1, 
  const int N, const _Complex double *a, const _Complex double *b, 
  const double r2)
//a[N+1], calculates 
//f0 = Sum{a[n]*r^2n } n = 0, .. N
//f1 = 1/r d/dr f0(r)
//   = Sum{b[n]*r^2n } n = 0, .. N-1
{
  _Complex double f0 = a[N-1] + a[N]*r2;
  for (int k = N - 2; k >= 0; k--)
  {
    f0 *= r2;
    f0 += a[k];
  }

  f1 = b[N-2] + b[N-1]*r2;
  for (int k = N - 3; k >= 0; k--)
  {
    f1 *= r2;
    f1 += b[k];
  }


  return f0;
}



}//end namespace NPME_Library



