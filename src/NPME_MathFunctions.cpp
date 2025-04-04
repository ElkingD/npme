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

#include "NPME_Constant.h"
#include "NPME_MathFunctions.h"


namespace NPME_Library
{
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
//*****************************Basic Support Functions*************************
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
double NPME_IntPowPositive (double x, int n)
//assumes n > 0
{
  double A = 1.0;
  while (n > 1)
  {
    if (n%2 == 0)
    {
      x = x*x;
      n = n/2;
    }
    else
    {
      A *= x;
      x  = x*x;
      n  = (n-1)/2;
    }
  }
  return A*x;
}

double NPME_IntPow (double x, int n)
//calculates x^n (n = integer)
{
  if (n < 0)
    return NPME_IntPowPositive (1.0/x, -n);
  else if (n == 0)
    return 1.0;
  else
    return NPME_IntPowPositive (x, n);
}
double NPME_Factorial2 (int n, int m)
//n!/m!
{
	double fact = 1.0;
	for (int i = n; i > m; i--)
		fact *= (double) i;
	return fact;
}


double NPME_Factorial (int n)
{
	double fact = 1.0;
	for (int i = 2; i <= n; i++)
		fact *= (double) i;
	return fact;
}


//*****************************************************************************
//*****************************************************************************
//*****************************************************************************
//***************Scalar Math Functions Used in Ewald Splitting*****************
//*****************************************************************************
//*****************************************************************************
//*****************************************************************************

double NPME_Berf_0 (const double x)
//B0 = erf(x)/x
//   = c[n]*x^2n (c[n] = 2/sqrt(Pi)*(-1)^n/n!/(2n+1)
{
  const double C0_0 =  1.128379167095512574;
  const double C0_2 = -3.761263890318375246E-1;
  const double C0_4 =  1.128379167095512574E-1;
  const double C0_6 = -2.686617064513125176E-2;
  const double C0_8 =  5.223977625442187842E-3;

  if (fabs(x) < 1.0E-1)
  {
    const double x2 = x*x;
    double f0;
    f0 =  x2*C0_8 + C0_6;
    f0 =  x2*f0   + C0_4;
    f0 =  x2*f0   + C0_2;
    f0 =  x2*f0   + C0_0;
    return f0;
  }
  else 
    return erf(x)/x;
}


double NPME_Berf_1 (double& B1, const double x)
//B0 = erf(x)/x
//B1 = 1/x d/dx B0
{
  const double C0_0 =  1.128379167095512574;    //2/sqrt(Pi)
  const double C0_2 = -3.761263890318375246E-1;
  const double C0_4 =  1.128379167095512574E-1;
  const double C0_6 = -2.686617064513125176E-2;
  const double C0_8 =  5.223977625442187842E-3;

  const double C1_0 = -7.522527780636750492E-1;   //  = 2*C0_2
  const double C1_2 =  4.513516668382050296E-1;   //  = 4*C0_4
  const double C1_4 = -1.611970238707875106E-1;   //  = 6*C0_6
  const double C1_6 =  4.179182100353750274E-2;   //  = 8*C0_8
  const double C1_8 = -8.548327023450852833E-3;   //  = 10*C0_10

  double B0;
  if (fabs(x) < 0.1)
  {
    const double x2 = x*x;
    B0 =  x2*C0_8 + C0_6;
    B0 =  x2*B0 + C0_4;
    B0 =  x2*B0 + C0_2;
    B0 =  x2*B0 + C0_0;

    B1 =  x2*C1_8 + C1_6;
    B1 =  x2*B1 + C1_4;
    B1 =  x2*B1 + C1_2;
    B1 =  x2*B1 + C1_0;
  }
  else 
  {
    const double x2 = x*x;
    B0 = erf(x)/x;
    B1 = (-B0 + C0_0*exp(-x2))/x2;
  }
  return B0;
}


double NPME_Berfc_0 (const double x)
//B0 = erfc(x)/x (singular at x = 0)
{
  return erfc(x)/x;
}
double NPME_Berfc_1 (double& B1, const double x)
//B0 = erfc(x)/x
//B1 = 1/x d/dx B0
{
  const double C0_0 =  1.128379167095512574;    //2/sqrt(Pi)
  const double x2   = x*x;
  const double B0   = erfc(x)/x;
  B1 = (-B0 - C0_0*exp(-x2))/x2;
  return B0;
}

double NPME_sinx_x (const double x)
//returns sin(x)/x
//sin_x   = x - x^3/3! + x^5/5! - x^7/7!
//sin_x/x = 1 - x^2/3! + x^4/5! - x^6/7!
{
  const double C0_0 =  1.000000000000000000;
  const double C0_2 = -1.666666666666666667E-1;   //-1/6      = -1/3!
  const double C0_4 =  8.333333333333333333E-3;   // 1/120    =  1/5!
  const double C0_6 = -1.984126984126984127E-4;   //-1/5040   = -1/7!
  const double C0_8 =  2.755731922398589065E-6;   // 1/362880 =  1/9!

  if (fabs(x) < 1.0E-1)
  {
    const double x2 = x*x;
    double f0;
    f0 =  x2*C0_8 + C0_6;
    f0 =  x2*f0   + C0_4;
    f0 =  x2*f0   + C0_2;
    f0 =  x2*f0   + C0_0;
    return f0;
  }
  else 
    return sin(x)/x;
}

double NPME_sinx_x (double& f1, const double x)
//returns sin(x)/x
//f1 = 1/x d/dx sin(x)/x = 1/x(cos_x/x - sin_x/x/x)
//   = cos_x/x/x - sin_x/x/x/x
//   = (x*cos_x - sin_x)/x^3
{
  const double C0_0 =  1.000000000000000000;
  const double C0_2 = -1.666666666666666667E-1;   //-1/6      = -1/3!
  const double C0_4 =  8.333333333333333333E-3;   // 1/120    =  1/5!
  const double C0_6 = -1.984126984126984127E-4;   //-1/5040   = -1/7!
  const double C0_8 =  2.755731922398589065E-6;   // 1/362880 =  1/9!

  const double C1_0 = -3.333333333333333333E-1;   //-1/3        = 2*C0_2
  const double C1_2 =  3.333333333333333333E-2;   // 1/30       = 4*C0_4
  const double C1_4 = -1.190476190476190476E-3;   //-1/840      = 6*C0_6
  const double C1_6 =  2.204585537918871252E-5;   // 1/45360    = 8*C0_8


  if (fabs(x) < 1.0E-1)
  {
    const double x2 = x*x;
    f1 =  x2*C1_6 + C1_4;
    f1 =  x2*f1   + C1_2;
    f1 =  x2*f1   + C1_0;

    double f0;
    f0 =  x2*C0_8 + C0_6;
    f0 =  x2*f0   + C0_4;
    f0 =  x2*f0   + C0_2;
    f0 =  x2*f0   + C0_0;

    return f0;
  }
  else
  {
    const double sin_x = sin(x);
    const double cos_x = cos(x);

    f1 = (cos_x - sin_x/x)/(x*x);
    return sin_x/x;
  }    
}



void NPME_SphereHankel (_Complex double *h,
  const int n, const _Complex double z)
//calculates spherical Hankel functions (first or second order) 
//by upward recursion
//input:  z = x + I*y
//        n = max order
//output: h[n+1] = {h0(z), h1(z), .. hn(z)}
//        h = h1 if y >= 0
//        h = h2 if y <  0
{
  const _Complex double inv_z = 1.0/z;

  //1) calculate h0(z) and h1(z) analytically
  if (n < 0)
  {
    std::cout << "Error in NPME_SphereBesselComplex_Hankel.\n";
    std::cout << "  n = " << n << std::endl;
    exit(0);
  }
  else if (n == 0)
  {
    if (cimag(z) >= 0.0)
    {
      h[0] = -I*cexp(+I*z)*inv_z;
    }
    else
    {
      h[0] =  I*cexp(-I*z)*inv_z;
    }
    return;
  }
  else if (n == 1)
  {
    if (cimag(z) >= 0.0)
    {
      h[0] = -I*cexp(+I*z)*inv_z;
      h[1] = (inv_z - I)*h[0];
    }
    else
    {
      h[0] =  I*cexp(-I*z)*inv_z;
      h[1] = (inv_z + I)*h[0];
    }
    return;
  }

  if (cimag(z) >= 0.0)
  {
    h[0] = -I*cexp(+I*z)*inv_z;
    h[1] = (inv_z - I)*h[0];
  }
  else
  {
    h[0] =  I*cexp(-I*z)*inv_z;
    h[1] = (inv_z + I)*h[0];
  }

  //upward recursion
  for (int i = 2; i <= n; i++)
    h[i] = (2*i-1)*inv_z*h[i-1] - h[i-2];
}


}//end namespace NPME_Library



