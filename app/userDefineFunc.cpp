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



//This application illustrates how a user can define their own kernel by the
//following example:            f(x,y,z) = r = sqrt(x*x + y*y + z*z)
//with smooth long range kernel f_lr (r) = r*erf (beta*r)
//and        short range kernel f_sr (r) = r*erfc(beta*r)    


#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstddef>

#include <iostream> 
#include <fstream> 
#include <sstream>
#include <vector>
#include <string>





#include "NPME_Constant.h"
#include "NPME_ReadPrint.h"
#include "NPME_Interface.h"
#include "NPME_SupportFunctions.h"
using namespace NPME_Library;


class rFunc : public NPME_KfuncReal
//calculates kernel f(r) = r
//and derivatives  df/dx, df/dy, df/dz
//for array inputs
{
public:
  virtual ~rFunc () { }

  //good practice to add default constructor, copy constructor, assignment 
  //operator.  omitted for clarity because not required 

  void Print (std::ostream& os) const { } //no internal parameters to print

  void Calc (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  //input:  x[N], y[N], z[N]   where x[] values are stored in x_f0[]
  //output: f0[N]              where f[] values are stored in x_f0[]
  //        f(r) = f0(x,y,z)

  void Calc (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  //input:  x[N], y[N], z[N]   where x[] values are stored in x_fX[]
  //                           where y[] values are stored in y_fY[]
  //                           where z[] values are stored in z_fZ[]
  //output: f0[N], dfdx[N], dfdy[N], dfdz[N]
  //                           where f[]    values are stored in f0[]
  //                           where dfdx[] values are stored in x_fX[]
  //                           where dfdy[] values are stored in y_fY[]
  //                           where dfdz[] values are stored in z_fZ[]

  //AVX intrinsic functions of above 
  //arrays are aligned 32 byte arrays and N is a multiple of 4
  #if NPME_USE_AVX
  void CalcAVX (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

  #if NPME_USE_AVX_512
  //AVX_512 intrinsic functions of above 
  //arrays are aligned 64 byte arrays and N is a multiple of 8
  void CalcAVX_512 (const size_t N, 
    double *x_f0, const double *y, const double *z) const;
  void CalcAVX_512 (const size_t N, 
    double *f0, double *x_fX, double *y_fY, double *z_fZ) const;
  #endif

private:

};


int main (int argc, char *argv[])
{
  char str[1000];
  using std::cout;


  //1) make up some random x,y,z points inside a cubic centered at the origin
  //   with length 10.0
  const size_t N = 8;    //note 8 is a multiple of 4 for AVX and 8 and AVX-512
  //usually N is much larger (e.g. N = 1024), 
  //keep N small for printing values out to screen

  double x[N]     __attribute__((aligned(64))); //aligned array 64 byte boundary
  double y[N]     __attribute__((aligned(64)));
  double z[N]     __attribute__((aligned(64)));
  double f0[N]    __attribute__((aligned(64)));
  double xSave[N] __attribute__((aligned(64)));
  for (size_t i = 0; i < N; i++)
  {
    x[i] = NPME_GetDoubleRand (-5.0, 5.0);
    y[i] = NPME_GetDoubleRand (-5.0, 5.0);
    z[i] = NPME_GetDoubleRand (-5.0, 5.0);

    xSave[i] = x[i];
    sprintf(str, "xyz = %8.4f %8.4f %8.4f\n", x[i], y[i], z[i]);  
    cout << str;
  }


  //2) calculate r = sqrt(x*x + y*y + z*z) for the N points
  //   note that x[] gets over-written with value of function
  rFunc func;
  func.Calc (N, &x[0], &y[0], &z[0]);
  for (size_t i = 0; i < N; i++)
  {
    sprintf(str, "r = %8.4f\n", x[i]);
    cout << str;
  }

  //3) calculate f = sqrt(x*x + y*y + z*z) and df/dx, df/dy, df/dz for N points
  //   note that x[] gets over-written with dfdx[]
  //   note that y[] gets over-written with dfdy[]
  //   note that z[] gets over-written with dfdz[]

  //first copy xSave[] back into x[]
  for (size_t i = 0; i < N; i++)
    x[i] = xSave[i];

  func.Calc (N, &f0[0], &x[0], &y[0], &z[0]);
  for (size_t i = 0; i < N; i++)
  {
    sprintf(str, "r = %8.4f  drdx = %8.4f  drdy = %8.4f  drdz = %8.4f\n", 
        f0[i], x[i], y[i], z[i]);
    cout << str;
  }




  //the following function checks the analytic/numerical derivatives and
  //if (N is a multiple of 4 or 8) compares scalar with AVX and AVX-512 function
  //however, it either requires non-empty implementations for AVX and AVX-512
  //or else set the macro constants in NPME_Constant.h to 0
  //    #define NPME_USE_AVX      0
  //    #define NPME_USE_AVX_512  0
  //this will turn off compiliation of explicitly vectorized functions
  {
    bool PRINT      = 1;
    bool PRINT_ALL  = 1;
    int vecOption   = 1;    //vecOption = 0, 1, 2 for none, avx, or avx-512
    rFunc func;
    NPME_KernelFuncCheck (func, N, "r", -5.0, 5.0, 
        vecOption, PRINT, PRINT_ALL, std::cout);
  }


  return 1;
}




void rFunc::Calc (const size_t N, 
  double *x_f0, const double *y, const double *z) const
//input:  x[N], y[N], z[N]   where x[] values are stored in x_f0[]
//output: f0[N]              where f[] values are stored in x_f0[]
//        f(r) = f0(x,y,z)
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_f0[i]*x_f0[i] + y[i]*y[i] + z[i]*z[i];
    const double r  = sqrt(fabs(r2));
    x_f0[i]         = r;
  }
}

void rFunc::Calc (const size_t N, 
  double *f0, double *x_fX, double *y_fY, double *z_fZ) const
//input:  x[N], y[N], z[N]   where x[] values are stored in x_fX[]
//                           where y[] values are stored in y_fY[]
//                           where z[] values are stored in z_fZ[]
//output: f0[N], dfdx[N], dfdy[N], dfdz[N]
//                           where f[]    values are stored in f0[]
//                           where dfdx[] values are stored in x_fX[]
//                           where dfdy[] values are stored in y_fY[]
//                           where dfdz[] values are stored in z_fZ[]
{
  for (size_t i = 0; i < N; i++)
  {
    const double r2 = x_fX[i]*x_fX[i] + y_fY[i]*y_fY[i] + z_fZ[i]*z_fZ[i];
    const double r  = sqrt(fabs(r2));
    f0[i]           = r;
    const double f1 = 1.0/r;

    x_fX[i]         = x_fX[i]*f1;
    y_fY[i]         = y_fY[i]*f1;
    z_fZ[i]         = z_fZ[i]*f1;
  }
}


//AVX intrinsic functions of above 
//arrays are aligned 32 byte arrays and N is a multiple of 4
#if NPME_USE_AVX
void rFunc::CalcAVX (const size_t N, 
  double *x_f0, const double *y, const double *z) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in rFunc::CalcAVX\n";
    std::cout << "N = " << N << "must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop = N/4;

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm256_load_pd (&x_f0[count]);
    yVec  = _mm256_load_pd (&y[count]);
    zVec  = _mm256_load_pd (&z[count]);

    r2Vec = _mm256_mul_pd  (xVec, xVec);
    r2Vec = _mm256_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm256_fmadd_pd  (zVec, zVec, r2Vec);
    rVec  = _mm256_sqrt_pd (r2Vec);

    _mm256_store_pd (&x_f0[count], rVec);

    count += 4;
  }
}

void rFunc::CalcAVX (const size_t N, 
  double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%4 != 0)
  {
    std::cout << "Error in rFunc::CalcAVX\n";
    std::cout << "N = " << N << "must be a multiple of 4\n";
    exit(0);
  }

  const size_t nLoop = N/4;

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m256d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm256_load_pd (&x_fX[count]);
    yVec  = _mm256_load_pd (&y_fY[count]);
    zVec  = _mm256_load_pd (&z_fZ[count]);

    r2Vec = _mm256_mul_pd  (xVec, xVec);
    r2Vec = _mm256_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm256_fmadd_pd  (zVec, zVec, r2Vec);
    rVec  = _mm256_sqrt_pd (r2Vec);

    __m256d fXVec, fYVec, fZVec;

    fXVec = _mm256_div_pd  (xVec, rVec);
    fYVec = _mm256_div_pd  (yVec, rVec);
    fZVec = _mm256_div_pd  (zVec, rVec);

    _mm256_store_pd (  &f0[count], rVec);
    _mm256_store_pd (&x_fX[count], fXVec);
    _mm256_store_pd (&y_fY[count], fYVec);
    _mm256_store_pd (&z_fZ[count], fZVec);

    count += 4;
  }
}

#endif

#if NPME_USE_AVX_512
//AVX_512 intrinsic functions of above 
//arrays are aligned 64 byte arrays and N is a multiple of 8
void rFunc::CalcAVX_512 (const size_t N, 
  double *x_f0, const double *y, const double *z) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in rFunc::CalcAVX_512\n";
    std::cout << "N = " << N << "must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop = N/8;

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm512_load_pd (&x_f0[count]);
    yVec  = _mm512_load_pd (&y[count]);
    zVec  = _mm512_load_pd (&z[count]);

    r2Vec = _mm512_mul_pd  (xVec, xVec);
    r2Vec = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm512_fmadd_pd  (zVec, zVec, r2Vec);
    rVec  = _mm512_sqrt_pd (r2Vec);

    _mm512_store_pd (&x_f0[count], rVec);

    count += 8;
  }
}

void rFunc::CalcAVX_512 (const size_t N, 
  double *f0, double *x_fX, double *y_fY, double *z_fZ) const
{
  if (N%8 != 0)
  {
    std::cout << "Error in rFunc::CalcAVX_512\n";
    std::cout << "N = " << N << "must be a multiple of 8\n";
    exit(0);
  }

  const size_t nLoop = N/8;

  size_t count = 0;
  for (size_t i = 0; i < nLoop; i++)
  {
    __m512d xVec, yVec, zVec, r2Vec, rVec;
    xVec  = _mm512_load_pd (&x_fX[count]);
    yVec  = _mm512_load_pd (&y_fY[count]);
    zVec  = _mm512_load_pd (&z_fZ[count]);

    r2Vec = _mm512_mul_pd  (xVec, xVec);
    r2Vec = _mm512_fmadd_pd  (yVec, yVec, r2Vec);
    r2Vec = _mm512_fmadd_pd  (zVec, zVec, r2Vec);
    rVec  = _mm512_sqrt_pd (r2Vec);

    __m512d fXVec, fYVec, fZVec;

    fXVec = _mm512_div_pd  (xVec, rVec);
    fYVec = _mm512_div_pd  (yVec, rVec);
    fZVec = _mm512_div_pd  (zVec, rVec);

    _mm512_store_pd (  &f0[count], rVec);
    _mm512_store_pd (&x_fX[count], fXVec);
    _mm512_store_pd (&y_fY[count], fYVec);
    _mm512_store_pd (&z_fZ[count], fZVec);

    count += 8;
  }
}
#endif





