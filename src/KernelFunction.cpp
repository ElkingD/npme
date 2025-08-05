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

#include <cstdio>
#include <cstdlib> 
#include <cstring> 
#include <cmath> 
#include <cstdio>

#include <iostream> 
#include <vector>

#include <immintrin.h>

#include "Constant.h"
#include "KernelFunction.h"
#include "FunctionDerivMatch.h"
#include "SupportFunctions.h"
#include "ExtLibrary.h"


namespace NPME_Library
{

bool NPME_KernelFuncCheck (NPME_Library::NPME_KfuncReal& func, 
  const size_t N, const char *funcName, const double Xmin, const double Xmax, 
  int vecOption, bool PRINT, bool PRINT_ALL, std::ostream& os)
//tests func with numerical derivatives and 
//compares AVX and AVX_512 implementation with scalar implementation
//N is the array size to test on
//random x,y,z coordinates between (Xmin,Xmax)
{
  const double tol            = 1.0E-12;
  const double tol_numeDeriv  = 1.0E-6;
  const double hNume          = 1.0E-5;
  char str[2000];

  if (PRINT_ALL)
  {
    std::cout << "\n\n\nNPME_KernelFuncCheck for " << funcName << "\n";
  }

  //input coordinates
  double *x       = (double *) NPME_malloc (N*sizeof(double), 64);
  double *y       = (double *) NPME_malloc (N*sizeof(double), 64);
  double *z       = (double *) NPME_malloc (N*sizeof(double), 64);

  double *f0      = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fX      = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fY      = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fZ      = (double *) NPME_malloc (N*sizeof(double), 64);

  double *f0_ph   = (double *) NPME_malloc (N*sizeof(double), 64);
  double *f0_mh   = (double *) NPME_malloc (N*sizeof(double), 64);
  double *f0_nume = (double *) NPME_malloc (N*sizeof(double), 64);

  double *f0_ref  = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fX_ref  = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fY_ref  = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fZ_ref  = (double *) NPME_malloc (N*sizeof(double), 64);

  for (int p = 0; p < N; p++)
  {
    x[p] = NPME_GetDoubleRand (Xmin, Xmax);
    y[p] = NPME_GetDoubleRand (Xmin, Xmax);
    z[p] = NPME_GetDoubleRand (Xmin, Xmax);
  }




  //1) calculate f0_ref[N]
  memcpy(&f0_ref[0], &x[0], N*sizeof(double));
  func.Calc (N, &f0_ref[0], &y[0], &z[0]);

  //2) calculate analytic derivatives
  memcpy(&fX_ref[0], &x[0], N*sizeof(double));
  memcpy(&fY_ref[0], &y[0], N*sizeof(double));
  memcpy(&fZ_ref[0], &z[0], N*sizeof(double));
  func.Calc (N, &f0[0], &fX_ref[0], &fY_ref[0], &fZ_ref[0]);

  //3) compare f0[N] and f0_ref (non-vectorized)
  double error;
  double maxError = 0;
  double maxErrorNumeDeriv = 0;

  NPME_CompareArrays (error, N, "f0", &f0_ref[0], "f0", &f0[0], PRINT_ALL, os);

  if (error > tol)
  {
    std::cout << "Error in NPME_KernelFuncCheck.\n";
    sprintf(str, "error in f0 (scalar) = %.2le > %.2le = tol\n", error, tol);
    std::cout << str;
    return false;
  }
  if (maxError < error)
    maxError = error;

  if (PRINT_ALL)
  {
    sprintf(str, "f0 error scalar = %.2le\n", error);
    os << str;
  }

  //4) test numerical derivatives
  for (int p = 0; p < 3; p++)
  {
    //perturb coords by +h
    for (size_t i = 0; i < N; i++)
    {
      if      (p == 0) x[i] += hNume;
      else if (p == 1) y[i] += hNume;
      else if (p == 2) z[i] += hNume;
    }
    memcpy(&f0_ph[0], &x[0], N*sizeof(double));
    func.Calc (N, &f0_ph[0], &y[0], &z[0]);

    //perturb coords by -h
    for (size_t i = 0; i < N; i++)
    {
      if      (p == 0) x[i] -= 2*hNume;
      else if (p == 1) y[i] -= 2*hNume;
      else if (p == 2) z[i] -= 2*hNume;
    }
    memcpy(&f0_mh[0], &x[0], N*sizeof(double));
    func.Calc (N, &f0_mh[0], &y[0], &z[0]);

    //perturb coords back to original values
    for (size_t i = 0; i < N; i++)
    {
      if      (p == 0) x[i] += hNume;
      else if (p == 1) y[i] += hNume;
      else if (p == 2) z[i] += hNume;

      f0_nume[i] = (f0_ph[i] - f0_mh[i])/2/hNume;
    }
  
    if (p == 0)
    {
      NPME_CompareArrays (error, N, "fX_analytic ",  &fX_ref[0], 
                                    "fX_numerical", &f0_nume[0], 
                        PRINT_ALL, os);
      if (maxErrorNumeDeriv < error)
        maxErrorNumeDeriv = error;
      if (error > tol_numeDeriv)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "numerical derivative error in fX = %.2le > %.2le = tol\n", 
          error, tol_numeDeriv);
        std::cout << str;
        return false;
      }
    }
    else if (p == 1)
    {
      NPME_CompareArrays (error, N, "fY_analytic ",  &fY_ref[0], 
                                    "fY_numerical", &f0_nume[0], 
                        PRINT_ALL, os);
      if (maxErrorNumeDeriv < error)
        maxErrorNumeDeriv = error;
      if (error > tol_numeDeriv)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "numerical derivative error in fY = %.2le > %.2le = tol\n", 
          error, tol_numeDeriv);
        std::cout << str;
        return false;
      }
    }
    else if (p == 2)
    {
      NPME_CompareArrays (error, N, "fZ_analytic ",  &fZ_ref[0], 
                                    "fZ_numerical", &f0_nume[0], 
                        PRINT_ALL, os);
      if (maxErrorNumeDeriv < error)
        maxErrorNumeDeriv = error;
      if (error > tol_numeDeriv)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "numerical derivative error in fZ = %.2le > %.2le = tol\n", 
          error, tol_numeDeriv);
        std::cout << str;
        return false;
      }
    }
  }

  if (vecOption >= 1)
  {
    #if NPME_USE_AVX
    if (N%4 == 0)
    {
      memcpy(&f0[0], &x[0], N*sizeof(double));
      func.CalcAVX (N, &f0[0], &y[0], &z[0]);

      NPME_CompareArrays (error, N, "f0_ref", &f0_ref[0], "f0_AVX", 
        &f0[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in f0 (AVX) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }

      memcpy(&fX[0], &x[0], N*sizeof(double));
      memcpy(&fY[0], &y[0], N*sizeof(double));
      memcpy(&fZ[0], &z[0], N*sizeof(double));
      func.CalcAVX (N, &f0[0], &fX[0], &fY[0], &fZ[0]);

      NPME_CompareArrays (error, N, "fX_ref", &fX_ref[0], "fX_AVX", 
        &fX[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fX (AVX) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }

      NPME_CompareArrays (error, N, "fY_ref", &fY_ref[0], "fY_AVX", 
        &fY[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fY (AVX) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }

      NPME_CompareArrays (error, N, "fZ_ref", &fZ_ref[0], "fZ_AVX", 
        &fZ[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fZ (AVX) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }
    }
    #else
    {
      std::cout << "Error in NPME_KernelFuncCheck.\n";
      sprintf(str, "vecOption = %d but NPME_USE_AVX is not set\n", vecOption);
      std::cout << str;
      return false;
    }
    #endif
  }

  if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    if (N%8 == 0)
    {
      memcpy(&f0[0], &x[0], N*sizeof(double));
      func.CalcAVX_512 (N, &f0[0], &y[0], &z[0]);

      NPME_CompareArrays (error, N, "f0_ref    ", &f0_ref[0], "f0_AVX_512", 
        &f0[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in f0 (AVX_512) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }

      memcpy(&fX[0], &x[0], N*sizeof(double));
      memcpy(&fY[0], &y[0], N*sizeof(double));
      memcpy(&fZ[0], &z[0], N*sizeof(double));
      func.CalcAVX_512 (N, &f0[0], &fX[0], &fY[0], &fZ[0]);

      NPME_CompareArrays (error, N, "fX_ref    ", &fX_ref[0], "fX_AVX_512", 
        &fX[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fX (AVX_512) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }

      NPME_CompareArrays (error, N, "fY_ref    ", &fY_ref[0], "fY_AVX_512", 
        &fY[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fY (AVX_512) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }

      NPME_CompareArrays (error, N, "fZ_ref    ", &fZ_ref[0], "fZ_AVX_512", 
        &fZ[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fZ (AVX_512) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }
    }

    #else
    {
      std::cout << "Error in NPME_KernelFuncCheck.\n";
      sprintf(str, "vecOption = %d but NPME_USE_AVX_512 is not set\n", 
        vecOption);
      std::cout << str;
      return false;
    }
    #endif
  }


  if (PRINT)
  {
    sprintf(str,"NPME_KernelFuncCheck %s  N = %4lu maxError = %.2le maxErrorNumeDeriv = %.2le\n",
      funcName, N, maxError, maxErrorNumeDeriv);
    os << str;
  }


  NPME_free(x);
  NPME_free(y);
  NPME_free(z);

  NPME_free(f0);
  NPME_free(fX);
  NPME_free(fY);
  NPME_free(fZ);

  NPME_free(f0_ph);
  NPME_free(f0_mh);
  NPME_free(f0_nume);

  NPME_free(f0_ref);
  NPME_free(fX_ref);
  NPME_free(fY_ref);
  NPME_free(fZ_ref);


  return true;
}

bool NPME_KernelFuncCheck (NPME_Library::NPME_KfuncComplex& func, 
  const size_t N, const char *funcName, const double Xmin, const double Xmax, 
  int vecOption, bool PRINT, bool PRINT_ALL, std::ostream& os)
//tests func with numerical derivatives and 
//compares AVX and AVX_512 implementation with scalar implementation
//N is the array size to test on
//random x,y,z coordinates between (Xmin,Xmax)
{
  const double tol            = 1.0E-12;
  const double tol_numeDeriv  = 1.0E-6;
  const double hNume          = 1.0E-5;
  char str[2000];

  if (PRINT_ALL)
  {
    std::cout << "\n\n\nNPME_KernelFuncCheck for " << funcName << "\n";
  }

  //input coordinates
  double *x         = (double *) NPME_malloc (N*sizeof(double), 64);
  double *y         = (double *) NPME_malloc (N*sizeof(double), 64);
  double *z         = (double *) NPME_malloc (N*sizeof(double), 64);
  double *f0_r      = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fX_r      = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fY_r      = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fZ_r      = (double *) NPME_malloc (N*sizeof(double), 64);

  double *f0_ph_r   = (double *) NPME_malloc (N*sizeof(double), 64);
  double *f0_mh_r   = (double *) NPME_malloc (N*sizeof(double), 64);
  double *f0_nume_r = (double *) NPME_malloc (N*sizeof(double), 64);

  double *f0_ref_r  = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fX_ref_r  = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fY_ref_r  = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fZ_ref_r  = (double *) NPME_malloc (N*sizeof(double), 64);

  double *f0_i      = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fX_i      = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fY_i      = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fZ_i      = (double *) NPME_malloc (N*sizeof(double), 64);

  double *f0_ph_i   = (double *) NPME_malloc (N*sizeof(double), 64);
  double *f0_mh_i   = (double *) NPME_malloc (N*sizeof(double), 64);
  double *f0_nume_i = (double *) NPME_malloc (N*sizeof(double), 64);

  double *f0_ref_i  = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fX_ref_i  = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fY_ref_i  = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fZ_ref_i  = (double *) NPME_malloc (N*sizeof(double), 64);

  for (int p = 0; p < N; p++)
  {
    x[p] = NPME_GetDoubleRand (Xmin, Xmax);
    y[p] = NPME_GetDoubleRand (Xmin, Xmax);
    z[p] = NPME_GetDoubleRand (Xmin, Xmax);
  }





  //1) calculate f0_ref[N]
  memcpy(&f0_ref_r[0], &x[0], N*sizeof(double));
  func.Calc (N, &f0_ref_r[0], &f0_ref_i[0], &y[0], &z[0]);

  //2) calculate analytic derivatives
  memcpy(&fX_ref_r[0], &x[0], N*sizeof(double));
  memcpy(&fY_ref_r[0], &y[0], N*sizeof(double));
  memcpy(&fZ_ref_r[0], &z[0], N*sizeof(double));
  func.Calc (N, &f0_r[0],     &f0_i[0], 
                &fX_ref_r[0], &fX_ref_i[0], 
                &fY_ref_r[0], &fY_ref_i[0], 
                &fZ_ref_r[0], &fZ_ref_i[0]);

  //3) compare f0[N] and f0_ref (non-vectorized)
  double error;
  double maxError = 0;
  double maxErrorNumeDeriv = 0;

  NPME_CompareArrays (error, N, "f0_r", &f0_ref_r[0], 
                                "f0_r", &f0_r[0], PRINT_ALL, os);

  if (error > tol)
  {
    std::cout << "Error in NPME_KernelFuncCheck.\n";
    sprintf(str, "error in f0_r (scalar) = %.2le > %.2le = tol\n", error, tol);
    std::cout << str;
    return false;
  }
  if (maxError < error)
    maxError = error;
  if (PRINT_ALL)
  {
    sprintf(str, "f0_r error scalar = %.2le\n", error);
    os << str;
  }

  if (error > tol)
  {
    std::cout << "Error in NPME_KernelFuncCheck.\n";
    sprintf(str, "error in f0_i (scalar) = %.2le > %.2le = tol\n", error, tol);
    std::cout << str;
    return false;
  }
  if (maxError < error)
    maxError = error;

  if (PRINT_ALL)
  {
    sprintf(str,"f0_i error scalar = %.2le\n", error);
    os << str;
  }


  //4) test numerical derivatives
  for (int p = 0; p < 3; p++)
  {
    //perturb coords by +h
    for (size_t i = 0; i < N; i++)
    {
      if      (p == 0) x[i] += hNume;
      else if (p == 1) y[i] += hNume;
      else if (p == 2) z[i] += hNume;
    }
    memcpy(&f0_ph_r[0], &x[0], N*sizeof(double));
    func.Calc (N, &f0_ph_r[0], &f0_ph_i[0], &y[0], &z[0]);

    //perturb coords by -h
    for (size_t i = 0; i < N; i++)
    {
      if      (p == 0) x[i] -= 2*hNume;
      else if (p == 1) y[i] -= 2*hNume;
      else if (p == 2) z[i] -= 2*hNume;
    }
    memcpy(&f0_mh_r[0], &x[0], N*sizeof(double));
    func.Calc (N, &f0_mh_r[0], &f0_mh_i[0], &y[0], &z[0]);

    //perturb coords back to original values
    for (size_t i = 0; i < N; i++)
    {
      if      (p == 0) x[i] += hNume;
      else if (p == 1) y[i] += hNume;
      else if (p == 2) z[i] += hNume;

      f0_nume_r[i] = (f0_ph_r[i] - f0_mh_r[i])/2/hNume;
      f0_nume_i[i] = (f0_ph_i[i] - f0_mh_i[i])/2/hNume;
    }
  
    if (p == 0)
    {
      NPME_CompareArrays (error, N, "fX_r_analytic ",  &fX_ref_r[0], 
                                    "fX_r_numerical", &f0_nume_r[0], 
                        PRINT_ALL, os);
      if (maxErrorNumeDeriv < error)
        maxErrorNumeDeriv = error;
      if (error > tol_numeDeriv)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "numerical derivative error in fX_r = %.2le > %.2le = tol\n", 
          error, tol_numeDeriv);
        std::cout << str;
        return false;
      }

      NPME_CompareArrays (error, N, "fX_i_analytic ",  &fX_ref_i[0], 
                                    "fX_i_numerical", &f0_nume_i[0], 
                        PRINT_ALL, os);
      if (maxErrorNumeDeriv < error)
        maxErrorNumeDeriv = error;
      if (error > tol_numeDeriv)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "numerical derivative error in fX_i = %.2le > %.2le = tol\n", 
          error, tol_numeDeriv);
        std::cout << str;
        return false;
      }
    }
    else if (p == 1)
    {
      NPME_CompareArrays (error, N, "fY_r_analytic ",  &fY_ref_r[0], 
                                    "fY_r_numerical", &f0_nume_r[0], 
                        PRINT_ALL, os);
      if (maxErrorNumeDeriv < error)
        maxErrorNumeDeriv = error;
      if (error > tol_numeDeriv)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "numerical derivative error in fX_r = %.2le > %.2le = tol\n", 
          error, tol_numeDeriv);
        std::cout << str;
        return false;
      }

      NPME_CompareArrays (error, N, "fY_i_analytic ",  &fY_ref_i[0], 
                                    "fY_i_numerical", &f0_nume_i[0], 
                        PRINT_ALL, os);
      if (maxErrorNumeDeriv < error)
        maxErrorNumeDeriv = error;
      if (error > tol_numeDeriv)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "numerical derivative error in fY_i = %.2le > %.2le = tol\n", 
          error, tol_numeDeriv);
        std::cout << str;
        return false;
      }
    }
    else if (p == 2)
    {
      NPME_CompareArrays (error, N, "fZ_r_analytic ",  &fZ_ref_r[0], 
                                    "fZ_r_numerical", &f0_nume_r[0], 
                        PRINT_ALL, os);
      if (maxErrorNumeDeriv < error)
        maxErrorNumeDeriv = error;
      if (error > tol_numeDeriv)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "numerical derivative error in fZ_r = %.2le > %.2le = tol\n", 
          error, tol_numeDeriv);
        std::cout << str;
        return false;
      }

      NPME_CompareArrays (error, N, "fZ_i_analytic ",  &fZ_ref_i[0], 
                                    "fZ_i_numerical", &f0_nume_i[0], 
                        PRINT_ALL, os);
      if (maxErrorNumeDeriv < error)
        maxErrorNumeDeriv = error;
      if (error > tol_numeDeriv)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "numerical derivative error in fZ_i = %.2le > %.2le = tol\n", 
          error, tol_numeDeriv);
        std::cout << str;
        return false;
      }
    }
  }

  if (vecOption >= 1)
  {
    #if NPME_USE_AVX
    if (N%4 == 0)
    {
      memcpy(&f0_r[0], &x[0], N*sizeof(double));
      func.CalcAVX (N, &f0_r[0], &f0_i[0], &y[0], &z[0]);

      NPME_CompareArrays (error, N, "f0_r_ref", &f0_ref_r[0], "f0_r_AVX", 
        &f0_r[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in f0_r (AVX) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }
      NPME_CompareArrays (error, N, "f0_i_ref", &f0_ref_i[0], "f0_i_AVX", 
        &f0_i[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in f0_i (AVX) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }


      memcpy(&fX_r[0], &x[0], N*sizeof(double));
      memcpy(&fY_r[0], &y[0], N*sizeof(double));
      memcpy(&fZ_r[0], &z[0], N*sizeof(double));
      func.CalcAVX (N, &f0_r[0], &f0_i[0], &fX_r[0], &fX_i[0], 
                       &fY_r[0], &fY_i[0], &fZ_r[0], &fZ_i[0]);

      NPME_CompareArrays (error, N, "fX_r_ref", &fX_ref_r[0], "fX_r_AVX", 
        &fX_r[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fX_r (AVX) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }
      NPME_CompareArrays (error, N, "fX_i_ref", &fX_ref_i[0], "fX_i_AVX", 
        &fX_i[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fX_i (AVX) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }


      NPME_CompareArrays (error, N, "fY_r_ref", &fY_ref_r[0], "fY_r_AVX", 
        &fY_r[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fY_r (AVX) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }
      NPME_CompareArrays (error, N, "fY_i_ref", &fY_ref_i[0], "fY_i_AVX", 
        &fY_i[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fY_i (AVX) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }



      NPME_CompareArrays (error, N, "fZ_r_ref", &fZ_ref_r[0], "fZ_r_AVX", 
        &fZ_r[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fZ_r (AVX) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }
      NPME_CompareArrays (error, N, "fZ_i_ref", &fZ_ref_i[0], "fZ_i_AVX", 
        &fZ_i[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fZ_i (AVX) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }
    }
    #else
    {
      std::cout << "Error in NPME_KernelFuncCheck.\n";
      sprintf(str, "vecOption = %d < 1 but NPME_USE_AVX is not set\n", 
        vecOption);
      std::cout << str;
      return false;
    }
    #endif
  }

  if (vecOption == 2)
  {
    #if NPME_USE_AVX_512
    if (N%8 == 0)
    {
      memcpy(&f0_r[0], &x[0], N*sizeof(double));
      func.CalcAVX_512 (N, &f0_r[0], &f0_i[0], &y[0], &z[0]);

      NPME_CompareArrays (error, N, "f0_r_ref    ", &f0_ref_r[0], "f0_r_AVX_512", 
        &f0_r[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in f0_r (AVX_512) = %.2le > %.2le = tol\n", 
          error, tol);
        std::cout << str;
        return false;
      }
      NPME_CompareArrays (error, N, "f0_i_ref    ", &f0_ref_i[0], "f0_i_AVX_512", 
        &f0_i[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in f0_i (AVX_512) = %.2le > %.2le = tol\n", 
          error, tol);
        std::cout << str;
        return false;
      }


      memcpy(&fX_r[0], &x[0], N*sizeof(double));
      memcpy(&fY_r[0], &y[0], N*sizeof(double));
      memcpy(&fZ_r[0], &z[0], N*sizeof(double));
      func.CalcAVX_512 (N, &f0_r[0], &f0_i[0], &fX_r[0], &fX_i[0], 
                       &fY_r[0], &fY_i[0], &fZ_r[0], &fZ_i[0]);

      NPME_CompareArrays (error, N, "fX_r_ref    ", &fX_ref_r[0], "fX_r_AVX_512", 
        &fX_r[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fX_r (AVX_512) = %.2le > %.2le = tol\n", 
          error, tol);
        std::cout << str;
        return false;
      }
      NPME_CompareArrays (error, N, "fX_i_ref    ", &fX_ref_i[0], "fX_i_AVX_512", 
        &fX_i[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fX_i (AVX_512) = %.2le > %.2le = tol\n", 
          error, tol);
        std::cout << str;
        return false;
      }


      NPME_CompareArrays (error, N, "fY_r_ref    ", &fY_ref_r[0], "fY_r_AVX_512", 
        &fY_r[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fY_r (AVX_512) = %.2le > %.2le = tol\n", 
          error, tol);
        std::cout << str;
        return false;
      }
      NPME_CompareArrays (error, N, "fY_i_ref    ", &fY_ref_i[0], "fY_i_AVX_512", 
        &fY_i[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fY_i (AVX_512) = %.2le > %.2le = tol\n", 
          error, tol);
        std::cout << str;
        return false;
      }



      NPME_CompareArrays (error, N, "fZ_r_ref    ", &fZ_ref_r[0], "fZ_r_AVX_512", 
        &fZ_r[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fZ_r (AVX_512) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }
      NPME_CompareArrays (error, N, "fZ_i_ref    ", &fZ_ref_i[0], "fZ_i_AVX_512", 
        &fZ_i[0], PRINT_ALL, os);
      if (maxError < error)
        maxError = error;
      if (error > tol)
      {
        std::cout << "Error in NPME_KernelFuncCheck.\n";
        sprintf(str, "error in fZ_i (AVX_512) = %.2le > %.2le = tol\n", error, tol);
        std::cout << str;
        return false;
      }
    }
    #else
    {
      std::cout << "Error in NPME_KernelFuncCheck.\n";
      sprintf(str, "vecOption = %d < 2 but NPME_USE_AVX_512 is not set\n", 
        vecOption);
      std::cout << str;
      return false;
    }
    #endif
  }


  if (PRINT)
  {
    sprintf(str,"NPME_KernelFuncCheck %s  N = %4lu maxError = %.2le maxErrorNumeDeriv = %.2le\n",
      funcName, N, maxError, maxErrorNumeDeriv);
    os << str;
  }

  NPME_free(x);
  NPME_free(y);
  NPME_free(z);
  NPME_free(f0_r);
  NPME_free(fX_r);
  NPME_free(fY_r);
  NPME_free(fZ_r);

  NPME_free(f0_ph_r);
  NPME_free(f0_mh_r);
  NPME_free(f0_nume_r);

  NPME_free(f0_ref_r);
  NPME_free(fX_ref_r);
  NPME_free(fY_ref_r);
  NPME_free(fZ_ref_r);

  NPME_free(f0_i);
  NPME_free(fX_i);
  NPME_free(fY_i);
  NPME_free(fZ_i);

  NPME_free(f0_ph_i);
  NPME_free(f0_mh_i);
  NPME_free(f0_nume_i);

  NPME_free(f0_ref_i);
  NPME_free(fX_ref_i);
  NPME_free(fY_ref_i);
  NPME_free(fZ_ref_i);

  return true;
}


double NPME_KernelFunc_GetTime (NPME_Library::NPME_KfuncReal& func, 
  const size_t N, int vecOption, const double Xmin, const double Xmax)
//Gets CPU Time for calculating f[N], dfdx[N], dfdy[N], dfdz[N]
//vecOption = 0, 1, 2 for no vectorization, AVX, AVX-512
{
  char str[2000];

  //input coordinates
  double *x       = (double *) NPME_malloc (N*sizeof(double), 64);
  double *y       = (double *) NPME_malloc (N*sizeof(double), 64);
  double *z       = (double *) NPME_malloc (N*sizeof(double), 64);

  double *f0      = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fX      = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fY      = (double *) NPME_malloc (N*sizeof(double), 64);
  double *fZ      = (double *) NPME_malloc (N*sizeof(double), 64);
  for (int p = 0; p < N; p++)
  {
    x[p] = NPME_GetDoubleRand (Xmin, Xmax);
    y[p] = NPME_GetDoubleRand (Xmin, Xmax);
    z[p] = NPME_GetDoubleRand (Xmin, Xmax);
  }

  size_t nTime = 100;
  for (;;)
  {
    double time;

    if (vecOption == 0)
    {
      double time0 = NPME_GetTime ();
      for (size_t i = 0; i < nTime; i++)
      {
        memcpy(&fX[0], &x[0], N*sizeof(double));
        memcpy(&fY[0], &y[0], N*sizeof(double));
        memcpy(&fZ[0], &z[0], N*sizeof(double));
        func.Calc (N, &f0[0], &fX[0], &fY[0], &fZ[0]);
      }
      time = NPME_GetTime () - time0;
    }
    else if (vecOption == 1)
    {
      #if NPME_USE_AVX
      {
        if (N%4 != 0)
        {
          std::cout << "Error in NPME_KernelFunc_GetTime";
          sprintf(str, "N = %lu is not a multiple of 4\n", N);
          std::cout << str;
          exit(0);
        }
        double time0 = NPME_GetTime ();
        for (size_t i = 0; i < nTime; i++)
        {
          memcpy(&fX[0], &x[0], N*sizeof(double));
          memcpy(&fY[0], &y[0], N*sizeof(double));
          memcpy(&fZ[0], &z[0], N*sizeof(double));
          func.CalcAVX (N, &f0[0], &fX[0], &fY[0], &fZ[0]);
        }
        time = NPME_GetTime () - time0;
      }
      #else
      {
        std::cout << "Error in NPME_KernelFunc_GetTime";
        sprintf(str, "vecOption = %d, but NPME_USE_AVX is not set to 1\n", 
          vecOption);
        std::cout << str;
        exit(0);
      }
      #endif
    }
    else if (vecOption == 2)
    {
      #if NPME_USE_AVX_512
      {
        if (N%8 != 0)
        {
          std::cout << "Error in NPME_KernelFunc_GetTime";
          sprintf(str, "N = %lu is not a multiple of 8\n", N);
          std::cout << str;
          exit(0);
        }
        double time0 = NPME_GetTime ();
        for (size_t i = 0; i < nTime; i++)
        {
          memcpy(&fX[0], &x[0], N*sizeof(double));
          memcpy(&fY[0], &y[0], N*sizeof(double));
          memcpy(&fZ[0], &z[0], N*sizeof(double));
          func.CalcAVX_512 (N, &f0[0], &fX[0], &fY[0], &fZ[0]);
        }
        time = NPME_GetTime () - time0;
      }
      #else
      {
        std::cout << "Error in NPME_KernelFunc_GetTime";
        sprintf(str, "vecOption = %d, but NPME_USE_AVX_512 is not set to 1\n", 
          vecOption);
        std::cout << str;
        exit(0);
      }
      #endif
    }

    if (time > 2.0)
    {
      NPME_free(x);
      NPME_free(y);
      NPME_free(z);

      NPME_free(f0);
      NPME_free(fX);
      NPME_free(fY);
      NPME_free(fZ);

      return time/nTime;
    }

    nTime *= 2;
  }


  std::cout << "Error in NPME_KernelFunc_GetTime\n";
  exit(0);
}

bool NPME_KernelFuncCheck (NPME_Library::NPME_KfuncReal& func, 
  const size_t N, const char *funcName, const double Xmin, const double Xmax, 
  bool PRINT, bool PRINT_ALL, std::ostream& os)
{
  int vecOption = NPME_DetermineVecOptionFromCompileFlag ();
  return NPME_KernelFuncCheck (func, N, funcName, Xmin, Xmax, vecOption,
    PRINT, PRINT_ALL, os);
}
bool NPME_KernelFuncCheck (NPME_Library::NPME_KfuncComplex& func, 
  const size_t N, const char *funcName, const double Xmin, const double Xmax, 
  bool PRINT, bool PRINT_ALL, std::ostream& os)
{
  int vecOption = NPME_DetermineVecOptionFromCompileFlag ();
  return NPME_KernelFuncCheck (func, N, funcName, Xmin, Xmax, vecOption,
    PRINT, PRINT_ALL, os);
}

}//end namespace NPME_Library



