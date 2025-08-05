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
#include <cstring>
#include <complex.h>
#include <cstdio>

#include <iostream> 
#include <vector> 


#include "Constant.h"
#include "ExtLibrary.h"

#include <mm_malloc.h>
#include "mkl.h"



namespace NPME_Library
{

void *NPME_malloc (size_t size, size_t alignment)
{
  return _mm_malloc(size, alignment);
}
void NPME_free (void* ptr)
{
  _mm_free(ptr);
}





void NPME_RandomNumberArrayBlock (const int N, double *x, const double a, 
  const double b, const size_t seed)
//generates random numbers in x[N] between [a,b)
{
  if (N < 0)
  {
    std::cout << "Error in NPME_RandomNumberArrayBlock. ";
    std::cout << "N = " << N << " < 0\n";
    exit(0);
  }

  // Initializing the streams with R250 random number generator with seed = 1
  VSLStreamStatePtr stream;
  vslNewStream (&stream, VSL_BRNG_R250, (MKL_UINT) seed);

  // Generating 
  vdRngUniform (VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, N, x, a, b);

  // Deleting the streams 
  vslDeleteStream( &stream );
}

void NPME_RandomNumberArray (const size_t N, double *x, const double a, 
  const double b, const size_t seed0)
//generates random numbers in x[N] between [a,b)
{
  //the max size of int is 2,147,483,647
  //assume a number less would be the maximum block size 
  //which can be passed to vdRngUniform
  const size_t blockSize  = 2000000000;
  const size_t r          = N%blockSize;
  const size_t n          = (N-r)/blockSize;
  
  for (size_t i = 0; i < n; i++)
  {
    const size_t startIndex = i*blockSize;
    NPME_RandomNumberArrayBlock ((int) blockSize, &x[startIndex], 
      a, b, i+seed0);
  }

  if (r > 0)
  {
    const size_t startIndex = n*blockSize;
    NPME_RandomNumberArrayBlock ((int) r, &x[startIndex], a, b, n+seed0);
  }
}

void NPME_RandomNumberArray (const size_t N, _Complex double *x, const double a, 
  const double b, const size_t seed0)
{
  return NPME_RandomNumberArray (2*N, (double *) x, a, b, seed0);
}

void NPME_Transpose (const size_t M, const size_t N, double *A)
//A is MxN -> At is NxM
{
  const double alpha = 1.0;

  //R = row first
  //T = transpose
  //M = rows
  //N = col
  mkl_dimatcopy ('R', 'T', M, N, alpha, A, N, M);

  //lda = N = col  if row-first
  //ldb = M = rows if row-first and transpose (or adjoint)
}

void NPME_TransposeComplex (const size_t M, const size_t N, _Complex double *A)
//A is MxN -> At is NxM
{
  MKL_Complex16 alpha;
  alpha.real = 1.0;
  alpha.imag = 0.0;

  //R = row first
  //T = transpose
  //M = rows
  //N = col
  mkl_zimatcopy ('R', 'T', M, N, alpha, (MKL_Complex16 *) A, N, M);

  //lda = N = col  if row-first
  //ldb = M = rows if row-first and transpose (or adjoint)
}

void NPME_AdjointComplex (const size_t M, const size_t N, _Complex double *A)
//A is MxN -> At is NxM
{
  MKL_Complex16 alpha;
  alpha.real = 1.0;
  alpha.imag = 0.0;

  //R = row first
  //C = conjugate transpose
  //M = rows
  //N = col
  mkl_zimatcopy ('R', 'C', M, N, alpha, (MKL_Complex16 *) A, N, M);

  //lda = N = col  if row-first
  //ldb = M = rows if row-first and transpose (or adjoint)
}

bool NPME_SolveLinearSystem (const size_t N, const size_t M, double *A,
  double *b, double *x)
//A[N][N]*x[N][M] = b[N][M]
{
  memcpy(&x[0], &b[0], N*M*sizeof(double));

  std::vector<MKL_INT> ipiv(N);
  MKL_INT info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, (MKL_INT) N, (MKL_INT) M, A, 
    N, &ipiv[0], x, M);

  if (info > 0) 
  {
    std::cout << "Error in NPME_SolveLinearSystem.  Singular matrix.\n";
    return false;
  }

  return true;
}

bool NPME_SolveLinearSystem (const size_t N, const size_t M, double *A,
  double *x)
//A[N][N]*x[N][M] = b[N][M]
//x[N][M] is initialized with b[N][M]
{
  std::vector<MKL_INT> ipiv(N);
  MKL_INT info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, (MKL_INT) N, (MKL_INT) M, A, 
    N, &ipiv[0], x, M);

  if (info > 0) 
  {
    std::cout << "Error in NPME_SolveLinearSystem.  Singular matrix.\n";
    return false;
  }

  return true;
}

bool NPME_SolveLinearSystemComplex (const size_t N, const size_t M, 
  _Complex double *A, _Complex double *b, _Complex double *x)
//A[N][N]*x[N][M] = b[N][M]
{
  memcpy(&x[0], &b[0], N*M*sizeof(_Complex double));

  std::vector<MKL_INT> ipiv(N);
  MKL_INT info = LAPACKE_zgesv( LAPACK_ROW_MAJOR, (MKL_INT) N, 
    (MKL_INT) M, (MKL_Complex16 *) A, 
    N, &ipiv[0], (MKL_Complex16 *) x, M);

  if (info > 0) 
  {
    std::cout << "Error in NPME_SolveLinearSystem.  Singular matrix.\n";
    return false;
  }

  return true;
}




bool NPME_SolveLinearSystemComplex (const size_t N, const size_t M, 
  _Complex double *A, _Complex double *x)
//A[N][N]*x[N][M] = b[N][M]
//x[N][M] is initialized with b[N][M]
{
  std::vector<MKL_INT> ipiv(N);
  MKL_INT info = LAPACKE_zgesv( LAPACK_ROW_MAJOR, (MKL_INT) N, 
    (MKL_INT) M, (MKL_Complex16 *) A, 
    N, &ipiv[0], (MKL_Complex16 *) x, M);

  if (info > 0) 
  {
    std::cout << "Error in NPME_SolveLinearSystem.  Singular matrix.\n";
    return false;
  }

  return true;
}


//*****************************************************************************
//*****************************************************************************
//***************Real Double Precision Matrix Multiply*************************
//*****************************************************************************
//*****************************************************************************
//basic algorithm:
//C = alpha*op(A)*op(B) + beta*C
//op = nothing, transpose, adjoint

//1) matrix multiply
void NPME_MatrixMatrixProd (double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha, const double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = alpha*A[M][K]*B[K][N] + beta*C[M][N]
//input:  A[M][Alda], B[K][Blda]
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixMatrixProd\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[M][K] is contained in A[M][Alda], K <= Alda
  if (K > Alda)
  {
    std::cout << "Error in NPME_MatrixMatrixProd\n";
    std::cout << "K = " << K << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[K][N] is contained in B[K][Blda], N <= Blda
  if (N > Blda)
  {
    std::cout << "Error in NPME_MatrixMatrixProd\n";
    std::cout << "N = " << N << " > " << Blda << "Blda\n";
    exit(0);
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, 
    A, Alda, 
    B, Blda, beta, 
    C, Clda);
}

void NPME_MatrixMatrixProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha, const double beta)
{
  NPME_MatrixMatrixProd (C, A, B, M, N, K, alpha, beta, K, N, N);
}

void NPME_MatrixMatrixTransposeProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha, const double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = alpha*A[M][K]*(B[N][K])^T + beta*C[M][N] (multiplies A*B^T)
//input:  A[M][Alda], B[N][Blda]  (K <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixMatrixTransposeProd\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[M][K] is contained in A[M][Alda], K <= Alda
  if (K > Alda)
  {
    std::cout << "Error in NPME_MatrixMatrixTransposeProd\n";
    std::cout << "K = " << K << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[N][K] is contained in B[N][Blda], K <= Blda
  if (K > Blda)
  {
    std::cout << "Error in NPME_MatrixMatrixTransposeProd\n";
    std::cout << "K = " << K << " > " << Blda << "Blda\n";
    exit(0);
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
      M, N, K, alpha, 
      A, Alda, 
      B, Blda, beta, 
      C, Clda);
}
void NPME_MatrixMatrixTransposeProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha, const double beta)
{
  NPME_MatrixMatrixTransposeProd (C, A, B, M, N, K, alpha, beta,
    K, K, N);
}

void NPME_MatrixTransposeMatrixProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha, const double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = alpha*(A[K][M])^T*B[K][N] + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[K][N]  (M <= Alda, N <= Blda, N <= Clda)
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixProd\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[K][M] is contained in A[K][Alda], M <= Alda
  if (M > Alda)
  {
    std::cout << "Error in NPME_MatrixMatrixTransposeProd\n";
    std::cout << "M = " << M << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[K][N] is contained in B[K][Blda], N <= Blda
  if (N > Blda)
  {
    std::cout << "Error in NPME_MatrixMatrixProd\n";
    std::cout << "N = " << N << " > " << Blda << "Blda\n";
    exit(0);
  }


  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,  
      M, N, K, alpha, 
      A, Alda, 
      B, Blda, beta, 
      C, Clda);
}
void NPME_MatrixTransposeMatrixProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha, const double beta)
//C[M][N] = alpha*(A[K][M])^T*B[K][N] + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[K][Blda]  (M <= Alda, N <= Blda, N <= Clda)
//output: C[M][Clda]
{
  return NPME_MatrixTransposeMatrixProd (C, A, B, M, N, K, alpha, beta,
      M, N, N);
}


void NPME_MatrixTransposeMatrixTransposeProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha, const double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixTransposeProd\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[K][M] is contained in A[K][Alda], M <= Alda
  if (M > Alda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixTransposeProd\n";
    std::cout << "M = " << M << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[N][K] is contained in B[N][Blda], K <= Blda
  if (K > Blda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixTransposeProd\n";
    std::cout << "K = " << K << " > " << Blda << "Blda\n";
    exit(0);
  }

  cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,  
      M, N, K, alpha, 
      A, Alda, 
      B, Blda, beta, 
      C, Clda);
}
void NPME_MatrixTransposeMatrixTransposeProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha, const double beta)
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  return NPME_MatrixTransposeMatrixTransposeProd (C, A, B, M, N, K, alpha, beta,
      M, K, N);
}


//*****************************************************************************
//*****************************************************************************
//***************Complex Double Precision Matrix Multiply**********************
//*****************************************************************************
//*****************************************************************************
//basic algorithm:
//C = alpha*op(A)*op(B) + beta*C
//op = nothing, transpose, adjoint


//1) matrix multiply
void NPME_MatrixMatrixProdComplex (_Complex double *C, 
  const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = alpha*A[M][K]*B[K][N] + beta*C[M][N]
//input:  A[M][Alda], B[K][Blda]
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixMatrixProdComplex\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[M][K] is contained in A[M][Alda], K <= Alda
  if (K > Alda)
  {
    std::cout << "Error in NPME_MatrixMatrixProdComplex\n";
    std::cout << "K = " << K << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[K][N] is contained in B[K][Blda], N <= Blda
  if (N > Blda)
  {
    std::cout << "Error in NPME_MatrixMatrixProdComplex\n";
    std::cout << "N = " << N << " > " << Blda << "Blda\n";
    exit(0);
  }

  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 
    (const void *) &alpha,
    (const void *) A, Alda, 
    (const void *) B, Blda, (const void *) &beta, 
          (void *) C, Clda);
}

void NPME_MatrixMatrixProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta)
{
  NPME_MatrixMatrixProdComplex (C, A, B, M, N, K, alpha, beta, K, N, N);
}



void NPME_MatrixMatrixTransposeProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = alpha*A[M][K]*(B[N][K])^T + beta*C[M][N] (multiplies A*B^T)
//input:  A[M][Alda], B[N][Blda]  (K <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixMatrixTransposeProdComplex\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[M][K] is contained in A[M][Alda], K <= Alda
  if (K > Alda)
  {
    std::cout << "Error in NPME_MatrixMatrixTransposeProdComplex\n";
    std::cout << "K = " << K << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[N][K] is contained in B[N][Blda], K <= Blda
  if (K > Blda)
  {
    std::cout << "Error in NPME_MatrixMatrixTransposeProdComplex\n";
    std::cout << "K = " << K << " > " << Blda << "Blda\n";
    exit(0);
  }

  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 
    (const void *) &alpha,
    (const void *) A, Alda, 
    (const void *) B, Blda, (const void *) &beta, 
          (void *) C, Clda);
}
void NPME_MatrixMatrixTransposeProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta)
{
  NPME_MatrixMatrixTransposeProdComplex (C, A, B, M, N, K, alpha, beta,
    K, K, N);
}


void NPME_MatrixTransposeMatrixProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = alpha*(A[K][M])^T*B[K][N] + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[K][N]  (M <= Alda, N <= Blda, N <= Clda)
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixProdComplex\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[K][M] is contained in A[K][Alda], M <= Alda
  if (M > Alda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixProdComplex\n";
    std::cout << "M = " << M << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[K][N] is contained in B[K][Blda], N <= Blda
  if (N > Blda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixProdComplex\n";
    std::cout << "N = " << N << " > " << Blda << "Blda\n";
    exit(0);
  }


  cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 
    (const void *) &alpha,
    (const void *) A, Alda, 
    (const void *) B, Blda, (const void *) &beta, 
          (void *) C, Clda);
}
void NPME_MatrixTransposeMatrixProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta)
//C[M][N] = alpha*(A[K][M])^T*B[K][N] + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[K][Blda]  (M <= Alda, N <= Blda, N <= Clda)
//output: C[M][Clda]
{
  return NPME_MatrixTransposeMatrixProdComplex (C, A, B, M, N, K, alpha, beta,
      M, N, N);
}


void NPME_MatrixTransposeMatrixTransposeProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixTransposeProdComplex\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[K][M] is contained in A[K][Alda], M <= Alda
  if (M > Alda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixTransposeProdComplex\n";
    std::cout << "M = " << M << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[N][K] is contained in B[N][Blda], K <= Blda
  if (K > Blda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixTransposeProdComplex\n";
    std::cout << "K = " << K << " > " << Blda << "Blda\n";
    exit(0);
  }

  cblas_zgemm(CblasRowMajor, CblasTrans, CblasTrans,  M, N, K, 
    (const void *) &alpha,
    (const void *) A, Alda, 
    (const void *) B, Blda, (const void *) &beta, 
          (void *) C, Clda);
}
void NPME_MatrixTransposeMatrixTransposeProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta)
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  return NPME_MatrixTransposeMatrixTransposeProdComplex (C, A, B, M, N, K, 
      alpha, beta, M, K, N);
}


void NPME_MatrixMatrixAdjointProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = alpha*A[M][K]*(B[N][K])^T + beta*C[M][N] (multiplies A*B^T)
//input:  A[M][Alda], B[N][Blda]  (K <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixMatrixAdjointProdComplex\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[M][K] is contained in A[M][Alda], K <= Alda
  if (K > Alda)
  {
    std::cout << "Error in NPME_MatrixMatrixAdjointProdComplex\n";
    std::cout << "K = " << K << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[N][K] is contained in B[N][Blda], K <= Blda
  if (K > Blda)
  {
    std::cout << "Error in NPME_MatrixMatrixAdjointProdComplex\n";
    std::cout << "K = " << K << " > " << Blda << "Blda\n";
    exit(0);
  }

  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, M, N, K, 
    (const void *) &alpha,
    (const void *) A, Alda, 
    (const void *) B, Blda, (const void *) &beta, 
          (void *) C, Clda);
}
void NPME_MatrixMatrixAdjointProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta)
{
  NPME_MatrixMatrixAdjointProdComplex (C, A, B, M, N, K, alpha, beta,
    K, K, N);
}



void NPME_MatrixAdjointMatrixProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = alpha*(A[K][M])^T*B[K][N] + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[K][N]  (M <= Alda, N <= Blda, N <= Clda)
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixAdjointMatrixProdComplex\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[K][M] is contained in A[K][Alda], M <= Alda
  if (M > Alda)
  {
    std::cout << "Error in NPME_MatrixAdjointMatrixProdComplex\n";
    std::cout << "M = " << M << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[K][N] is contained in B[K][Blda], N <= Blda
  if (N > Blda)
  {
    std::cout << "Error in NPME_MatrixAdjointMatrixProdComplex\n";
    std::cout << "N = " << N << " > " << Blda << "Blda\n";
    exit(0);
  }


  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans, M, N, K, 
    (const void *) &alpha,
    (const void *) A, Alda, 
    (const void *) B, Blda, (const void *) &beta, 
          (void *) C, Clda);
}
void NPME_MatrixAdjointMatrixProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta)
//C[M][N] = alpha*(A[K][M])^T*B[K][N] + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[K][Blda]  (M <= Alda, N <= Blda, N <= Clda)
//output: C[M][Clda]
{
  return NPME_MatrixAdjointMatrixProdComplex (C, A, B, M, N, K, alpha, beta,
      M, N, N);
}


void NPME_MatrixAdjointMatrixAdjointProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixAdjointMatrixAdjointProdComplex\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[K][M] is contained in A[K][Alda], M <= Alda
  if (M > Alda)
  {
    std::cout << "Error in NPME_MatrixAdjointMatrixAdjointProdComplex\n";
    std::cout << "M = " << M << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[N][K] is contained in B[N][Blda], K <= Blda
  if (K > Blda)
  {
    std::cout << "Error in NPME_MatrixAdjointMatrixAdjointProdComplex\n";
    std::cout << "K = " << K << " > " << Blda << "Blda\n";
    exit(0);
  }

  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasConjTrans,  M, N, K, 
    (const void *) &alpha,
    (const void *) A, Alda, 
    (const void *) B, Blda, (const void *) &beta, 
          (void *) C, Clda);
}
void NPME_MatrixAdjointMatrixAdjointProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta)
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  return NPME_MatrixAdjointMatrixAdjointProdComplex (C, A, B, M, N, K, 
      alpha, beta, M, K, N);
}


void NPME_MatrixAdjointMatrixTransposeProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixAdjointMatrixTransposeProdComplex\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[K][M] is contained in A[K][Alda], M <= Alda
  if (M > Alda)
  {
    std::cout << "Error in NPME_MatrixAdjointMatrixTransposeProdComplex\n";
    std::cout << "M = " << M << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[N][K] is contained in B[N][Blda], K <= Blda
  if (K > Blda)
  {
    std::cout << "Error in NPME_MatrixAdjointMatrixTransposeProdComplex\n";
    std::cout << "K = " << K << " > " << Blda << "Blda\n";
    exit(0);
  }

  cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasTrans,  M, N, K, 
    (const void *) &alpha,
    (const void *) A, Alda, 
    (const void *) B, Blda, (const void *) &beta, 
          (void *) C, Clda);
}
void NPME_MatrixAdjointMatrixTransposeProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta)
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  return NPME_MatrixAdjointMatrixTransposeProdComplex (C, A, B, M, N, K, 
      alpha, beta, M, K, N);
}


void NPME_MatrixTransposeMatrixAdjointProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda)
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  //C[M][N] is contained in C[M][Clda], N <= Clda
  if (N > Clda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixAdjointProdComplex\n";
    std::cout << "N = " << N << " > " << Clda << "Clda\n";
    exit(0);
  }

  //A[K][M] is contained in A[K][Alda], M <= Alda
  if (M > Alda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixAdjointProdComplex\n";
    std::cout << "M = " << M << " > " << Alda << "Alda\n";
    exit(0);
  }

  //B[N][K] is contained in B[N][Blda], K <= Blda
  if (K > Blda)
  {
    std::cout << "Error in NPME_MatrixTransposeMatrixAdjointProdComplex\n";
    std::cout << "K = " << K << " > " << Blda << "Blda\n";
    exit(0);
  }

  cblas_zgemm(CblasRowMajor, CblasTrans,  CblasConjTrans, M, N, K, 
    (const void *) &alpha,
    (const void *) A, Alda, 
    (const void *) B, Blda, (const void *) &beta, 
          (void *) C, Clda);
}
void NPME_MatrixTransposeMatrixAdjointProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta)
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
{
  return NPME_MatrixTransposeMatrixAdjointProdComplex (C, A, B, M, N, K, 
      alpha, beta, M, K, N);
}

//*****************************************************************************
//*****************************************************************************
//*******************************  FFT   **************************************
//*****************************************************************************
//*****************************************************************************


void NPME_1D_FFT_NoNorm (_Complex double *F, long int N1)
{
  DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
  MKL_LONG status;

  status = DftiCreateDescriptor(&my_desc1_handle, 
            DFTI_DOUBLE, DFTI_COMPLEX, 1, N1);
  status = DftiCommitDescriptor(my_desc1_handle);
  status = DftiComputeForward(my_desc1_handle, &F[0]);
  status = DftiFreeDescriptor(&my_desc1_handle);
}

void NPME_1D_FFT (_Complex double *F, long int N1)
{
  DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
  MKL_LONG status;

  status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, 
            DFTI_COMPLEX, 1, N1);
  status = DftiCommitDescriptor(my_desc1_handle);
  status = DftiComputeForward(my_desc1_handle, &F[0]);
  status = DftiFreeDescriptor(&my_desc1_handle);

  const double C = (double) 1.0/N1;
  for (long int i = 0; i < N1; i++)
    F[i] *= C;
}

void NPME_1D_FFT_Inverse (_Complex double *f, long int N1)
{
  DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
  MKL_LONG status;

  status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, 
              DFTI_COMPLEX, 1, N1);
  status = DftiCommitDescriptor(my_desc1_handle);
  status = DftiComputeBackward(my_desc1_handle, &f[0]);
  status = DftiFreeDescriptor(&my_desc1_handle);
}





void NPME_2D_FFT_NoNorm (_Complex double *F, long int N1, long int N2)
{
  DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
  MKL_LONG status;
  MKL_LONG dim_sizes[2] = {N1, N2};

  status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, 
            DFTI_COMPLEX, 2, dim_sizes);
  status = DftiCommitDescriptor(my_desc1_handle);
  status = DftiComputeForward(my_desc1_handle, &F[0]);
  status = DftiFreeDescriptor(&my_desc1_handle);
}

void NPME_2D_FFT (_Complex double *F, long int N1, long int N2)
{
  DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
  MKL_LONG status;
  MKL_LONG dim_sizes[2] = {N1, N2};

  status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, 
              DFTI_COMPLEX, 2, dim_sizes);
  status = DftiCommitDescriptor(my_desc1_handle);
  status = DftiComputeForward(my_desc1_handle, &F[0]);
  status = DftiFreeDescriptor(&my_desc1_handle);


  const double C = (double) 1.0/(N1*N2);
  for (long int i = 0; i < N1*N2; i++)
    F[i] *= C;
}

void NPME_2D_FFT_Inverse (_Complex double *f, long int N1, long int N2)
{
  DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
  MKL_LONG status;
  MKL_LONG dim_sizes[2] = {N1, N2};

  status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, 
            DFTI_COMPLEX, 2, dim_sizes);
  status = DftiCommitDescriptor(my_desc1_handle);
  status = DftiComputeBackward(my_desc1_handle, &f[0]);
  status = DftiFreeDescriptor(&my_desc1_handle);
}





void NPME_3D_FFT_NoNorm (_Complex double *F, long int N1, 
  long int N2, long int N3)
{
  DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
  MKL_LONG status;
  MKL_LONG dim_sizes[3] = {N1, N2, N3};

  status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, 
            DFTI_COMPLEX, 3, dim_sizes);
  status = DftiCommitDescriptor(my_desc1_handle);
  status = DftiComputeForward(my_desc1_handle, &F[0]);
  status = DftiFreeDescriptor(&my_desc1_handle);
}
void NPME_3D_FFT (_Complex double *F, long int N1, 
  long int N2, long int N3)
{
  DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
  MKL_LONG status;
  MKL_LONG dim_sizes[3] = {N1, N2, N3};

  status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, 
              DFTI_COMPLEX, 3, dim_sizes);
  status = DftiCommitDescriptor(my_desc1_handle);
  status = DftiComputeForward(my_desc1_handle, &F[0]);
  status = DftiFreeDescriptor(&my_desc1_handle);

  const double C = (double) 1.0/(N1*N2*N3);
  for (long int i = 0; i < N1*N2*N3; i++)
    F[i] *= C;
}

void NPME_3D_FFT_Inverse (_Complex double *f, long int N1, 
  long int N2, long int N3)
{
  DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
  MKL_LONG status;
  MKL_LONG dim_sizes[3] = {N1, N2, N3};

  status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, 
              DFTI_COMPLEX, 3, dim_sizes);
  status = DftiCommitDescriptor(my_desc1_handle);
  status = DftiComputeBackward(my_desc1_handle, &f[0]);
  status = DftiFreeDescriptor(&my_desc1_handle);

}

void NPME_3D_FFT_NoNorm_MultipleInput (_Complex double *F, long int K, 
  long int N1, long int N2, long int N3)
//K         = number of inputs
//N1,N2,N3  = dimensions of FFT
//F[K*N1*N2*N3] = F[K][N1][N2][N3]
{
  DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
  MKL_LONG status;
  MKL_LONG dim_sizes[3] = {N1, N2, N3};

  status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, 
              DFTI_COMPLEX, 3, dim_sizes);
  status = DftiSetValue(my_desc1_handle, DFTI_NUMBER_OF_TRANSFORMS, K);
  status = DftiSetValue(my_desc1_handle, DFTI_INPUT_DISTANCE, N1*N2*N3);
  status = DftiCommitDescriptor(my_desc1_handle);
  status = DftiComputeForward(my_desc1_handle, &F[0]);
  status = DftiFreeDescriptor(&my_desc1_handle);
}

void NPME_3D_FFT_Inverse_MultipleInput (_Complex double *f, long int K, 
  long int N1, long int N2, long int N3)
//K         = number of inputs
//N1,N2,N3  = dimensions of FFT
//F[K*N1*N2*N3] = F[K][N1][N2][N3]
{
  DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
  MKL_LONG status;
  MKL_LONG dim_sizes[3] = {N1, N2, N3};

  status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE, 
              DFTI_COMPLEX, 3, dim_sizes);
  status = DftiSetValue(my_desc1_handle, DFTI_NUMBER_OF_TRANSFORMS, K);
  status = DftiSetValue(my_desc1_handle, DFTI_INPUT_DISTANCE, N1*N2*N3);
  status = DftiCommitDescriptor(my_desc1_handle);
  status = DftiComputeBackward(my_desc1_handle, &f[0]);
  status = DftiFreeDescriptor(&my_desc1_handle);

}



}//end NPME_Library



