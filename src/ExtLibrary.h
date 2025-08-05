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

#ifndef NPME_EXT_LIBRARY_H
#define NPME_EXT_LIBRARY_H

#include "mkl.h"


//The following functions are an interface to Intel's Math Kernel Library (MKL)
//The number of threads can be set BEFORE calling the following functions
//void mkl_set_num_threads (int nt);

namespace NPME_Library
{

void *NPME_malloc (size_t size, size_t alignment);

void NPME_free (void* ptr);



void NPME_RandomNumberArray (const size_t N, double *x, const double a, 
  const double b, const size_t seed = 1);
void NPME_RandomNumberArray (const size_t N, _Complex double *x, const double a, 
  const double b, const size_t seed0);
//generates random numbers in x[N] between [a,b)

void NPME_Transpose (const size_t M, const size_t N, double *A);
//A is MxN -> At is NxM

void NPME_TransposeComplex (const size_t M, const size_t N, _Complex double *A);
//A is MxN -> At is NxM

void NPME_AdjointComplex (const size_t M, const size_t N, _Complex double *A);
//A is MxN -> At is NxM

bool NPME_SolveLinearSystem (const size_t N, const size_t M, double *A,
  double *b, double *x);
//A[N][N]*x[N][M] = b[N][M]

bool NPME_SolveLinearSystem (const size_t N, const size_t M, double *A,
  double *x);
//A[N][N]*x[N][M] = b[N][M]
//x[N][M] is initialized with b[N][M]

bool NPME_SolveLinearSystemComplex (const size_t N, const size_t M, 
  _Complex double *A, _Complex double *b, _Complex double *x);
//A[N][N]*x[N][M] = b[N][M]

bool NPME_SolveLinearSystemComplex (const size_t N, const size_t M, 
  _Complex double *A, _Complex double *x);
//A[N][N]*x[N][M] = b[N][M]
//x[N][M] is initialized with b[N][M]


//*****************************************************************************
//*****************************************************************************
//****************************Matrix Multiply**********************************
//*****************************************************************************
//*****************************************************************************

void NPME_MatrixMatrixProd (double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha, const double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = alpha*A[M][K]*B[K][N] + beta*C[M][N]
//input:  A[M][Alda], B[K][Blda]
//output: C[M][Clda]
void NPME_MatrixMatrixProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha = 1.0, const double beta = 0.0);


void NPME_MatrixMatrixTransposeProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha, const double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = alpha*A[M][K]*(B[N][K])^T + beta*C[M][N] (multiplies A*B^T)
//input:  A[M][Alda], B[Blda][K]  (K <= Alda, N <= Blda, N <= Clda)
//output: C[M][Clda]
void NPME_MatrixMatrixTransposeProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha = 1.0, const double beta = 0.0);


void NPME_MatrixTransposeMatrixProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha, const double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = alpha*(A[K][M])^T*B[K][N] + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[K][N]  (M <= Alda, N <= Blda, N <= Clda)
//output: C[M][Clda]
void NPME_MatrixTransposeMatrixProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha = 1.0, const double beta = 0.0);
//C[M][N] = alpha*(A[K][M])^T*B[K][N] + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[K][N]  (M <= Alda, N <= Blda, N <= Clda)
//output: C[M][Clda]


void NPME_MatrixTransposeMatrixTransposeProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha, const double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
void NPME_MatrixTransposeMatrixTransposeProd (
  double *C, const double *A, const double *B, 
  const size_t M, const size_t N, const size_t K, 
  const double alpha = 1.0, const double beta = 0.0);
//C[M][N] = alpha*(A[K][M])^T*B[K][N] + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[K][Blda]  (M <= Alda, N <= Blda, N <= Clda)
//output: C[M][Clda]

void NPME_MatrixMatrixProdComplex (_Complex double *C, 
  const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = alpha*A[M][K]*B[K][N] + beta*C[M][N]
//input:  A[M][Alda], B[K][Blda]
//output: C[M][Clda]
void NPME_MatrixMatrixProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha = 1.0, const _Complex double beta = 0.0);

void NPME_MatrixMatrixTransposeProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = alpha*A[M][K]*(B[N][K])^T + beta*C[M][N] (multiplies A*B^T)
//input:  A[M][Alda], B[N][Blda]  (K <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
void NPME_MatrixMatrixTransposeProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha = 1.0, const _Complex double beta = 0.0);

void NPME_MatrixTransposeMatrixProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = alpha*(A[K][M])^T*B[K][N] + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[K][N]  (M <= Alda, N <= Blda, N <= Clda)
//output: C[M][Clda]
void NPME_MatrixTransposeMatrixProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha = 1.0, const _Complex double beta = 0.0);


void NPME_MatrixTransposeMatrixTransposeProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
void NPME_MatrixTransposeMatrixTransposeProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha = 1.0, const _Complex double beta = 0.0);


void NPME_MatrixMatrixAdjointProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = alpha*A[M][K]*(B[N][K])^T + beta*C[M][N] (multiplies A*B^T)
//input:  A[M][Alda], B[N][Blda]  (K <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
void NPME_MatrixMatrixAdjointProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha = 1.0, const _Complex double beta = 0.0);

void NPME_MatrixAdjointMatrixProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = alpha*(A[K][M])^T*B[K][N] + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[K][N]  (M <= Alda, N <= Blda, N <= Clda)
//output: C[M][Clda]
void NPME_MatrixAdjointMatrixProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha = 1.0, const _Complex double beta = 0.0);


void NPME_MatrixAdjointMatrixAdjointProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
void NPME_MatrixAdjointMatrixAdjointProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha = 1.0, const _Complex double beta = 0.0);

void NPME_MatrixAdjointMatrixTransposeProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
void NPME_MatrixAdjointMatrixTransposeProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha = 1.0, const _Complex double beta = 0.0);

void NPME_MatrixTransposeMatrixAdjointProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha, const _Complex double beta,
  const size_t Alda, const size_t Blda, const size_t Clda);
//C[M][N] = (A[K][M])^T*(B[N][K])^T (multiplies A^T*B^T)
//C[M][N] = alpha*(A[K][M])^T*(B[N][K])^T + beta*C[M][N] (multiplies A^T*B)
//input:  A[K][Alda], B[N][Blda]  (M <= Alda, K <= Blda, N <= Clda)
//output: C[M][Clda]
void NPME_MatrixTransposeMatrixAdjointProdComplex (
  _Complex double *C, const _Complex double *A, const _Complex double *B, 
  const size_t M, const size_t N, const size_t K, 
  const _Complex double alpha = 1.0, const _Complex double beta = 0.0);


//*****************************************************************************
//*****************************************************************************
//*******************************  FFT   **************************************
//*****************************************************************************
//*****************************************************************************

void NPME_1D_FFT_NoNorm  (_Complex double *F, long int N1);
void NPME_1D_FFT_Inverse (_Complex double *f, long int N1);
void NPME_1D_FFT         (_Complex double *F, long int N1);

void NPME_2D_FFT_NoNorm  (_Complex double *F, long int N1, long int N2);
void NPME_2D_FFT_Inverse (_Complex double *f, long int N1, long int N2);
void NPME_2D_FFT         (_Complex double *F, long int N1, long int N2);


void NPME_3D_FFT_NoNorm (_Complex double *F, long int N1, 
  long int N2, long int N3);

void NPME_3D_FFT_Inverse (_Complex double *f, long int N1, 
  long int N2, long int N3);


void NPME_3D_FFT_NoNorm_MultipleInput (_Complex double *F, long int K, 
  long int N1, long int N2, long int N3);
void NPME_3D_FFT_Inverse_MultipleInput (_Complex double *f, long int K, 
  long int N1, long int N2, long int N3);
//K         = number of inputs
//N1,N2,N3  = dimensions of FFT
//F[K*N1*N2*N3] = F[K][N1][N2][N3]


}//end namespace NPME_Library



#endif // NPME_EXT_LIBRARY_H



