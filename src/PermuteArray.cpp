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
#include <cstdio>

#include <iostream> 
#include <vector>


#include "Constant.h"
#include "PermuteArray.h"


#define NPME_PERMUTE_ARRAY_CHECK_ALL 1

namespace NPME_Library
{
//******************************************************************************
//******************************************************************************
//********************Permutations Overwriting Array****************************
//******************************************************************************
//******************************************************************************

bool NPME_GetNextPstart (size_t& PindexStart,
  const size_t PindexStart0, const size_t N, const bool *Pmask)
{
  for (size_t i = PindexStart0 + 1; i < N; i++)
    if (Pmask[i] == 0)
    {
      PindexStart = i;
      return true;
    }

  return false;
}
bool NPME_GetKcycle (size_t& KcycleSize,
  const size_t PindexStart, const size_t KindexStart,
  const size_t N, bool *Pmask, size_t *K, const size_t *P)
{
  #if NPME_PERMUTE_ARRAY_CHECK_ALL
  {
    if (Pmask[PindexStart] != 0)
    {
      std::cout << "Error in NPME_GetKcycle.\n";
      std::cout << "Pmask[PindexStart] = 1, PindexStart = ";
      std::cout << PindexStart << "\n";
      exit(0);
    }
  }
  #endif


  KcycleSize          = 1;
  size_t Kindex       = KindexStart;
  K[Kindex]           = PindexStart;
  Pmask[PindexStart]  = 1;
  Kindex++;

  for (;;)
  {
    size_t a = P[ K[Kindex-1] ];
    if (a == PindexStart)
      break;

    //printf("K[Kindex-1] = %lu a = %lu\n", K[Kindex-1], a);

    #if NPME_PERMUTE_ARRAY_CHECK_ALL
    {
      if (Kindex >= N)
      {
        std::cout << "Error in NPME_GetKcycle.\n";
        std::cout << "Kindex = " << Kindex << " >= " << N << "\n";
        exit(0);
      }
      if (Pmask[a] == 1)
      {
        std::cout << "Error in NPME_GetKcycle.\n";
        std::cout << "a = " << a << "Pmask[a] = 1\n";
        exit(0);
      }
    }
    #endif


    K[Kindex] = a;
    Pmask[a]  = 1;
    Kindex++;
    KcycleSize++;
  }

  return true;
}


NPME_Kcycle::NPME_Kcycle (const NPME_Kcycle& rhs)
{
  _N            = rhs._N;
  _nKcycle      = rhs._nKcycle;
  _KcycleSize   = rhs._KcycleSize;
  _KcycleStart  = rhs._KcycleStart;
  _K            = rhs._K;
}
NPME_Kcycle& NPME_Kcycle::operator= (const NPME_Kcycle& rhs)
{
  if (this != &rhs)
  {
    _N            = rhs._N;
    _nKcycle      = rhs._nKcycle;
    _KcycleSize   = rhs._KcycleSize;
    _KcycleStart  = rhs._KcycleStart;
    _K            = rhs._K;
  }

  return *this;
}


bool NPME_Kcycle::SetPermuteArray (const size_t N, const size_t *P)
{
  _N = N;
  _K.resize(N);
  _KcycleSize.clear();
  _KcycleStart.clear();

  bool *Pmask = new bool [N];
  for (size_t i = 0; i < N; i++)
    Pmask[i] = 0;

  _nKcycle = 0;
  size_t KcycleSize;
  size_t PindexStart = 0;
  size_t KindexStart = 0;

  if (!NPME_GetKcycle (KcycleSize, PindexStart, 
    KindexStart, N, Pmask, &_K[0], P))
  {
    std::cout << "Error in NPME_Kcycle::SetPermuteArray.\n";
    std::cout << "NPME_GetKcycle failed\n";
    return false;
  }

  _KcycleSize.push_back(KcycleSize);
  _KcycleStart.push_back(KindexStart);

  KindexStart += KcycleSize;
  _nKcycle++;


  for (;;)
  {
    if (NPME_GetNextPstart (PindexStart, PindexStart, N, Pmask))
    {
      if (!NPME_GetKcycle (KcycleSize, PindexStart, 
          KindexStart, N, Pmask, &_K[0], P))
      {
        std::cout << "Error in NPME_Kcycle::SetPermuteArray.\n";
        std::cout << "NPME_GetKcycle failed\n";
        return false;
      }

      _KcycleSize.push_back(KcycleSize);
      _KcycleStart.push_back(KindexStart);

      KindexStart += KcycleSize;
      _nKcycle++;
    }
    else
      break;
  }


  delete [] Pmask;

  return true;
}

void NPME_Kcycle::PrintClass (std::ostream& os) const
{
  os << "_N       = " << _N << "\n";
  os << "_nKcycle = " << _nKcycle << "\n";

  for (size_t i = 0; i < _nKcycle; i++)
  {
    char str[2000];
    sprintf(str, "  cycle %lu size = %lu start = %lu\n", 
      i, _KcycleSize[i], _KcycleStart[i]);
    os << str;

    os << "    ";
    for (size_t j = _KcycleStart[i]; j < _KcycleStart[i]+_KcycleSize[i]; j++)
    {
      os.width(4);
      os << _K[j] << " ";
    }
    os << "\n";
  }
}

void NPME_Kcycle::PrintClassShort (std::ostream& os) const
{
  os << "_N       = " << _N << "\n";
  os << "_nKcycle = " << _nKcycle << "\n";
  for (size_t i = 0; i < _nKcycle; i++)
  {
    char str[2000];
    sprintf(str, "  cycle %lu size = %lu start = %lu\n", 
      i, _KcycleSize[i], _KcycleStart[i]);
    os << str;
  }
}

bool NPME_Kcycle::CheckClass () const
{
  if (_N == 0)
  {
    std::cout << "Error in NPME_Kcycle::CheckClass.\n";
    std::cout << "N = " << _N << "  class is not initialized\n";
    return false;
  }
  if (_nKcycle == 0)
  {
    std::cout << "Error in NPME_Kcycle::CheckClass.\n";
    std::cout << "N = " << _N << "  class is not initialized\n";
    return false;
  }

  size_t index = 0;
  for (size_t i = 0; i < _nKcycle; i++)
  {
    if (index != _KcycleStart[i])
    {
      std::cout << "Error in NPME_Kcycle::CheckClass.\n";
      std::cout << "index = " << index;
      std::cout << " != " << _KcycleStart[i] << "_KcycleStart[i].\n";
      return false;
    }

    index += _KcycleSize[i];
  }


  return true;
}



bool NPME_Kcycle_2_PermuteArray (const size_t N, 
  const NPME_Kcycle& Kcycle, size_t *P)
{
  if (N != Kcycle.Get_N())
  {
    std::cout << "Error in NPME_Kcycle_2_PermuteArray.\n";
    std::cout << "N != Kcycle.Get_N()\n";
    return false;
  }

  const size_t nKcycle      = Kcycle.GetNumKcycle();
  const size_t *KcycleSize  = Kcycle.GetKcycleSize();
  const size_t *KcycleStart = Kcycle.GetKcycleStart();
  const size_t *K           = Kcycle.GetK();


  for (size_t i = 0; i < nKcycle; i++)
  {
    size_t start = KcycleStart[i];
    for (size_t j = 0; j < KcycleSize[i]-1; j++)
    {
      const size_t a = K[start+j];
      P[a] = K[start+j+1];
    }
    {
      const size_t a = K[start+KcycleSize[i]-1];
      P[a] = K[start];
    }
  }

  return true;
}

bool NPME_Kcycle_PermuteArray (const size_t N, 
  const NPME_Library::NPME_Kcycle& Kcycle, double *Y, int nProc)
{
  if (N != Kcycle.Get_N())
  {
    std::cout << "Error in NPME_Kcycle_2_PermuteArray.\n";
    std::cout << "N != Kcycle.Get_N()\n";
    return false;
  }

  size_t i;
  #pragma omp parallel shared(Kcycle, Y) private(i) num_threads(nProc)
  {
    const size_t nKcycle      = Kcycle.GetNumKcycle();
    const size_t *KcycleSize  = Kcycle.GetKcycleSize();
    const size_t *KcycleStart = Kcycle.GetKcycleStart();
    const size_t *K           = Kcycle.GetK();

    #pragma omp for schedule(dynamic)
    for (i = 0; i < nKcycle; i++)
    {
      size_t start = KcycleStart[i];
      double Y0    = Y[ K[start] ];

      for (size_t j = 0; j < KcycleSize[i]-1; j++)
      {
        const size_t a0 = K[start+j];
        const size_t a1 = K[start+j+1];
        Y[a0] = Y[a1];
      }
      {
        const size_t a0 = K[start+KcycleSize[i]-1];
        Y[a0] = Y0;
      }
    }
  }

  return true;
}

bool NPME_Kcycle_PermuteArray_2N (const size_t N, 
  const NPME_Library::NPME_Kcycle& Kcycle, double *Y, int nProc)
{
  if (N != Kcycle.Get_N())
  {
    std::cout << "Error in NPME_Kcycle_2_PermuteArray.\n";
    std::cout << "N != Kcycle.Get_N()\n";
    return false;
  }

  size_t i;
  #pragma omp parallel shared(Kcycle, Y) private(i) num_threads(nProc)
  {
    const size_t nKcycle      = Kcycle.GetNumKcycle();
    const size_t *KcycleSize  = Kcycle.GetKcycleSize();
    const size_t *KcycleStart = Kcycle.GetKcycleStart();
    const size_t *K           = Kcycle.GetK();

    #pragma omp for schedule(dynamic)
    for (size_t i = 0; i < nKcycle; i++)
    {
      size_t start    = KcycleStart[i];
      double *Yloc    = &Y[ 2*K[start] ];
      double Ysave[2] = {Yloc[0], Yloc[1]};

      for (size_t j = 0; j < KcycleSize[i]-1; j++)
      {
        double *Y0      = &Y[2*K[start+j]   ];
        double *Y1      = &Y[2*K[start+j+1] ];

        Y0[0] = Y1[0];
        Y0[1] = Y1[1];
      }
      {
        const size_t a0 = K[start+KcycleSize[i]-1];

        double *Y0 = &Y[2*a0];
        Y0[0] = Ysave[0];
        Y0[1] = Ysave[1];
      }
    }
  }

  return true;
}
bool NPME_Kcycle_PermuteArray_3N (const size_t N, 
  const NPME_Library::NPME_Kcycle& Kcycle, double *Y, int nProc)
{
  if (N != Kcycle.Get_N())
  {
    std::cout << "Error in NPME_Kcycle_2_PermuteArray.\n";
    std::cout << "N != Kcycle.Get_N()\n";
    return false;
  }

  size_t i;
  #pragma omp parallel shared(Kcycle, Y) private(i) num_threads(nProc)
  {
    const size_t nKcycle      = Kcycle.GetNumKcycle();
    const size_t *KcycleSize  = Kcycle.GetKcycleSize();
    const size_t *KcycleStart = Kcycle.GetKcycleStart();
    const size_t *K           = Kcycle.GetK();

    #pragma omp for schedule(dynamic)
    for (size_t i = 0; i < nKcycle; i++)
    {
      size_t start    = KcycleStart[i];
      double *Yloc    = &Y[ 3*K[start] ];
      double Ysave[3] = {Yloc[0], Yloc[1], Yloc[2]};

      for (size_t j = 0; j < KcycleSize[i]-1; j++)
      {
        double *Y0      = &Y[3*K[start+j]   ];
        double *Y1      = &Y[3*K[start+j+1] ];

        Y0[0] = Y1[0];
        Y0[1] = Y1[1];
        Y0[2] = Y1[2];
      }
      {
        const size_t a0 = K[start+KcycleSize[i]-1];

        double *Y0 = &Y[3*a0];
        Y0[0] = Ysave[0];
        Y0[1] = Ysave[1];
        Y0[2] = Ysave[2];
      }
    }
  }

  return true;
}

bool NPME_Kcycle_PermuteArray_MN (const size_t M, const size_t N, 
  const NPME_Library::NPME_Kcycle& Kcycle, double *Y, int nProc)
//input:  Y[N][M] and P[N]
//output: Y[N][M] 
//        Y[P(n)][] -> Y[n][]
{
  const size_t MaxTempArraySize = 1000;


  if      (M == 1)  return NPME_Kcycle_PermuteArray    (N, Kcycle, Y, nProc);
  else if (M == 2)  return NPME_Kcycle_PermuteArray_2N (N, Kcycle, Y, nProc);
  else if (M == 3)  return NPME_Kcycle_PermuteArray_3N (N, Kcycle, Y, nProc);
  else
  {
    size_t i;
    #pragma omp parallel shared(Kcycle, Y) private(i) num_threads(nProc)
    {
      const size_t nKcycle      = Kcycle.GetNumKcycle();
      const size_t *KcycleSize  = Kcycle.GetKcycleSize();
      const size_t *KcycleStart = Kcycle.GetKcycleStart();
      const size_t *K           = Kcycle.GetK();


      double *Ysave;
      double _Yconst[MaxTempArraySize];
      std::vector<double> _Yvec;

      if (M > MaxTempArraySize)
      {
        _Yvec.resize(M);
        Ysave = &_Yvec[0];
      }
      else
        Ysave = _Yconst;


      #pragma omp for schedule(dynamic)
      for (size_t i = 0; i < nKcycle; i++)
      {
        size_t start    = KcycleStart[i];
        double *Yloc    = &Y[ M*K[start] ];
        for (size_t m = 0; m < M; m++)
          Ysave[m] = Yloc[m];


        for (size_t j = 0; j < KcycleSize[i]-1; j++)
        {
          double *Y0      = &Y[M*K[start+j]   ];
          double *Y1      = &Y[M*K[start+j+1] ];

          for (size_t m = 0; m < M; m++)
            Y0[m] = Y1[m];
        }
        {
          const size_t a0 = K[start+KcycleSize[i]-1];

          double *Y0 = &Y[M*a0];
          for (size_t m = 0; m < M; m++)
            Y0[m] = Ysave[m];
        }
      }
    }
  }

  return true;
}




bool NPME_Kcycle_InversePermuteArray (const size_t N, 
  const NPME_Library::NPME_Kcycle& Kcycle, double *Y, int nProc)
{
  if (N != Kcycle.Get_N())
  {
    std::cout << "Error in NPME_Kcycle_InversePermuteArray.\n";
    std::cout << "N != Kcycle.Get_N()\n";
    return false;
  }

  size_t i;
  #pragma omp parallel shared(Kcycle, Y) private(i) num_threads(nProc)
  {
    const size_t nKcycle      = Kcycle.GetNumKcycle();
    const size_t *KcycleSize  = Kcycle.GetKcycleSize();
    const size_t *KcycleStart = Kcycle.GetKcycleStart();
    const size_t *K           = Kcycle.GetK();

    #pragma omp for schedule(dynamic)
    for (i = 0; i < nKcycle; i++)
    {
      const size_t start  = KcycleStart[i];
      const size_t Ncycle = KcycleSize[i];

      double Y1    = Y[ K[start+Ncycle-1] ];

      for (size_t j = 0; j < Ncycle-1; j++)
      {
        const size_t a0 = K[start+Ncycle-2-j];
        const size_t a1 = K[start+Ncycle-1-j];
        Y[a1] = Y[a0];
      }
      {
        const size_t a1 = K[start];
        Y[a1] = Y1;
      }
    }
  }

  return true;
}

bool NPME_Kcycle_InversePermuteArray_2N (const size_t N, 
  const NPME_Library::NPME_Kcycle& Kcycle, double *Y, int nProc)
{
  if (N != Kcycle.Get_N())
  {
    std::cout << "Error in NPME_Kcycle_InversePermuteArray_2N.\n";
    std::cout << "N != Kcycle.Get_N()\n";
    return false;
  }

  size_t i;
  #pragma omp parallel shared(Kcycle, Y) private(i) num_threads(nProc)
  {
    const size_t nKcycle      = Kcycle.GetNumKcycle();
    const size_t *KcycleSize  = Kcycle.GetKcycleSize();
    const size_t *KcycleStart = Kcycle.GetKcycleStart();
    const size_t *K           = Kcycle.GetK();

    #pragma omp for schedule(dynamic)
    for (i = 0; i < nKcycle; i++)
    {
      const size_t start  = KcycleStart[i];
      const size_t Ncycle = KcycleSize[i];

      double *Yloc    = &Y[ 2*K[start+Ncycle-1] ];
      double Ysave[2] = {Yloc[0], Yloc[1]};

      for (size_t j = 0; j < Ncycle-1; j++)
      {
        double *Y0      = &Y[2*K[start+Ncycle-2-j] ];
        double *Y1      = &Y[2*K[start+Ncycle-1-j] ];

        Y1[0] = Y0[0];
        Y1[1] = Y0[1];
      }
      {
        const size_t a1 = K[start];

        double *Y1 = &Y[2*a1   ];
        Y1[0] = Ysave[0];
        Y1[1] = Ysave[1];
      }
    }
  }

  return true;
}

bool NPME_Kcycle_InversePermuteArray_3N (const size_t N, 
  const NPME_Library::NPME_Kcycle& Kcycle, double *Y, int nProc)
{
  if (N != Kcycle.Get_N())
  {
    std::cout << "Error in NPME_Kcycle_InversePermuteArray_3N.\n";
    std::cout << "N != Kcycle.Get_N()\n";
    return false;
  }

  size_t i;
  #pragma omp parallel shared(Kcycle, Y) private(i) num_threads(nProc)
  {
    const size_t nKcycle      = Kcycle.GetNumKcycle();
    const size_t *KcycleSize  = Kcycle.GetKcycleSize();
    const size_t *KcycleStart = Kcycle.GetKcycleStart();
    const size_t *K           = Kcycle.GetK();

    #pragma omp for schedule(dynamic)
    for (i = 0; i < nKcycle; i++)
    {
      const size_t start  = KcycleStart[i];
      const size_t Ncycle = KcycleSize[i];

      double *Yloc    = &Y[ 3*K[start+Ncycle-1] ];
      double Ysave[3] = {Yloc[0], Yloc[1], Yloc[2]};

      for (size_t j = 0; j < Ncycle-1; j++)
      {
        double *Y0      = &Y[3*K[start+Ncycle-2-j] ];
        double *Y1      = &Y[3*K[start+Ncycle-1-j] ];

        Y1[0] = Y0[0];
        Y1[1] = Y0[1];
        Y1[2] = Y0[2];
      }
      {
        const size_t a1 = K[start];

        double *Y1 = &Y[3*a1   ];
        Y1[0] = Ysave[0];
        Y1[1] = Ysave[1];
        Y1[2] = Ysave[2];
      }
    }
  }

  return true;
}

bool NPME_Kcycle_InversePermuteArray_MN (const size_t M, const size_t N, 
  const NPME_Library::NPME_Kcycle& Kc, double *Y, int nProc)
//input:  Y[N][M] and P[N]
//output: Y[N][M] 
//        Y[n][] -> Y[P(n)][]
{
  if (N != Kc.Get_N())
  {
    std::cout << "Error in NPME_Kcycle_InversePermuteArray_MN.\n";
    std::cout << "N != Kc.Get_N()\n";
    return false;
  }

  const size_t MaxTempArraySize = 1000;

  if      (M == 1)  return NPME_Kcycle_InversePermuteArray    (N, Kc, Y, nProc);
  else if (M == 2)  return NPME_Kcycle_InversePermuteArray_2N (N, Kc, Y, nProc);
  else if (M == 3)  return NPME_Kcycle_InversePermuteArray_3N (N, Kc, Y, nProc);
  else
  {
    size_t i;
    #pragma omp parallel shared(Kc, Y) private(i) num_threads(nProc)
    {
      const size_t nKcycle      = Kc.GetNumKcycle();
      const size_t *KcycleSize  = Kc.GetKcycleSize();
      const size_t *KcycleStart = Kc.GetKcycleStart();
      const size_t *K           = Kc.GetK();

      #pragma omp for schedule(dynamic)
      for (i = 0; i < nKcycle; i++)
      {
        const size_t start  = KcycleStart[i];
        const size_t Ncycle = KcycleSize[i];

        double *Ysave;
        double _Yconst[MaxTempArraySize];
        std::vector<double> _Yvec;

        if (M > MaxTempArraySize)
        {
          _Yvec.resize(M);
          Ysave = &_Yvec[0];
        }
        else
          Ysave = _Yconst;

        double *Yloc    = &Y[ M*K[start+Ncycle-1] ];
        for (size_t m = 0; m < M; m++)
          Ysave[m] = Yloc[m];

        for (size_t j = 0; j < Ncycle-1; j++)
        {
          double *Y0      = &Y[M*K[start+Ncycle-2-j] ];
          double *Y1      = &Y[M*K[start+Ncycle-1-j] ];

          for (size_t m = 0; m < M; m++)
            Y1[m] = Y0[m];
        }
        {
          const size_t a1 = K[start];

          double *Y1 = &Y[M*a1   ];
          for (size_t m = 0; m < M; m++)
            Y1[m] = Ysave[m];
        }
      }
    }
  }

  return true;
}

//******************************************************************************
//******************************************************************************
//********************Permutations Using a 2nd Copy*****************************
//******************************************************************************
//******************************************************************************
void NPME_PermuteArray (const size_t N, const size_t *P, 
  double *Y, const double *X, int nProc)
{
  size_t i;
  #pragma omp parallel for schedule(static) shared(X, Y, P) private(i) num_threads(nProc)
  for (i = 0; i < N; i++)
    Y[i] = X[ P[i] ];
}
void NPME_PermuteArray_2N (const size_t N, const size_t *P, 
  double *Y, const double *X, int nProc)
{
  size_t i;
  #pragma omp parallel for schedule(static) shared(X, Y, P) private(i) num_threads(nProc)
  for (i = 0; i < N; i++)
  {
          double *Yloc = &Y[2*i];
    const double *Xloc = &X[2*P[i]];

    Yloc[0] = Xloc[0];
    Yloc[1] = Xloc[1];
  }
}
void NPME_PermuteArray_3N (const size_t N, const size_t *P, 
  double *Y, const double *X, int nProc)
{
  size_t i;
  #pragma omp parallel for schedule(static) shared(X, Y, P) private(i) num_threads(nProc)
  for (i = 0; i < N; i++)
  {
          double *Yloc = &Y[3*i];
    const double *Xloc = &X[3*P[i]];

    Yloc[0] = Xloc[0];
    Yloc[1] = Xloc[1];
    Yloc[2] = Xloc[2];
  }
}

void NPME_PermuteArray_MN (const size_t M, const size_t N, const size_t *P, 
  double *Y, const double *X, int nProc)
//input:  X[N][M] and P[N]
//output: Y[N][M] 
//        Y[n][] = X[P(n)][]
{
  if      (M == 1)  return NPME_PermuteArray    (N, P, Y, X, nProc);
  else if (M == 2)  return NPME_PermuteArray_2N (N, P, Y, X, nProc);
  else if (M == 3)  return NPME_PermuteArray_3N (N, P, Y, X, nProc);
  else
  {
    size_t i;
    #pragma omp parallel for schedule(static) shared(X, Y, P) private(i) num_threads(nProc)
    for (i = 0; i < N; i++)
    {
            double *Yloc = &Y[M*i];
      const double *Xloc = &X[M*P[i]];

      for (int p = 0; p < M; p++)
        Yloc[p] = Xloc[p];
    }
  }
}

void NPME_PermuteArrayInverse (const size_t N, const size_t *P, 
  double *Y, const double *X, int nProc)
{
  size_t i;
  #pragma omp parallel for schedule(static) shared(X, Y, P) private(i) num_threads(nProc)
  for (i = 0; i < N; i++)
    Y[ P[i] ] = X[ i ];
}
void NPME_PermuteArrayInverse_2N (const size_t N, const size_t *P, 
  double *Y, const double *X, int nProc)
{
  size_t i;
  #pragma omp parallel for schedule(static) shared(X, Y, P) private(i) num_threads(nProc)
  for (i = 0; i < N; i++)
  {
          double *Yloc = &Y[2*P[i]];
    const double *Xloc = &X[2*i];

    Yloc[0] = Xloc[0];
    Yloc[1] = Xloc[1];
  }
}
void NPME_PermuteArrayInverse_3N (const size_t N, const size_t *P, 
  double *Y, const double *X, int nProc)
{
  size_t i;
  #pragma omp parallel for schedule(static) shared(X, Y, P) private(i) num_threads(nProc)
  for (i = 0; i < N; i++)
  {
          double *Yloc = &Y[3*P[i]];
    const double *Xloc = &X[3*i];

    Yloc[0] = Xloc[0];
    Yloc[1] = Xloc[1];
    Yloc[2] = Xloc[2];
  }
}


void NPME_PermuteArrayInverse_MN (const size_t M, const size_t N, 
  const size_t *P, double *Y, const double *X, int nProc)
//input:  X[N][M] and P[N]
//output: Y[N][M] 
//        Y[P(n)][] = X[n][]
{
  if      (M == 1)  return NPME_PermuteArrayInverse    (N, P, Y, X, nProc);
  else if (M == 2)  return NPME_PermuteArrayInverse_2N (N, P, Y, X, nProc);
  else if (M == 3)  return NPME_PermuteArrayInverse_3N (N, P, Y, X, nProc);
  else
  {
    size_t i;
    #pragma omp parallel for schedule(static) shared(X, Y, P) private(i) num_threads(nProc)
    for (i = 0; i < N; i++)
    {
            double *Yloc = &Y[M*P[i]];
      const double *Xloc = &X[M*i];

      for (int p = 0; p < M; p++)
        Yloc[p] = Xloc[p];
    }
  }
}


//******************************************************************************
//******************************************************************************
//***************Random Permutation Operator used for Testing*******************
//******************************************************************************
//******************************************************************************
void NPME_GenerateRandomPermutationOperator (const size_t N, size_t *P)
//used for testing
{
  const size_t NumPijInterchange = N*1000;

  for (size_t i = 0; i < N; i++)
    P[i] = i;

  for (size_t n = 0; n < NumPijInterchange; n++)
  {
    const size_t r1     = (size_t) rand();
    const size_t r2     = (size_t) rand();
    const size_t index1 = r1%N;
    const size_t index2 = r2%N;

    const size_t iTemp  = P[index1];
    P[index1]           = P[index2];
    P[index2]           = iTemp;
  }
}

bool NPME_IsPermutationOperatorValid (const size_t *P, const size_t N)
//1) P[i] must occur once and only once
//2) 0 <= P[i] < N
{
  std::vector<size_t> count(N);
  memset(&count[0], 0, N*sizeof(size_t));

  for (size_t i = 0; i < N; i++)
  {
    if (P[i] >= N)
      return false;
    count[P[i]]++;
  }

  for (size_t i = 0; i < N; i++)
    if (count[i] != 1)
      return false;

  return true;
}

}//end namespace NPME_Library



