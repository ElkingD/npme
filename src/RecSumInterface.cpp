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
#include "RecSumInterface.h"

#include "RecSumSupportFunctions.h"
#include "RecSumQ.h"
#include "RecSumV.h"
#include "RecSumGrid.h"

#include "SupportFunctions.h"
#include "ExtLibrary.h"
#include "Bspline.h"

#include "PotentialGenFunc.h"


namespace NPME_Library
{
NPME_RecSumInterface::NPME_RecSumInterface ()
{
  _isSetUp        = 0;
  _nNonZeroBlock  = 0;
  _funcReal       = NULL;
  _funcComplex    = NULL;
}


//copy constructor, assignment operator, and destructor
NPME_RecSumInterface::NPME_RecSumInterface
                               (const NPME_RecSumInterface& rhs)
{
  if (rhs._isSetUp)
  {
    _isSetUp                      = rhs._isSetUp;
    _useRealFunc                  = rhs._useRealFunc;
    _funcReal                     = rhs._funcReal;
    _funcComplex                  = rhs._funcComplex;

    _nCharge                      = rhs._nCharge;

    _nProc                        = rhs._nProc;
    _BnOrder                      = rhs._BnOrder;

    _a1                           = rhs._a1;
    _a2                           = rhs._a2;
    _a3                           = rhs._a3;
    _del1                         = rhs._del1;
    _del2                         = rhs._del2;
    _del3                         = rhs._del3;

    _N1                           = rhs._N1;
    _N2                           = rhs._N2;
    _N3                           = rhs._N3;
    _n1                           = rhs._n1;
    _n2                           = rhs._n2;
    _n3                           = rhs._n3;

    _L1                           = rhs._L1;
    _L2                           = rhs._L2;
    _L3                           = rhs._L3;
    _X                            = rhs._X;
    _Y                            = rhs._Y;
    _Z                            = rhs._Z;
    _R[0]                         = rhs._R[0];
    _R[1]                         = rhs._R[1];
    _R[2]                         = rhs._R[2];

    _nNonZeroBlock                = rhs._nNonZeroBlock;
    _NonZeroBlock2BlockIndex_XYZ  = rhs._NonZeroBlock2BlockIndex_XYZ;
    _nNonZeroBlock2Charge         = rhs._nNonZeroBlock2Charge;
    _NonZeroBlock2Charge1D        = rhs._NonZeroBlock2Charge1D;

    _lamda1_2                     = rhs._lamda1_2;
    _lamda2_2                     = rhs._lamda2_2;
    _lamda3_2                     = rhs._lamda3_2;
    _C1                           = rhs._C1;
    _C2                           = rhs._C2;
    _C3                           = rhs._C3;

    _useSphereSymmCompactF        = rhs._useSphereSymmCompactF;
    _F                            = rhs._F;
    _Xtmp                         = rhs._Xtmp;

    //copy _NonZeroBlock2Charge[_nNonZeroBlock][] as a 2D array of pointers to
    //     _NonZeroBlock2Charge1D

    if ( (_nNonZeroBlock != rhs._NonZeroBlock2Charge.size()) ||
         (_nNonZeroBlock != _nNonZeroBlock2Charge.size()) ||
         (_nNonZeroBlock != _NonZeroBlock2BlockIndex_XYZ.size()/3))
    {
      std::cout << "Unexpected error while trying to copy ";
      std::cout << "NPME_RecSumInterface\n";
      exit(0);
    }
    _NonZeroBlock2Charge.resize(_nNonZeroBlock);

    size_t count = 0;
    for (size_t i = 0; i < _nNonZeroBlock; i++)
    {
      _NonZeroBlock2Charge[i] = &_NonZeroBlock2Charge1D[count];
      count += _nNonZeroBlock2Charge[i];
    }
  }
}


NPME_RecSumInterface& NPME_RecSumInterface::operator= (const NPME_RecSumInterface& rhs)
{

  if ( (this != &rhs) && (rhs._isSetUp))
  {
    _isSetUp                      = rhs._isSetUp;
    _useRealFunc                  = rhs._useRealFunc;
    _funcReal                     = rhs._funcReal;
    _funcComplex                  = rhs._funcComplex;

    _nCharge                      = rhs._nCharge;

    _nProc                        = rhs._nProc;
    _BnOrder                      = rhs._BnOrder;

    _a1                           = rhs._a1;
    _a2                           = rhs._a2;
    _a3                           = rhs._a3;
    _del1                         = rhs._del1;
    _del2                         = rhs._del2;
    _del3                         = rhs._del3;

    _N1                           = rhs._N1;
    _N2                           = rhs._N2;
    _N3                           = rhs._N3;
    _n1                           = rhs._n1;
    _n2                           = rhs._n2;
    _n3                           = rhs._n3;

    _L1                           = rhs._L1;
    _L2                           = rhs._L2;
    _L3                           = rhs._L3;
    _X                            = rhs._X;
    _Y                            = rhs._Y;
    _Z                            = rhs._Z;
    _R[0]                         = rhs._R[0];
    _R[1]                         = rhs._R[1];
    _R[2]                         = rhs._R[2];

    _nNonZeroBlock                = rhs._nNonZeroBlock;
    _NonZeroBlock2BlockIndex_XYZ  = rhs._NonZeroBlock2BlockIndex_XYZ;
    _nNonZeroBlock2Charge         = rhs._nNonZeroBlock2Charge;
    _NonZeroBlock2Charge1D        = rhs._NonZeroBlock2Charge1D;

    _lamda1_2                     = rhs._lamda1_2;
    _lamda2_2                     = rhs._lamda2_2;
    _lamda3_2                     = rhs._lamda3_2;
    _C1                           = rhs._C1;
    _C2                           = rhs._C2;
    _C3                           = rhs._C3;

    _useSphereSymmCompactF        = rhs._useSphereSymmCompactF;
    _F                            = rhs._F;
    _Xtmp                         = rhs._Xtmp;

    //copy _NonZeroBlock2Charge[_nNonZeroBlock][] as a 2D array of pointers to
    //     _NonZeroBlock2Charge1D

    if ( (_nNonZeroBlock != rhs._NonZeroBlock2Charge.size()) ||
         (_nNonZeroBlock != _nNonZeroBlock2Charge.size()) ||
         (_nNonZeroBlock != _NonZeroBlock2BlockIndex_XYZ.size()/3))
    {
      std::cout << "Unexpected error while trying to copy ";
      std::cout << "NPME_RecSumInterface\n";
      exit(0);
    }
    _NonZeroBlock2Charge.resize(_nNonZeroBlock);

    size_t count = 0;
    for (size_t i = 0; i < _nNonZeroBlock; i++)
    {
      _NonZeroBlock2Charge[i] = &_NonZeroBlock2Charge1D[count];
      count += _nNonZeroBlock2Charge[i];
    }
  }

  return *this;
}



bool NPME_RecSumInterface::SetUp (const size_t nCharge, 
    const double *coord, const long int BnOrder, bool useSphereSymmCompactF, 
    const long int N1, const long int N2, const long int N3,
    const long int n1, const long int n2, const long int n3,
    const double del1, const double del2, const double del3,
    const double a1,   const double a2,   const double a3,
    NPME_KfuncReal *funcLR, int nProc, 
    bool PRINT, std::ostream& os)
{
  _useRealFunc = 1;
  _funcReal    = funcLR;

  return SetUp (nCharge, coord, BnOrder, useSphereSymmCompactF, 
                N1,   N2,   N3,
                n1,   n2,   n3,
                del1, del2, del3,
                a1,   a2,   a3, 
                nProc, PRINT, os);
}

bool NPME_RecSumInterface::SetUp (const size_t nCharge, 
    const double *coord, const long int BnOrder, bool useSphereSymmCompactF, 
    const long int N1, const long int N2, const long int N3,
    const long int n1, const long int n2, const long int n3,
    const double del1, const double del2, const double del3,
    const double a1,   const double a2,   const double a3,
    NPME_KfuncComplex *funcLR, int nProc, 
    bool PRINT, std::ostream& os)
{
  _useRealFunc  = 0;
  _funcComplex  = funcLR;

  return SetUp (nCharge, coord, BnOrder, useSphereSymmCompactF, 
                N1,   N2,   N3,
                n1,   n2,   n3,
                del1, del2, del3,
                a1,   a2,   a3, 
                nProc, PRINT, os);
}

bool NPME_RecSumInterface::AllocateFFTmem (bool useSphereSymmCompactF, 
    const long int N1, const long int N2, const long int N3,
    const long int n1, const double maxFFTmemGB, bool PRINT, std::ostream& os)
{
  double time0, time;

  _N1                     = N1;
  _N2                     = N2;
  _N3                     = N3;
  _n1                     = n1;
  _useSphereSymmCompactF  = useSphereSymmCompactF;

  {
    const double memNeed = NPME_CalcFFTmemGB (_N1, _N2, _N3, _n1, 
                            useSphereSymmCompactF);
    if (memNeed > maxFFTmemGB)
    {
      std::cout << "Error in NPME_RecSumInterface::AllocateFFTmem\n";
      char str[500];
      sprintf(str, "FFTmemGB (needed) = %f GB > %f GB = maxFFTmemGB\n",
        memNeed, maxFFTmemGB);
      std::cout << str;
      return false;
    }
  }



  if (PRINT)
  {
    os << "\n\nNPME_RecSumInterface::AllocateFFTmem\n";
    os.flush();
  }

  time0 = NPME_GetTime ();
  //1) allocate FFT of smooth kernel evaluated on grid
  if (_useSphereSymmCompactF)
  {
    const size_t M1     = (size_t)  (_N1/2);
    const size_t M2     = (size_t)  (_N2/2);
    const size_t M3     = (size_t)  (_N3/2);
    const size_t sizeFc = (M1+1)*(M2+1)*(M3+1);
    if (_F.size() < sizeFc)
    {
      _F.resize(sizeFc);
      memset(&_F[0], 0, sizeFc*sizeof(_Complex double));
    }
  }
  else
  {
    const size_t sizeFf = (size_t) (_N1*_N2*_N3);
    if (_F.size() < sizeFf)
    {
      _F.resize(sizeFf);
      memset(&_F[0], 0, sizeFf*sizeof(_Complex double));
    }
  }

  //2) Xtmp array for storing/processing FFTs
  {
    long int size_Q     = _N1*_N2*_N3/8;
    long int size_Qtmp  = _n1*_N2*_N3/4;
    long int size_QXT   = 3*size_Q + size_Qtmp;

    size_t size_Xtmp = size_QXT;
    if (size_Xtmp < _F.size())
      size_Xtmp = _F.size();

    if (_Xtmp.size() < size_Xtmp)
    {
      _Xtmp.resize(size_Xtmp);
      memset(&_Xtmp[0], 0, size_Xtmp*sizeof(_Complex double));
    }
  }
  time = NPME_GetTime () - time0;

  if (PRINT)
  {
    char str[2000];
    sprintf(str, "  time           =  %8.4f (FFT allocation time)\n", time);
    os << str;
    os.flush();
  }

  return true;
}


bool NPME_RecSumInterface::SetUp (const size_t nCharge, 
    const double *coord, const long int BnOrder, bool useSphereSymmCompactF, 
    const long int N1, const long int N2, const long int N3,
    const long int n1, const long int n2, const long int n3,
    const double del1, const double del2, const double del3,
    const double a1,   const double a2,   const double a3,
    int nProc, bool PRINT, std::ostream& os)
{
  bool PRINT_ALL = 0;
  double time0, time;

  _nCharge                = nCharge;
  _BnOrder                = BnOrder;
  _nProc                  = nProc;

  _N1                     = N1;
  _N2                     = N2;
  _N3                     = N3;
  _n1                     = n1;
  _n2                     = n2;
  _n3                     = n3;
  _del1                   = del1;
  _del2                   = del2;
  _del3                   = del3;
  _a1                     = a1;
  _a2                     = a2;
  _a3                     = a3;
  _useSphereSymmCompactF  = useSphereSymmCompactF;

  if (!NPME_CheckFFTParm (_BnOrder, _N1, _N2, _N3, _n1, _n2, _n3))
  {
    std::cout << "Error in NPME_RecSumInterface::SetUp.\n";
    std::cout << "  NPME_CheckFFTParm failed\n";
    return false;
  }

  //Box dimensions and center
  {
    double X0, Y0, Z0;
    double R0[3];
    NPME_CalcBoxDimensionCenter (nCharge, coord, X0, Y0, Z0, R0);

    NPME_RecSumInterface_GetPMECorrBox (
       _X,    _Y,     _Z,  _R,
        X0,    Y0,     Z0,  R0,
      _N1,    _N2,    _N3, 
      _del1,  _del2,  _del3, _BnOrder);



    //FFT volume dimensions
    _L1 = 2*(_X + _del1);
    _L2 = 2*(_Y + _del2);
    _L3 = 2*(_Z + _del3);

    if (PRINT)
    {
      char str[2000];
      os << "\n\nNPME_RecSumInterface::SetUp\n";
      sprintf(str, "  X0 Y0 Z0       =  %8.4f %8.4f %8.4f\n", X0,  Y0,  Z0);
      os << str;
      sprintf(str, "  R0             =  %8.4f %8.4f %8.4f\n", R0[0], R0[1], R0[2]);
      os << str;

      Print (os);
    }
  }

  time0 = NPME_GetTime ();
  //1) allocate FFT of smooth kernel evaluated on grid
  if (_useSphereSymmCompactF)
  {
    const size_t M1     = (size_t)  (_N1/2);
    const size_t M2     = (size_t)  (_N2/2);
    const size_t M3     = (size_t)  (_N3/2);
    const size_t sizeFc = (M1+1)*(M2+1)*(M3+1);
    if (_F.size() < sizeFc)
    {
      _F.resize(sizeFc);
      memset(&_F[0], 0, sizeFc*sizeof(_Complex double));
    }
  }
  else
  {
    const size_t sizeFf = (size_t) (_N1*_N2*_N3);
    if (_F.size() < sizeFf)
    {
      _F.resize(sizeFf);
      memset(&_F[0], 0, sizeFf*sizeof(_Complex double));
    }
  }

  //2) Xtmp array for storing/processing FFTs
  {
    long int size_Q     = _N1*_N2*_N3/8;
    long int size_Qtmp  = _n1*_N2*_N3/4;
    long int size_QXT   = 3*size_Q + size_Qtmp;

    size_t size_Xtmp = size_QXT;
    if (size_Xtmp < _F.size())
      size_Xtmp = _F.size();

    if (_Xtmp.size() < size_Xtmp)
    {
      _Xtmp.resize(size_Xtmp);
      memset(&_Xtmp[0], 0, size_Xtmp*sizeof(_Complex double));
    }
  }
  time = NPME_GetTime () - time0;

  if (PRINT)
  {
    char str[2000];
    sprintf(str, "  time           =  %8.4f (FFT allocation time)\n", time);
    os << str;
    os.flush();
  }


  if (_useSphereSymmCompactF)
  {
    if (_useRealFunc)
    {
      if (!NPME_RecSumGrid_CompactFourier_Func (&_F[0],
        _N1, _a1, _del1,
        _N2, _a2, _del2,
        _N3, _a3, _del3,
        _X,  _Y,  _Z,
        *_funcReal, _nProc, &_Xtmp[0], PRINT_ALL, os))
      {
        std::cout << "Error in NPME_RecSumInterface::SetUp\n";
        std::cout << "  NPME_RecSumGrid_CompactFourier_Func failed\n";
        return false;
      }
    }
    else
    {
      if (!NPME_RecSumGrid_CompactFourier_Func (&_F[0],
        _N1, _a1, _del1,
        _N2, _a2, _del2,
        _N3, _a3, _del3,
        _X,  _Y,  _Z,
        *_funcComplex, _nProc, &_Xtmp[0], PRINT_ALL, os))
      {
        std::cout << "Error in NPME_RecSumInterface::SetUp\n";
        std::cout << "  NPME_RecSumGrid_CompactFourier_Func failed\n";
        return false;
      }
    }
  }
  else
  {
    if (_useRealFunc)
    {
      if (!NPME_RecSumGrid_Full_Func (&_F[0],
        _N1, _a1, _del1,
        _N2, _a2, _del2,
        _N3, _a3, _del3,
        _X,  _Y,  _Z,
        *_funcReal, _nProc, PRINT_ALL, os))
      {
        std::cout << "Error in NPME_RecSumInterface::SetUp\n";
        std::cout << "  NPME_RecSumGrid_Full_Func failed\n";
        return false;
      }
    }
    else
    {
      if (!NPME_RecSumGrid_Full_Func (&_F[0],
        _N1, _a1, _del1,
        _N2, _a2, _del2,
        _N3, _a3, _del3,
        _X,  _Y,  _Z,
        *_funcComplex, _nProc, PRINT_ALL, os))
      {
        std::cout << "Error in NPME_RecSumInterface::SetUp\n";
        std::cout << "  NPME_RecSumGrid_Full_Func failed\n";
        return false;
      }
    }

    mkl_set_num_threads (_nProc);
    NPME_3D_FFT_NoNorm (&_F[0], _N1, _N2, _N3);
    mkl_set_num_threads (1);
  }


  NPME_RecSumQ_CreateBlock (_nNonZeroBlock, _NonZeroBlock2BlockIndex_XYZ, 
    _nNonZeroBlock2Charge, _NonZeroBlock2Charge, _NonZeroBlock2Charge1D, 
    _nCharge, coord, _BnOrder,
    _N1, _N2, _N3, 
    _L1, _L2, _L3, 
    _n1, _n2, _n3, 
    _R, PRINT_ALL, os);

  std::vector<_Complex double> lamda1(_N1);
  std::vector<_Complex double> lamda2(_N2);
  std::vector<_Complex double> lamda3(_N3);
  _lamda1_2.resize(_N1);
  _lamda2_2.resize(_N2);
  _lamda3_2.resize(_N3);
  NPME_RSTP_CalcLamda (_N1, _BnOrder, &lamda1[0], &_lamda1_2[0]);
  NPME_RSTP_CalcLamda (_N2, _BnOrder, &lamda2[0], &_lamda2_2[0]);
  NPME_RSTP_CalcLamda (_N3, _BnOrder, &lamda3[0], &_lamda3_2[0]);

  _C1.resize(_N1);
  _C2.resize(_N2);
  _C3.resize(_N3);
  NPME_RecSumQ_Calc_C1 (&_C1[0], _N1);
  NPME_RecSumQ_Calc_C1 (&_C2[0], _N2);
  NPME_RecSumQ_Calc_C1 (&_C3[0], _N3);

  _isSetUp = 1;

  return true;
}



void NPME_RecSumInterface::Print (std::ostream& os) const
{
  char str[2000];
  os << "\n\nNPME_RecSumInterface::Print\n";
  sprintf(str, "  del            =  %8.4f %8.4f %8.4f\n", _del1, _del2, _del3);
  os << str;
  sprintf(str, "  a              =  %8.4f %8.4f %8.4f\n", _a1, _a2, _a3);
  os << str;
  os << "\n";
  sprintf(str, "  X  Y  Z        =  %8.4f %8.4f %8.4f\n", _X,  _Y,  _Z);
  os << str;
  sprintf(str, "  L1 L2 L3       =  %8.4f %8.4f %8.4f\n", _L1, _L2, _L3);
  os << str;
  sprintf(str, "  N1 N2 N3       =  %4ld %4ld %4ld\n", _N1, _N2, _N3);
  os << str;
  sprintf(str, "  n1 n2 n3       =  %4ld %4ld %4ld\n", _n1, _n2, _n3);
  os << str;
  sprintf(str, "  R              =  %8.4f %8.4f %8.4f\n", _R[0], _R[1], _R[2]);
  os << str;

  os.flush();
}



bool NPME_RecSumInterface::CalcV1 (double *V1, 
  const size_t nCharge, const double *charge, const double *coord,
  bool zeroVarray, bool PRINT, std::ostream& os)
//input:  charge[nCharge]
//output: V1[nCharge][4]
{
  double time0, time;
  char str[2000];

  if (_useRealFunc == 0)
  {
    std::cout << "Error in NPME_RecSumInterface::CalcV1.\n";
    std::cout << "  input real charges, but useRealFunc = 0\n";
    return false;
  }

  if (nCharge != _nCharge)
  {
    std::cout << "Error in NPME_RecSumInterface::CalcV1.\n";
    sprintf(str, "  nCharge = %lu != %lu\n", nCharge, _nCharge);
    std::cout << str;
    return false;
  }



  long int size_Q     = _N1*_N2*_N3/8;
  long int size_Qtmp  = _n1*_N2*_N3/4;
  long int size_QXT   = 3*size_Q + size_Qtmp;

  if (PRINT)
  {
    sprintf(str, "\n\nNPME_RecSumInterface::CalcV1\n");
    os << str;
    os.flush();
  }


  time0 = NPME_GetTime ();
  if (_Xtmp.size() < size_QXT)
    _Xtmp.resize(size_QXT);
  time = NPME_GetTime () - time0;

  if (PRINT)
  {
    sprintf(str, " X allocation time                    = %6.2le\n", time);
    os << str;
    os.flush();
  }

  _Complex double *Qtmp = &_Xtmp[0];
  _Complex double *Qc   = &_Xtmp[size_Qtmp];
  _Complex double *Xc   = &_Xtmp[size_Qtmp +   size_Q];
  _Complex double *Tc   = &_Xtmp[size_Qtmp + 2*size_Q];


  time0 = NPME_GetTime ();
  NPME_RecSumQ_CalcQ1 (&Qc[0], _nCharge, charge, coord, 
    _N1, _N2, _N3,
    _L1, _L2, _L3,
    _n1, _n2, _n3,
    _BnOrder, _nNonZeroBlock, &_NonZeroBlock2BlockIndex_XYZ[0],  
    &_nNonZeroBlock2Charge[0], (const size_t **) &(_NonZeroBlock2Charge[0]), 
    _R, _nProc, 0, os);
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, " Q1 calc time                         = %6.2le\n", time);
    os << str;
    os.flush();
  }

  time0 = NPME_GetTime ();
  NPME_TransformBlock_Block_2_Q (&Qc[0], &Qtmp[0], 
    _N1/2, _N2/2, _N3/2, _n1, _n2, _n3, _nProc);
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, " Transform Block Q to Q time          = %6.2le\n", time);
    os << str;
    os.flush();
  }

  //2) calculate compact Tc = FFT of theta
  time0 = NPME_GetTime ();
  NPME_RSTP_CalcTheta (Tc, Xc, &_F[0], Qc,
    _N1, _N2, _N3, &_C1[0], &_C2[0], &_C3[0], 
    &_lamda1_2[0], &_lamda2_2[0], &_lamda3_2[0], _useSphereSymmCompactF,
    _nProc, 0, os);
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, " Calc Theta time                      = %6.2le\n", time);
    os << str;
    os.flush();
  }

  //convert theta to block form
  time0 = NPME_GetTime ();
  NPME_TransformBlock_Q_2_Block (Tc, &Qtmp[0], _N1/2, _N2/2, _N3/2, 
    _n1, _n2, _n3, _nProc);
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, " Transform Theta to Theta Block Time  = %6.2le\n", time);
    os << str;
    os.flush();
  }


  //initialized V1 to zero
  if (zeroVarray)
    NPME_ZeroArray (_nCharge*4, _nProc, V1, 1000);

  //3) NPME_RecSumV_V1 converts theta to block form
  const int outputFormat = 1;
  time0 = NPME_GetTime ();
  NPME_RecSumV_CalcV1 (V1, Tc, 
    _nCharge, coord, 
    _N1, _N2, _N3, 
    _L1, _L2, _L3,
    _n1, _n2, _n3,
    _BnOrder, _nProc, 
    _nNonZeroBlock, &_NonZeroBlock2BlockIndex_XYZ[0],  
    &_nNonZeroBlock2Charge[0], (const size_t **) &(_NonZeroBlock2Charge[0]), 
    _R, outputFormat, 0, os);


  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, " Calc V time                          = %6.2le\n", time);
    os << str;
    os.flush();
  }


  return true;
}


bool NPME_RecSumInterface::CalcV1 (_Complex double *V1, 
  const size_t nCharge, const _Complex double *charge, const double *coord,
  bool zeroVarray, bool PRINT, std::ostream& os)
//input:  charge[nCharge]
//output: V1[nCharge][4]
{
  double time0, time;
  char str[2000];

  if (_useRealFunc == 1)
  {
    std::cout << "Error in NPME_RecSumInterface::CalcV1.\n";
    std::cout << "  input complex charges, but useRealFunc = 1\n";
    return false;
  }

  if (nCharge != _nCharge)
  {
    std::cout << "Error in NPME_RecSumInterface::CalcV1.\n";
    sprintf(str, "  nCharge = %lu != %lu\n", nCharge, _nCharge);
    std::cout << str;
    return false;
  }



  long int size_Q     = _N1*_N2*_N3/8;
  long int size_Qtmp  = _n1*_N2*_N3/4;
  long int size_QXT   = 3*size_Q + size_Qtmp;

  if (PRINT)
  {
    sprintf(str, "\n\nNPME_RecSumInterface::CalcV1\n");
    os << str;
    os.flush();
  }


  time0 = NPME_GetTime ();
  if (_Xtmp.size() < size_QXT)
    _Xtmp.resize(size_QXT);
  time = NPME_GetTime () - time0;

  if (PRINT)
  {
    sprintf(str, " X allocation time                    = %6.2le\n", time);
    os << str;
    os.flush();
  }

  _Complex double *Qtmp = &_Xtmp[0];
  _Complex double *Qc   = &_Xtmp[size_Qtmp];
  _Complex double *Xc   = &_Xtmp[size_Qtmp +   size_Q];
  _Complex double *Tc   = &_Xtmp[size_Qtmp + 2*size_Q];


  time0 = NPME_GetTime ();
  NPME_RecSumQ_CalcQ1 (&Qc[0], _nCharge, charge, coord, 
    _N1, _N2, _N3,
    _L1, _L2, _L3,
    _n1, _n2, _n3,
    _BnOrder, _nNonZeroBlock, &_NonZeroBlock2BlockIndex_XYZ[0],  
    &_nNonZeroBlock2Charge[0], (const size_t **) &(_NonZeroBlock2Charge[0]), 
    _R, _nProc, 0, os);
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, " Q1 calc time                         = %6.2le\n", time);
    os << str;
    os.flush();
  }

  time0 = NPME_GetTime ();
  NPME_TransformBlock_Block_2_Q (&Qc[0], &Qtmp[0], 
    _N1/2, _N2/2, _N3/2, _n1, _n2, _n3, _nProc);
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, " Transform Block Q to Q time          = %6.2le\n", time);
    os << str;
    os.flush();
  }

  //2) calculate compact Tc = FFT of theta
  time0 = NPME_GetTime ();
  NPME_RSTP_CalcTheta (Tc, Xc, &_F[0], Qc,
    _N1, _N2, _N3, &_C1[0], &_C2[0], &_C3[0], 
    &_lamda1_2[0], &_lamda2_2[0], &_lamda3_2[0], _useSphereSymmCompactF,
    _nProc, 0, os);
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, " Calc Theta time                      = %6.2le\n", time);
    os << str;
    os.flush();
  }

  //convert theta to block form
  time0 = NPME_GetTime ();
  NPME_TransformBlock_Q_2_Block (Tc, &Qtmp[0], _N1/2, _N2/2, _N3/2, 
    _n1, _n2, _n3, _nProc);
  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, " Transform Theta to Theta Block Time  = %6.2le\n", time);
    os << str;
    os.flush();
  }


  //initialized V1 to zero
  if (zeroVarray)
    NPME_ZeroArray (_nCharge*4, _nProc, V1, 1000);

  //3) NPME_RecSumV_V1 converts theta to block form
  const int outputFormat = 1;
  time0 = NPME_GetTime ();
  NPME_RecSumV_CalcV1 (V1, Tc, 
    _nCharge, coord, 
    _N1, _N2, _N3, 
    _L1, _L2, _L3,
    _n1, _n2, _n3,
    _BnOrder, _nProc, 
    _nNonZeroBlock, &_NonZeroBlock2BlockIndex_XYZ[0],  
    &_nNonZeroBlock2Charge[0], (const size_t **) &(_NonZeroBlock2Charge[0]), 
    _R, outputFormat, 0, os);


  time = NPME_GetTime () - time0;
  if (PRINT)
  {
    sprintf(str, " Calc V time                          = %6.2le\n", time);
    os << str;
    os.flush();
  }


  return true;
}












bool NPME_RecSumInterface::CalcV1_exact (double *V1, 
        const size_t nCharge, const double *charge, 
        const double *coord, int vecOption)
{
  char str[2000];

  if (_useRealFunc == 0)
  {
    std::cout << "Error in NPME_RecSumInterface::CalcV1_exact.\n";
    std::cout << "  input real charges, but useRealFunc = 0\n";
    return false;
  }

  if (nCharge != _nCharge)
  {
    std::cout << "Error in NPME_RecSumInterface::CalcV1_exact.\n";
    sprintf(str, "  nCharge = %lu != %lu\n", nCharge, _nCharge);
    std::cout << str;
    return false;
  }

  NPME_PotGenFunc_MacroSelf_V1 (*_funcReal, _nCharge, coord, charge, 
              V1, _nProc, vecOption);

  NPME_PotGenFunc_AddSelfTerm_V (*_funcReal, _nCharge, charge, V1);

  return true;
}



bool NPME_RecSumInterface::CalcV1_exact (_Complex double *V1, 
        const size_t nCharge, const _Complex double *charge, 
        const double *coord, int vecOption)
{
  char str[2000];

  if (_useRealFunc == 1)
  {
    std::cout << "Error in NPME_RecSumInterface::CalcV1_exact.\n";
    std::cout << "  input complex charges, but useRealFunc = 1\n";
    return false;
  }

  if (nCharge != _nCharge)
  {
    std::cout << "Error in NPME_RecSumInterface::CalcV1_exact.\n";
    sprintf(str, "  nCharge = %lu != %lu\n", nCharge, _nCharge);
    std::cout << str;
    return false;
  }

  NPME_PotGenFunc_MacroSelf_V1 (*_funcComplex, _nCharge, coord, charge, 
              V1, _nProc, vecOption);

  NPME_PotGenFunc_AddSelfTerm_V (*_funcComplex, _nCharge, charge, V1);

  return true;
}





double NPME_CalcFFTmemGB (
  const long int N1, const long int N2, const long int N3,
  const long int n1, bool useSphereSymmCompactF)
//input:  N1, N2, N3            = FFT sizes
//        n1                    = FFT block size in x direction
//        useSphereSymmCompactF = 0, 1 (set to 1 for a single box radially 
//                                      symmetric kernel and 0 otherwise)
//output: FFT memory in GB
{
  long int size_Q     = (N1*N2*N3)/8;
  long int size_Qtmp  = (n1*N2*N3)/4;
  long int size_QXT   = 3*size_Q + size_Qtmp;
  size_t size_F       = (size_t) (N1*N2*N3);
  if (useSphereSymmCompactF)
  {
    long int M1 = N1/2 + 1;
    long int M2 = N2/2 + 1;
    long int M3 = N3/2 + 1;

    size_F = (size_t) (M1*M2*M3);
  }

  double FFT_memory = (size_QXT + size_F)*16.0/1.0E9;


  return FFT_memory;
}








}//end namespace NPME_Library



