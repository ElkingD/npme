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

#ifndef NPME_INTERFACE_H
#define NPME_INTERFACE_H



#include "NPME_Constant.h"
#include "NPME_ReadPrint.h"
#include "NPME_KernelFunction.h"
#include "NPME_KernelFunctionLaplace.h"
#include "NPME_KernelFunctionRalpha.h"
#include "NPME_KernelFunctionHelmholtz.h"

#include "NPME_PartitionBox.h"
#include "NPME_PartitionEmbeddedBox.h"
#include "NPME_PermuteArray.h"
#include "NPME_RecSumInterface.h"

#define NPME_INTERFACE_DEBUG        0

namespace NPME_Library
{
//NPME_AddMissingKeywords = add default values to keywords and apply RSEM
bool NPME_AddMissingKeywords (const char *keywordFile, 
  NPME_KeywordInput& keyword, 
  const size_t nCharge, const bool isChargeReal, 
  const std::vector<double>& coord, const std::vector<double>& chargeReal, 
  const std::vector<_Complex double>& chargeComplex, 
  bool printLog, std::ostream& ofs_log);
bool NPME_AddMissingKeywords (const char *keywordFile, 
  NPME_KeywordInput& keyword, const size_t nCharge, 
  const std::vector<double>& coord, const std::vector<double>& chargeReal, 
  bool printLog, std::ostream& ofs_log);
bool NPME_AddMissingKeywords (const char *keywordFile, 
  NPME_KeywordInput& keyword, const size_t nCharge, 
  const std::vector<double>& coord, 
  const std::vector<_Complex double>& chargeComplex, 
  bool printLog, std::ostream& ofs_log);



class NPME_InterfaceBase
//NPME_InterfaceBase is an interface class for single box npme calcs
//NPME_InterfaceBase contains
//  1) system data (nProc, vecOption, etc.  - set with SetUpBase(..))
//  2) nCharge, coord, coordPermute         - set with SetUpBase(..))
//  3) direct sum data (e.g. cell list      - set with SetUpBase(..))
//NPME_InterfaceBase does NOT contain
//  1) real or complex charge data          - stored externally to interfaces
//  2) real or complex kernel functions     - stored externally to interfaces
//  3) recSum data (requires a smooth kernel function to be set)
{
public:
  NPME_InterfaceBase ()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceBase empty constructor\n";
    #endif

    _isBaseSet = 0;
  }

  //copy constructor, assignment operator, and destructor
  NPME_InterfaceBase (const NPME_Library::NPME_InterfaceBase& rhs);
  NPME_InterfaceBase& operator= (const NPME_Library::NPME_InterfaceBase& rhs);
  virtual ~NPME_InterfaceBase ()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  virtual ~NPME_InterfaceBase\n";
    #endif
  }




  int       GetNumProc      ()  const { return _nProc;                  }
  double    GetFFTmemGB     ()  const { return _FFTmemGB;               }
  int       GetVecOption    ()  const { return _vecOption;              }

  size_t    GetNumCharge    ()  const { return _nCharge;                }
  size_t    GetNumNeigh     ()  const { return _nNeigh;                 }
  double    GetRdir         ()  const { return _Rdir;                   }


  //rec sum parameters
  long int  Get_BnOrder     ()  const { return _BnOrder;                }

  long int  Get_N1          ()  const { return _N1;                     }
  long int  Get_N2          ()  const { return _N2;                     }
  long int  Get_N3          ()  const { return _N3;                     }

  long int  Get_n1          ()  const { return _n1;                     }
  long int  Get_n2          ()  const { return _n2;                     }
  long int  Get_n3          ()  const { return _n3;                     }

  double    Get_a1          ()  const { return _a1;                     }
  double    Get_a2          ()  const { return _a2;                     }
  double    Get_a3          ()  const { return _a3;                     }

  double    Get_del1        ()  const { return _del1;                   }
  double    Get_del2        ()  const { return _del2;                   }
  double    Get_del3        ()  const { return _del3;                   }



  bool SetUpBase (const NPME_Library::NPME_KeywordInput& keyword,
    const size_t nCharge, const double *coord, 
    std::vector<double>& coordPermute,
    bool printLog, std::ostream& ofs_log);

  void PrintBase (std::ostream& ofs_log) const;

protected:
  bool ResetCoordBase (
    const size_t nCharge, const double *coord, 
    std::vector<double>& coordPermute,
    bool printLog, std::ostream& ofs_log);

  bool      _isBaseSet;

  //system parameters
  int       _nProc;
  double    _FFTmemGB;
  int       _vecOption;

  //charge coords  
  size_t              _nCharge;

  //direct sum parameters
  std::vector<size_t> _P;             //P[nCharge] = permutation operator
  size_t              _nNeigh;
  double              _Rdir;          //direct sum cutoff



  size_t _nCellClust1D;
  size_t _nAvgChgPerCell,     _maxChgPerCell;
  size_t _nAvgChgPerCluster,  _maxChgPerCluster;
  size_t _nAvgCellPerCluster, _maxCellPerCluster;
  std::vector<NPME_Library::NPME_ClusterPair>       _cluster;

  NPME_Library::NPME_Kcycle _kcycle;



  //rec sum parameters
  bool      _useCompactF;           //use spherically symmetric F storage
  long int  _BnOrder;               //B spline order, must be even integer <= 40
  double    _a1,    _a2,    _a3;    //Fourier extension parameters
  double    _del1,  _del2,  _del3;  //Fourier extension parameters
  long int  _N1, _N2, _N3;          //FFT size
  long int  _n1, _n2, _n3;          //block size
};

class NPME_InterfaceReal : public NPME_InterfaceBase
{
public:
  NPME_InterfaceReal() : NPME_InterfaceBase()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceReal empty constructor\n";
    #endif
  }
  NPME_InterfaceReal (const NPME_InterfaceReal& rhs) : NPME_InterfaceBase(rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceReal copy constructor\n";
    #endif
  }
  NPME_InterfaceReal& operator= (const NPME_InterfaceReal& rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceReal assignment operator\n";
    #endif

    NPME_InterfaceBase::operator= (rhs);
    return *this;
  }
  virtual ~NPME_InterfaceReal ()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  virtual ~NPME_InterfaceReal\n";
    #endif
  }

  virtual bool ResetCoord (const size_t nCharge, const double *coord, 
        bool printLog, std::ostream& ofs_log) = 0;

  //PME V1 functions for real charges
  virtual bool CalcV (double *V1, const size_t nCharge, 
                  double *charge, double *coord, 
                  bool PRINT, std::ostream& ofs_log) = 0;
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]


  //exact V1 functions (brute force ~N^2 sum)
  virtual bool CalcV_exact (double *V1, const size_t nCharge, 
                  double *charge, double *coord, 
                  bool PRINT, std::ostream& ofs_log) = 0;
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]


  virtual void Print (std::ostream& ofs_log) = 0;

private:

};


class NPME_InterfaceReal_GenFunc : public NPME_InterfaceReal
{
public:
  NPME_InterfaceReal_GenFunc() : NPME_InterfaceReal()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceReal_GenFunc empty constructor\n";
    #endif
  }
  NPME_InterfaceReal_GenFunc (const NPME_InterfaceReal_GenFunc& rhs) 
                : NPME_InterfaceReal(rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceReal_GenFunc copy constructor\n";
    #endif

    _func   = rhs._func;
    _funcLR = rhs._funcLR;
    _funcSR = rhs._funcSR;
    _fself  = rhs._fself;
    _recSum = rhs._recSum;
  }
  NPME_InterfaceReal_GenFunc& operator= (const NPME_InterfaceReal_GenFunc& rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceReal_GenFunc assignment operator\n";
    #endif

    if (this != &rhs)
    {
      NPME_InterfaceReal::operator= (rhs);
      _func   = rhs._func;
      _funcLR = rhs._funcLR;
      _funcSR = rhs._funcSR;
      _fself  = rhs._fself;
      _recSum = rhs._recSum;
    }

    return *this;
  }
  virtual ~NPME_InterfaceReal_GenFunc ()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  virtual ~NPME_InterfaceReal_GenFunc\n";
    #endif
  }

  //PME V1 functions for real charges
  bool CalcV (double *V, const size_t nCharge, 
                  double *charge, double *coord, 
                  bool PRINT, std::ostream& ofs_log);
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]


  //exact V1 functions (brute force ~N^2 sum)
  bool CalcV_exact (double *V, const size_t nCharge, 
                  double *charge, double *coord, 
                  bool PRINT, std::ostream& ofs_log);
  //input:  charge[nCharge](real)
  //output: V1[nCharge][4]



  void Print (std::ostream& ofs_log);

  bool SetUp (const NPME_Library::NPME_KeywordInput& keyword,
    const size_t nCharge, const double *coord, 
    NPME_Library::NPME_KfuncReal *func, 
    NPME_Library::NPME_KfuncReal *funcLR,
    NPME_Library::NPME_KfuncReal *funcSR,
    bool printLog, std::ostream& ofs_log);

  bool ResetCoord (const size_t nCharge, const double *coord, 
    bool printLog, std::ostream& ofs_log);

private:
  NPME_Library::NPME_KfuncReal *_func;
  NPME_Library::NPME_KfuncReal *_funcLR;
  NPME_Library::NPME_KfuncReal *_funcSR;
  double _fself;    //self-interaction smooth kernel 
                    //_fself = (*_funcLR)(0,0,0)


  NPME_Library::NPME_RecSumInterface _recSum;
};


class NPME_InterfaceReal_Laplace_DM : public NPME_InterfaceReal
{
public:
  NPME_InterfaceReal_Laplace_DM() : NPME_InterfaceReal()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceReal_Laplace_DM empty constructor\n";
    #endif
  }
  NPME_InterfaceReal_Laplace_DM (const NPME_InterfaceReal_Laplace_DM& rhs) 
                : NPME_InterfaceReal(rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceReal_Laplace_DM copy constructor\n";
    #endif

    _fself  = rhs._fself;
    _Nder   = rhs._Nder;
    _Rdir   = rhs._Rdir;
    memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(double));
    memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(double));
    _func   = rhs._func;
    _funcLR = rhs._funcLR;
    _funcSR = rhs._funcSR;
    _recSum = rhs._recSum;
    _recSum.Set_KfuncReal (&_funcLR);

  }
  NPME_InterfaceReal_Laplace_DM& operator= 
    (const NPME_InterfaceReal_Laplace_DM& rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceReal_Laplace_DM assignment operator\n";
    #endif

    if (this != &rhs)
    {
      NPME_InterfaceReal::operator= (rhs);
      _fself  = rhs._fself;
      _Nder   = rhs._Nder;
      _Rdir   = rhs._Rdir;
      memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(double));
      memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(double));
      _func   = rhs._func;
      _funcLR = rhs._funcLR;
      _funcSR = rhs._funcSR;
      _recSum = rhs._recSum;
      _recSum.Set_KfuncReal (&_funcLR);
    }

    return *this;
  }
  virtual ~NPME_InterfaceReal_Laplace_DM ()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  virtual ~NPME_InterfaceReal_Laplace_DM\n";
    #endif
  }

  //PME V1 functions for real charges
  bool CalcV (double *V, const size_t nCharge, 
                  double *charge, double *coord, 
                  bool PRINT, std::ostream& ofs_log);
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]


  //exact V1 functions (brute force ~N^2 sum)
  bool CalcV_exact (double *V, const size_t nCharge, 
                  double *charge, double *coord, 
                  bool PRINT, std::ostream& ofs_log);
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]


  bool ResetCoord (const size_t nCharge, const double *coord, 
        bool printLog, std::ostream& ofs_log);

  void Print (std::ostream& ofs_log);

  bool SetUp (const NPME_Library::NPME_KeywordInput& keyword,
    const size_t nCharge, const double *coord,
    bool printLog, std::ostream& ofs_log);

private:

  double _fself;    //self-interaction smooth kernel 
                    //_fself = (*_funcLR)(0,0,0)

  int _Nder;
  double _Rdir;
  double _a[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
  double _b[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]

  NPME_Library::NPME_Kfunc_Laplace        _func;
  NPME_Library::NPME_Kfunc_Laplace_LR_DM  _funcLR;
  NPME_Library::NPME_Kfunc_Laplace_SR_DM  _funcSR;
  NPME_Library::NPME_RecSumInterface      _recSum;
};

class NPME_InterfaceReal_Laplace_Original : public NPME_InterfaceReal
{
public:
  NPME_InterfaceReal_Laplace_Original() : NPME_InterfaceReal()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceReal_Laplace_Original empty constructor\n";
    #endif
  }
  NPME_InterfaceReal_Laplace_Original 
                (const NPME_InterfaceReal_Laplace_Original& rhs) 
                : NPME_InterfaceReal(rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceReal_Laplace_Original copy constructor\n";
    #endif

    _fself  = rhs._fself;
    _beta   = rhs._beta;

    _func   = rhs._func;
    _funcLR = rhs._funcLR;
    _funcSR = rhs._funcSR;
    _recSum = rhs._recSum;
    _recSum.Set_KfuncReal (&_funcLR);

  }
  NPME_InterfaceReal_Laplace_Original& operator= 
    (const NPME_InterfaceReal_Laplace_Original& rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceReal_Laplace_Original ";
      std::cout << "assignment operator\n";
    #endif

    if (this != &rhs)
    {
      NPME_InterfaceReal::operator= (rhs);
      _fself  = rhs._fself;
      _beta   = rhs._beta;
      _func   = rhs._func;
      _funcLR = rhs._funcLR;
      _funcSR = rhs._funcSR;
      _recSum = rhs._recSum;
      _recSum.Set_KfuncReal (&_funcLR);
    }

    return *this;
  }
  virtual ~NPME_InterfaceReal_Laplace_Original ()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  virtual ~NPME_InterfaceReal_Laplace_Original\n";
    #endif
  }

  //PME V1 functions for real charges
  bool CalcV (double *V, const size_t nCharge, 
                  double *charge, double *coord, 
                  bool PRINT, std::ostream& ofs_log);
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]


  //exact V1 functions (brute force ~N^2 sum)
  bool CalcV_exact (double *V, const size_t nCharge, 
                  double *charge, double *coord, 
                  bool PRINT, std::ostream& ofs_log);
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]


  bool ResetCoord (const size_t nCharge, const double *coord, 
        bool printLog, std::ostream& ofs_log);

  void Print (std::ostream& ofs_log);

  bool SetUp (const NPME_Library::NPME_KeywordInput& keyword,
    const size_t nCharge, const double *coord,
    bool printLog, std::ostream& ofs_log);

private:
  double _fself;    //self-interaction smooth kernel 
                    //_fself = (*_funcLR)(0,0,0)

  double _beta;

  NPME_Library::NPME_Kfunc_Laplace              _func;
  NPME_Library::NPME_Kfunc_Laplace_LR_Original  _funcLR;
  NPME_Library::NPME_Kfunc_Laplace_SR_Original  _funcSR;
  NPME_Library::NPME_RecSumInterface            _recSum;
};

class NPME_InterfaceComplex : public NPME_InterfaceBase
{
public:
  NPME_InterfaceComplex() : NPME_InterfaceBase()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceComplex empty constructor\n";
    #endif
  }
  NPME_InterfaceComplex (const NPME_InterfaceComplex& rhs) 
                          : NPME_InterfaceBase(rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceComplex copy constructor\n";
    #endif
  }
  NPME_InterfaceComplex& operator= (const NPME_InterfaceComplex& rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceComplex assignment operator\n";
    #endif

    NPME_InterfaceBase::operator= (rhs);
    return *this;
  }
  virtual ~NPME_InterfaceComplex ()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  virtual ~NPME_InterfaceComplex\n";
    #endif
  }

  virtual bool ResetCoord (const size_t nCharge, const double *coord, 
        bool printLog, std::ostream& ofs_log) = 0;

  //PME V1 functions for real charges
  virtual bool CalcV (_Complex double *V1, const size_t nCharge, 
                  _Complex double *charge, double *coord,
                  bool PRINT, std::ostream& ofs_log) = 0;
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]


  //exact V1 functions (brute force ~N^2 sum)
  virtual bool CalcV_exact (_Complex double *V1, const size_t nCharge, 
                  _Complex double *charge, double *coord,
                  bool PRINT, std::ostream& ofs_log) = 0;
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]


  virtual void Print (std::ostream& ofs_log) = 0;

private:

};

class NPME_InterfaceComplex_GenFunc : public NPME_InterfaceComplex
{
public:
  NPME_InterfaceComplex_GenFunc() : NPME_InterfaceComplex()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceComplex_GenFunc empty constructor\n";
    #endif
  }
  NPME_InterfaceComplex_GenFunc (const NPME_InterfaceComplex_GenFunc& rhs) 
                : NPME_InterfaceComplex(rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceComplex_GenFunc copy constructor\n";
    #endif

    _func   = rhs._func;
    _funcLR = rhs._funcLR;
    _funcSR = rhs._funcSR;
    _fself  = rhs._fself;
    _recSum = rhs._recSum;
  }
  NPME_InterfaceComplex_GenFunc& operator= 
              (const NPME_InterfaceComplex_GenFunc& rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceComplex_GenFunc assignment operator\n";
    #endif

    if (this != &rhs)
    {
      NPME_InterfaceComplex::operator= (rhs);
      _func   = rhs._func;
      _funcLR = rhs._funcLR;
      _funcSR = rhs._funcSR;
      _fself  = rhs._fself;
      _recSum = rhs._recSum;
    }

    return *this;
  }
  virtual ~NPME_InterfaceComplex_GenFunc ()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  virtual ~NPME_InterfaceComplex_GenFunc\n";
    #endif
  }

  //PME V1 functions for real charges
  bool CalcV (_Complex double *V, const size_t nCharge, 
                  _Complex double *charge, double *coord,
                  bool PRINT, std::ostream& ofs_log);
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]


  //exact V1 functions (brute force ~N^2 sum)
  bool CalcV_exact (_Complex double *V, const size_t nCharge, 
                  _Complex double *charge, double *coord,
                  bool PRINT, std::ostream& ofs_log);
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]



  void Print (std::ostream& ofs_log);

  bool SetUp (const NPME_Library::NPME_KeywordInput& keyword,
    const size_t nCharge, const double *coord,
    NPME_Library::NPME_KfuncComplex *func, 
    NPME_Library::NPME_KfuncComplex *funcLR,
    NPME_Library::NPME_KfuncComplex *funcSR,
    bool printLog, std::ostream& ofs_log);

  bool ResetCoord (const size_t nCharge, const double *coord, 
        bool printLog, std::ostream& ofs_log);

private:
  NPME_Library::NPME_KfuncComplex *_func;
  NPME_Library::NPME_KfuncComplex *_funcLR;
  NPME_Library::NPME_KfuncComplex *_funcSR;
  _Complex double _fself;   //self-interaction smooth kernel 
                            //_fself = (*_funcLR)(0,0,0)



  NPME_Library::NPME_RecSumInterface _recSum;
};

class NPME_InterfaceComplex_Helmholtz_DM : public NPME_InterfaceComplex
{
public:
  NPME_InterfaceComplex_Helmholtz_DM() : NPME_InterfaceComplex()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceComplex_Helmholtz_DM empty constructor\n";
    #endif
  }
  NPME_InterfaceComplex_Helmholtz_DM 
          (const NPME_InterfaceComplex_Helmholtz_DM& rhs) 
                : NPME_InterfaceComplex(rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceComplex_Helmholtz_DM copy constructor\n";
    #endif

    _fself  = rhs._fself;
    _Nder   = rhs._Nder;
    _Rdir   = rhs._Rdir;
    _k0     = rhs._k0;
    memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(_Complex double));
    memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(_Complex double));
    _func   = rhs._func;
    _funcLR = rhs._funcLR;
    _funcSR = rhs._funcSR;
    _recSum = rhs._recSum;
    _recSum.Set_KfuncComplex (&_funcLR);

  }
  NPME_InterfaceComplex_Helmholtz_DM& operator= 
    (const NPME_InterfaceComplex_Helmholtz_DM& rhs)
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  NPME_InterfaceComplex_Helmholtz_DM assignment operator\n";
    #endif

    if (this != &rhs)
    {
      NPME_InterfaceComplex::operator= (rhs);
      _fself  = rhs._fself;
      _Nder   = rhs._Nder;
      _Rdir   = rhs._Rdir;
      _k0     = rhs._k0;
      memcpy(_a, rhs._a, (rhs._Nder+1)*sizeof(_Complex double));
      memcpy(_b, rhs._b, (rhs._Nder+1)*sizeof(_Complex double));
      _func   = rhs._func;
      _funcLR = rhs._funcLR;
      _funcSR = rhs._funcSR;
      _recSum = rhs._recSum;
      _recSum.Set_KfuncComplex (&_funcLR);
    }

    return *this;
  }
  virtual ~NPME_InterfaceComplex_Helmholtz_DM ()
  {
    #if NPME_INTERFACE_DEBUG
      std::cout << "  virtual ~NPME_InterfaceComplex_Helmholtz_DM\n";
    #endif
  }

  //PME V1 functions for real charges
  bool CalcV (_Complex double *V, const size_t nCharge, 
                  _Complex double *charge, double *coord,
                  bool PRINT, std::ostream& ofs_log);
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]


  //exact V1 functions (brute force ~N^2 sum)
  bool CalcV_exact (_Complex double *V, const size_t nCharge, 
                  _Complex double *charge, double *coord,
                  bool PRINT, std::ostream& ofs_log);
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]



  void Print (std::ostream& ofs_log);

  bool SetUp (const NPME_Library::NPME_KeywordInput& keyword,
    const size_t nCharge, const double *coord,
    bool printLog, std::ostream& ofs_log);

  bool ResetCoord (const size_t nCharge, const double *coord, 
        bool printLog, std::ostream& ofs_log);

private:
  _Complex double _fself;     //self-interaction smooth kernel 
                              //_fself = (*_funcLR)(0,0,0)

  int _Nder;
  double _Rdir;
  _Complex double _k0;
  _Complex double _a[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]
  _Complex double _b[NPME_MaxDerivMatchOrder+1];  //[_Nder+1]

  NPME_Library::NPME_Kfunc_Helmholtz        _func;
  NPME_Library::NPME_Kfunc_Helmholtz_LR_DM  _funcLR;
  NPME_Library::NPME_Kfunc_Helmholtz_SR_DM  _funcSR;
  NPME_Library::NPME_RecSumInterface        _recSum;
};


struct NPME_KernelList
{
  NPME_Kfunc_Laplace              funcLaplace;          //real kernel
  NPME_Kfunc_Laplace_LR_DM        funcLaplace_LR_DM;    //real kernel
  NPME_Kfunc_Laplace_SR_DM        funcLaplace_SR_DM;    //real kernel
  NPME_Kfunc_Laplace_LR_Original  funcLaplace_LR_Orig;  //real kernel
  NPME_Kfunc_Laplace_SR_Original  funcLaplace_SR_Orig;  //real kernel

  NPME_Kfunc_Ralpha               funcRalpha;           //real kernel
  NPME_Kfunc_Ralpha_LR_DM         funcRalpha_LR_DM;     //real kernel
  NPME_Kfunc_Ralpha_SR_DM         funcRalpha_SR_DM;     //real kernel

  NPME_Kfunc_Helmholtz            funcHelmholtz;         //complex kernel
  NPME_Kfunc_Helmholtz_LR_DM      funcHelmholtz_LR_DM;   //complex kernel
  NPME_Kfunc_Helmholtz_SR_DM      funcHelmholtz_SR_DM;   //complex kernel
};


bool NPME_Interface_SelectKernelPtr (
  NPME_Library::NPME_KfuncReal*& func, 
  NPME_Library::NPME_KfuncReal*& func_LR, 
  NPME_Library::NPME_KfuncReal*& func_SR,
  NPME_Library::NPME_KernelList& kernelList, 
  const NPME_Library::NPME_KeywordInput& keyword, 
  bool printLog, std::ostream& os);
bool NPME_Interface_SelectKernelPtr (
  NPME_Library::NPME_KfuncComplex*& func, 
  NPME_Library::NPME_KfuncComplex*& func_LR, 
  NPME_Library::NPME_KfuncComplex*& func_SR,
  NPME_Library::NPME_KernelList& kernelList, 
  const NPME_Library::NPME_KeywordInput& keyword, 
  bool printLog, std::ostream& os);
//input:  kernelList, keyword
//output: func, func_LR, func_SR
//a) using 'keyword', selects correct kernels from 'kernelList'
//b) set correct kernels with appropriate parameters from 'keyword'
//c) set kernel function pointers ('func', 'func_LR', 'func_SR')
//   to the correct kernel functions contained in kernelList





bool NPME_Interface_SetUpRealKernel (
  NPME_Library::NPME_InterfaceReal*& npme, 
  NPME_Library::NPME_KfuncReal *func, 
  NPME_Library::NPME_KfuncReal *funcLR, 
  NPME_Library::NPME_KfuncReal *funcSR,
  const size_t nCharge, const double *coord,
  const NPME_Library::NPME_KeywordInput& keyword, 
  bool useGenericKernel, bool printLog, std::ostream& ofs_log);
//input:  nCharge, coord[nCharge*3], keyword,
//        func, funcLR, funcSR
//output: npme = base pointer to npme implementation class
//  a) allocates and sets up the appropriate derived npme interface class
//  b) sets 'NPME_InterfaceReal' pointer to the the derived npme interface 
//     class
//there are (optimized) specific kernel implementations for 
//  1) Laplace   + Deriv Match Ewald Splitting
//  2) Laplace   + Original    Ewald Splitting
//  3) Helmholtz + Deriv Match Ewald Splitting
//useGenericKernel = 0 uses the optimized specific kernel implementations
//useGenericKernel = 1 uses the generic            kernel implementations


bool NPME_Interface_SetUpComplexKernel (
  NPME_Library::NPME_InterfaceComplex*& npme, 
  NPME_Library::NPME_KfuncComplex *func, 
  NPME_Library::NPME_KfuncComplex *funcLR, 
  NPME_Library::NPME_KfuncComplex *funcSR,
  const size_t nCharge, const double *coord,
  const NPME_Library::NPME_KeywordInput& keyword, 
  bool useGenericKernel, bool printLog, std::ostream& ofs_log);
//input:  nCharge, coord[nCharge*3], keyword,
//        func, funcLR, funcSR
//output: npme = base pointer to npme implementation class
//  a) allocates and sets up the appropriate derived npme interface class
//  b) sets 'NPME_InterfaceComplex' pointer to the the derived npme interface 
//     class
//there are (optimized) specific kernel implementations for 
//  1) Helmholtz + Deriv Match Ewald Splitting
//useGenericKernel = 0 uses the optimized specific kernel implementations
//useGenericKernel = 1 uses the generic            kernel implementations


}//end namespace NPME_Library


#endif // NPME_INTERFACE_H







