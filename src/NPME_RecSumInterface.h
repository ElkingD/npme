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

#ifndef NPME_REC_SUM_INTERFACE_H
#define NPME_REC_SUM_INTERFACE_H

#include <vector>

#include "NPME_KernelFunction.h"


namespace NPME_Library
{


class NPME_RecSumInterface
{
public:
  NPME_RecSumInterface ();

  //copy constructor, assignment operator, and destructor
  NPME_RecSumInterface (const NPME_RecSumInterface& rhs);
  NPME_RecSumInterface& operator= 
        (const NPME_RecSumInterface& rhs);
  virtual ~NPME_RecSumInterface ()  {   }




  bool SetUp (const size_t nCharge, const double *coord, 
    const long int BnOrder, bool useSphereSymmCompactF, 
    const long int N1, const long int N2, const long int N3,
    const long int n1, const long int n2, const long int n3,
    const double del1, const double del2, const double del3,
    const double a1,   const double a2,   const double a3,
    NPME_KfuncReal *funcLR, int nProc, 
    bool PRINT, std::ostream& os);

  bool SetUp (const size_t nCharge, 
    const double *coord, const long int BnOrder, bool useSphereSymmCompactF, 
    const long int N1, const long int N2, const long int N3,
    const long int n1, const long int n2, const long int n3,
    const double del1, const double del2, const double del3,
    const double a1,   const double a2,   const double a3,
    NPME_KfuncComplex *funcLR, int nProc, 
    bool PRINT, std::ostream& os);

  void Print (std::ostream& os) const;

  //PME V1 functions for real/complex charges
  bool CalcV1 (double *V1, const size_t nCharge, 
          const double *charge, const double *coord, bool zeroVarray, 
          bool PRINT, std::ostream& os);
  bool CalcV1 (_Complex double *V1, const size_t nCharge, 
          const _Complex double *charge, const double *coord, 
          bool zeroVarray, bool PRINT, std::ostream& os);
  //input:  charge[nCharge] (real or complex)
  //output: V1[nCharge][4]




  //exact V1 functions (brute force ~N^2 sum)
  bool CalcV1_exact (double *V1, const size_t nCharge, 
          const double *charge, const double *coord, int vecOption);
  bool CalcV1_exact (_Complex double *V1, const size_t nCharge, 
          const _Complex double *charge, const double *coord, int vecOption);
  //input:  charge[nCharge] (real or complex)
  //        vecOption = 0, 1, 2 (for no vectorization, AVX, or AVX-512)
  //output: V1[nCharge][4]




  bool      IsSetUp         ()  const { return _isSetUp;    }
  int       GetNumProc      ()  const { return _nProc;      }

  long int  Get_BnOrder     ()  const { return _BnOrder;    }

  long int  Get_N1          ()  const { return _N1;         }
  long int  Get_N2          ()  const { return _N2;         }
  long int  Get_N3          ()  const { return _N3;         }

  long int  Get_n1          ()  const { return _n1;         }
  long int  Get_n2          ()  const { return _n2;         }
  long int  Get_n3          ()  const { return _n3;         }

  double    Get_a1          ()  const { return _a1;         }
  double    Get_a2          ()  const { return _a2;         }
  double    Get_a3          ()  const { return _a3;         }

  double    Get_del1        ()  const { return _del1;       }
  double    Get_del2        ()  const { return _del2;       }
  double    Get_del3        ()  const { return _del3;       }

  double    Get_X           ()  const { return _X;          }
  double    Get_Y           ()  const { return _Y;          }
  double    Get_Z           ()  const { return _Z;          }

  double    Get_L1          ()  const { return _L1;         }
  double    Get_L2          ()  const { return _L2;         }
  double    Get_L3          ()  const { return _L3;         }

  const double *Get_Rtrans  ()  const { return _R;          }

  void Set_KfuncReal (NPME_KfuncReal *funcLR)
  {
    _funcReal = funcLR;
  }
  void Set_KfuncComplex (NPME_KfuncComplex *funcLR)
  {
    _funcComplex = funcLR;
  }

  bool AllocateFFTmem (bool useSphereSymmCompactF, 
    const long int N1, const long int N2, const long int N3,
    const long int n1, const double maxFFTmemGB, bool PRINT, std::ostream& os);
  //function is useful in testing for pre-allocating the 2 FFT arrays once
  //instead of multiple re-allocations.  not necessary to call


private:

  bool SetUp (const size_t nCharge, 
      const double *coord, const long int BnOrder, bool useSphereSymmCompactF, 
      const long int N1, const long int N2, const long int N3,
      const long int n1, const long int n2, const long int n3,
      const double del1, const double del2, const double del3,
      const double a1,   const double a2,   const double a3,
      int nProc, bool PRINT, std::ostream& os);

  bool _isSetUp;

  bool          _useRealFunc;
  NPME_KfuncReal    *_funcReal;
  NPME_KfuncComplex *_funcComplex;


  size_t        _nCharge;

  int       _nProc;
  long int  _BnOrder;               //PME B-spline order
  double    _a1,    _a2,    _a3;    //Fourier extension parameters
  double    _del1,  _del2,  _del3;  //Fourier extension parameters
  long int  _N1, _N2, _N3;          //FFT size
  long int  _n1, _n2, _n3;          //block size
  double    _L1, _L2, _L3;          //L1 = 2*(_X + _del1)
                                    //L2 = 2*(_Y + _del2)
                                    //L3 = 2*(_Z + _del3)
  double    _X,  _Y,  _Z;           //B-spline corrected volume parameters
  double    _R[3];                  //translation std::vector to center coord


  //block transform parameters
  size_t _nNonZeroBlock;
  std::vector<long int> _NonZeroBlock2BlockIndex_XYZ;
  std::vector<size_t> _nNonZeroBlock2Charge;
  std::vector<size_t*> _NonZeroBlock2Charge;
  std::vector<size_t>  _NonZeroBlock2Charge1D;

  std::vector<double> _lamda1_2;    //[N1]
  std::vector<double> _lamda2_2;    //[N2]
  std::vector<double> _lamda3_2;    //[N3]

  std::vector<_Complex double> _C1;
  std::vector<_Complex double> _C2;
  std::vector<_Complex double> _C3;

  bool _useSphereSymmCompactF;
  std::vector<_Complex double> _F;   //[N1*N2*N3]   (_useSphereSymmCompactF = 0)
                                     //[N1*N2*N3/8] (_useSphereSymmCompactF = 1)
  std::vector<_Complex double> _Xtmp;
};

double NPME_CalcFFTmemGB (
  const long int N1, const long int N2, const long int N3,
  const long int n1, bool useSphereSymmCompactF);
//input:  N1, N2, N3            = FFT sizes
//        n1                    = FFT block size in x direction
//        useSphereSymmCompactF = 0, 1 (set to 1 for a single box radially 
//                                      symmetric kernel and 0 otherwise)
//output: FFT memory in GB



}//end namespace NPME_Library


#endif // NPME_REC_SUM_INTERFACE_H






