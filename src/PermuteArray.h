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

#ifndef NPME_PERMUTE_ARRAY_H
#define NPME_PERMUTE_ARRAY_H



namespace NPME_Library
{
//******************************************************************************
//******************************************************************************
//********************Permutations Overwriting Array****************************
//******************************************************************************
//******************************************************************************
class NPME_Kcycle
//following class decomposes a symmetric permutation into disjoint K-cycles 
//using group theory
{
public:
  NPME_Kcycle () {  _N = 0; _nKcycle = 0;}

  //copy constructor, assignment operator, and destructor
  NPME_Kcycle (const NPME_Kcycle& rhs);
  NPME_Kcycle& operator= (const NPME_Kcycle& rhs);
  virtual ~NPME_Kcycle ()  {   }



  bool SetPermuteArray (const size_t N, const size_t *P);
  //P[N] array -> Kcycle

  void PrintClass      (std::ostream& os) const;
  void PrintClassShort (std::ostream& os) const;
  bool CheckClass () const;



        size_t  Get_N          ()  const { return _N;              }
        size_t  GetNumKcycle   ()  const { return _nKcycle;        }
  const size_t *GetKcycleSize  ()  const { return &_KcycleSize[0]; }
  const size_t *GetKcycleStart ()  const { return &_KcycleStart[0];}
  const size_t *GetK           ()  const { return &_K[0];          }

private:
  size_t _N;
  size_t _nKcycle;
  std::vector<size_t> _KcycleSize;   //[_nKcycle]
  std::vector<size_t> _KcycleStart;  //[_nKcycle]
  std::vector<size_t> _K;            //[_N]
};

bool NPME_Kcycle_PermuteArray_MN (const size_t M, const size_t N, 
  const NPME_Library::NPME_Kcycle& Kcycle, double *Y, int nProc);
//input:  Y[N][M] and P[N]
//output: Y[N][M] 
//        Y[P(n)][] -> Y[n][]

bool NPME_Kcycle_InversePermuteArray_MN (const size_t M, const size_t N, 
  const NPME_Library::NPME_Kcycle& Kc, double *Y, int nProc);
//input:  Y[N][M] and P[N]
//output: Y[N][M] 
//        Y[n][] -> Y[P(n)][]

bool NPME_Kcycle_2_PermuteArray (const size_t N, 
  const NPME_Kcycle& Kcycle, size_t *P);

//******************************************************************************
//******************************************************************************
//********************Permutations Using a 2nd Copy*****************************
//******************************************************************************
//******************************************************************************
void NPME_PermuteArray_MN (const size_t M, const size_t N, const size_t *P, 
  double *Y, const double *X, int nProc = 1);
//input:  X[N][M] and P[N]
//output: Y[N][M] 
//        Y[n][] = X[P(n)][]

void NPME_PermuteArrayInverse_MN (const size_t M, const size_t N, 
  const size_t *P, double *Y, const double *X, int nProc = 1);
//input:  X[N][M] and P[N]
//output: Y[N][M] 
//        Y[P(n)][] = X[n][]

//***************Random Permutation Operator used for Testing*******************
void NPME_GenerateRandomPermutationOperator (const size_t N, size_t *P);
//used for testing

bool NPME_IsPermutationOperatorValid (const size_t *P, const size_t N);
//1) P[i] must occur once and only once
//2) 0 <= P[i] < N

}//end namespace NPME_Library


#endif // NPME_PERMUTE_ARRAY_H







