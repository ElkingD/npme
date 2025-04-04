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

#ifndef NPME_CONSTANT_H
#define NPME_CONSTANT_H

#include <complex.h>
#include <omp.h>




#define NPME_USE_AVX      1
#define NPME_USE_AVX_512  1
#define NPME_USE_AVX_FMA  1 //some older AVX versions do not have 
                            //fused multiply add (FMA)





namespace NPME_Library
{
//Default values
const double   NPME_Default_FFTmemGB      = 10.0;  //10.0 GB max of FFT memory
const long int NPME_Default_BnOrder       = 8;
const double   NPME_Default_tol           = 1.0E-6;
const size_t   NPME_Default_nDeriv        = 8;
const size_t   NPME_Default_nNeigh        = 2;
const size_t   NPME_Default_nCellClust1D  = 3;





const size_t NPME_Pot_MaxChgBlock_V1    = 1024;   //should be divisible by 8
const int    NPME_MaxDerivMatchOrder    = 30;
const char   NPME_InteractListType      = 'C';
const double NPME_Pot_Xpad              = 1.0E6;
const size_t NPME_IdealRecSumBlockSize  = 20;



//numerical constant
const double NPME_Pi =   3.1415926535897932384626433;

}//end namespace NPME_Library


#endif // NPME_CONSTANT_H







