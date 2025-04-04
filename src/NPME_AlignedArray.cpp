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


#include "NPME_Constant.h"
#include "NPME_AlignedArray.h"


namespace NPME_Library
{
bool NPME_AlignedArrayDouble::resize (long int N, int bitAlign)
{
  if (_size > NPME_AlignedArray_MaxConstantArraySize)
    _mm_free(_A1_mem);

  _bitAlign = bitAlign;
  _size     = N;

  if (_size > NPME_AlignedArray_MaxConstantArraySize)
  {
    _A1_mem = (double *) _mm_malloc (_size*sizeof(double), _bitAlign);
    _A      = _A1_mem;
  }
  else
    _A = _A2_mem;

  if (_A != NULL)
    return true;
  else
    return false;
}

bool NPME_AlignedArrayDoubleComplex::resize (long int N, int bitAlign)
{
  if (_size > NPME_AlignedArray_MaxConstantArraySize)
    _mm_free(_A1_mem);

  _bitAlign = bitAlign;
  _size     = N;

  if (_size > NPME_AlignedArray_MaxConstantArraySize)
  {
    _A1_mem = (_Complex double *) _mm_malloc (
                      _size*sizeof(_Complex double), _bitAlign);
    _A      = _A1_mem;
  }
  else
    _A = _A2_mem;

  if (_A != NULL)
    return true;
  else
    return false;
}


bool NPME_AlignedArrayFloat::resize (long int N, int bitAlign)
{
  if (_size > NPME_AlignedArray_MaxConstantArraySize)
    _mm_free(_A1_mem);

  _bitAlign = bitAlign;
  _size     = N;

  if (_size > NPME_AlignedArray_MaxConstantArraySize)
  {
    _A1_mem = (float *) _mm_malloc (_size*sizeof(float), _bitAlign);
    _A      = _A1_mem;
  }
  else
    _A = _A2_mem;

  if (_A != NULL)
    return true;
  else
    return false;
}


bool NPME_AlignedArrayFloatComplex::resize (long int N, int bitAlign)
{
  if (_size > NPME_AlignedArray_MaxConstantArraySize)
    _mm_free(_A1_mem);

  _bitAlign = bitAlign;
  _size     = N;

  if (_size > NPME_AlignedArray_MaxConstantArraySize)
  {
    _A1_mem = (_Complex float *) _mm_malloc (
                        _size*sizeof(_Complex float), _bitAlign);
    _A      = _A1_mem;
  }
  else
    _A = _A2_mem;

  if (_A != NULL)
    return true;
  else
    return false;
}

bool NPME_AlignedArrayLongInt::resize (long int N, int bitAlign)
{
  if (_size > NPME_AlignedArray_MaxConstantArraySize)
    _mm_free(_A1_mem);

  _bitAlign = bitAlign;
  _size     = N;

  if (_size > NPME_AlignedArray_MaxConstantArraySize)
  {
    _A1_mem = (long int *) _mm_malloc (_size*sizeof(long int), _bitAlign);
    _A      = _A1_mem;
  }
  else
    _A = _A2_mem;

  if (_A != NULL)
    return true;
  else
    return false;
}

}//end namespace NPME_Library



