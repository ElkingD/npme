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

#ifndef NPME_ALIGNED_ARRAY_H
#define NPME_ALIGNED_ARRAY_H

#include "Constant.h"
#include "ExtLibrary.h"

namespace NPME_Library
{
//examples

//NPME_AlignedArrayDouble A;
//A.resize(N, 32)
//double *Aptr = A.GetPtr()       <- do this after size has been set

//NPME_AlignedArrayDouble A(N, 64);
//double *Aptr = A.GetPtr()       <- do this after size has been set


//const size_t NPME_AlignedArray_MaxConstantArraySize = 4*NPME_Pot_MaxChgBlock_V1;
//const size_t NPME_AlignedArray_MaxConstantArraySize = 4*NPME_Pot_MaxChgBlock_V1;

//const size_t NPME_AlignedArray_MaxConstantArraySize = 10000;

const size_t NPME_AlignedArray_MaxConstantArraySize = 0;

class NPME_AlignedArrayDouble
{
public:
  NPME_AlignedArrayDouble ()
  {
    _size     = 0;
    _bitAlign = 64;
    _A        = NULL;
  }
  NPME_AlignedArrayDouble (long int size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize (size, bitAlign);
  }
  NPME_AlignedArrayDouble (int size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize ((long int) size, bitAlign);
  }
  NPME_AlignedArrayDouble (size_t size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize ((long int) size, bitAlign);
  }
 ~NPME_AlignedArrayDouble () { clear (); }

  bool resize (long int N, int bitAlign);
  bool resize (int N,    int bitAlign)
  {
    return resize( (long int) N, bitAlign);
  }
  bool resize (size_t N, int bitAlign)
  {
    return resize( (long int) N, bitAlign);
  }
  
  void clear ()
  {
    if (_size > NPME_AlignedArray_MaxConstantArraySize)
      NPME_free(_A1_mem);
  }

  int       GetBitAlign ()  { return _bitAlign;       }
  long int  GetSize     ()  { return _size;           }
  double   *GetPtr      ()  { return _A;              }


private:
  int _bitAlign;  //32 or 64
  long int _size;
  double *_A1_mem;
  double  _A2_mem[NPME_AlignedArray_MaxConstantArraySize]
     __attribute__((aligned(64)));
  double *_A;
};

class NPME_AlignedArrayDoubleComplex
{
public:
  NPME_AlignedArrayDoubleComplex ()
  {
    _size     = 0;
    _bitAlign = 64;
    _A        = NULL;
  }
  NPME_AlignedArrayDoubleComplex (long int size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize (size, bitAlign);
  }
  NPME_AlignedArrayDoubleComplex (int size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize ((long int) size, bitAlign);
  }
  NPME_AlignedArrayDoubleComplex (size_t size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize ((long int) size, bitAlign);
  }
 ~NPME_AlignedArrayDoubleComplex () { clear (); }

  bool resize (long int N, int bitAlign);
  bool resize (int N,    int bitAlign)
  {
    return resize( (long int) N, bitAlign);
  }
  bool resize (size_t N, int bitAlign)
  {
    return resize( (long int) N, bitAlign);
  }

  void clear ()
  {
    if (_size > NPME_AlignedArray_MaxConstantArraySize)
      NPME_free(_A1_mem);
  }

  int               GetBitAlign ()  { return _bitAlign;       }
  long int          GetSize     ()  { return _size;           }
  _Complex double  *GetPtr      ()  { return _A;              }


private:
  int _bitAlign;  //32 or 64
  long int _size;
  _Complex double *_A1_mem;
  _Complex double  _A2_mem[NPME_AlignedArray_MaxConstantArraySize]
     __attribute__((aligned(64)));
  _Complex double *_A;
};




class NPME_AlignedArrayFloat
{
public:
  NPME_AlignedArrayFloat ()
  {
    _size     = 0;
    _bitAlign = 64;
    _A        = NULL;
  }
  NPME_AlignedArrayFloat (long int size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize (size, bitAlign);
  }
  NPME_AlignedArrayFloat (int size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize ((long int) size, bitAlign);
  }
  NPME_AlignedArrayFloat (size_t size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize ((long int) size, bitAlign);
  }
 ~NPME_AlignedArrayFloat () { clear (); }

  bool resize (long int N, int bitAlign);
  bool resize (int N,    int bitAlign)
  {
    return resize( (long int) N, bitAlign);
  }
  bool resize (size_t N, int bitAlign)
  {
    return resize( (long int) N, bitAlign);
  }
  
  void clear ()
  {
    if (_size > NPME_AlignedArray_MaxConstantArraySize)
      NPME_free(_A1_mem);
  }

  int       GetBitAlign ()  { return _bitAlign;       }
  long int  GetSize     ()  { return _size;           }
  float    *GetPtr      ()  { return _A;              }


private:
  int _bitAlign;  //32 or 64
  long int _size;
  float *_A1_mem;
  float  _A2_mem[NPME_AlignedArray_MaxConstantArraySize]
     __attribute__((aligned(64)));
  float *_A;
};

class NPME_AlignedArrayFloatComplex
{
public:
  NPME_AlignedArrayFloatComplex ()
  {
    _size     = 0;
    _bitAlign = 64;
    _A        = NULL;
  }
  NPME_AlignedArrayFloatComplex (long int size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize (size, bitAlign);
  }
  NPME_AlignedArrayFloatComplex (int size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize ((long int) size, bitAlign);
  }
  NPME_AlignedArrayFloatComplex (size_t size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize ((long int) size, bitAlign);
  }
 ~NPME_AlignedArrayFloatComplex () { clear (); }

  bool resize (long int N, int bitAlign);
  bool resize (int N,    int bitAlign)
  {
    return resize( (long int) N, bitAlign);
  }
  bool resize (size_t N, int bitAlign)
  {
    return resize( (long int) N, bitAlign);
  }
  
  void clear ()
  {
    if (_size > NPME_AlignedArray_MaxConstantArraySize)
      NPME_free(_A1_mem);
  }

  int              GetBitAlign ()  { return _bitAlign;       }
  long int         GetSize     ()  { return _size;           }
  _Complex float  *GetPtr      ()  { return _A;              }


private:
  int _bitAlign;  //32 or 64
  long int _size;
  _Complex float *_A1_mem;
  _Complex float  _A2_mem[NPME_AlignedArray_MaxConstantArraySize]
     __attribute__((aligned(64)));
  _Complex float *_A;
};

class NPME_AlignedArrayLongInt
{
public:
  NPME_AlignedArrayLongInt ()
  {
    _size     = 0;
    _bitAlign = 64;
    _A        = NULL;
  }
  NPME_AlignedArrayLongInt (long int size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize (size, bitAlign);
  }
  NPME_AlignedArrayLongInt (int size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize ((long int) size, bitAlign);
  }
  NPME_AlignedArrayLongInt (size_t size, int bitAlign)
  {
    _size     = 0;
    _bitAlign = 64;
    resize ((long int) size, bitAlign);
  }
 ~NPME_AlignedArrayLongInt () { clear (); }

  bool resize (long int N, int bitAlign);
  bool resize (int N,    int bitAlign)
  {
    return resize( (long int) N, bitAlign);
  }
  bool resize (size_t N, int bitAlign)
  {
    return resize( (long int) N, bitAlign);
  }

  void clear ()
  {
    if (_size > NPME_AlignedArray_MaxConstantArraySize)
      NPME_free(_A1_mem);
  }

  int        GetBitAlign ()  { return _bitAlign;       }
  long int   GetSize     ()  { return _size;           }
  long int  *GetPtr      ()  { return _A;              }


private:
  int _bitAlign;  //32 or 64
  long int _size;
  long int *_A1_mem;
  long int  _A2_mem[NPME_AlignedArray_MaxConstantArraySize]
     __attribute__((aligned(64)));
  long int *_A;
};



}//end namespace NPME_Library


#endif //NPME_ALIGNED_ARRAY_H






