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

#ifndef NPME_READ_PRINT_H
#define NPME_READ_PRINT_H

#include <map>
#include <string>
#include <sstream>
#include <vector>

namespace NPME_Library
{



class NPME_KeywordInput
//a class to store and access (keywords, value) used for program instructions
//  keyword = string
//  value   = string (can be converted to integers or real numbers)
{
  public:
  //default constructor and input file constructor
  NPME_KeywordInput ()  {   }
  NPME_KeywordInput (const std::string& inputFile)
  {
    ReadInputKeyword(inputFile);
  }

  //copy constructor, assignment operator, and destructor
  NPME_KeywordInput (const NPME_KeywordInput& rhs)
  {
    _token = rhs._token;
  }
  NPME_KeywordInput& operator= (const NPME_KeywordInput& rhs)
  {
    if (this != &rhs)
    {
      _token = rhs._token;
    }
    return *this;
  }
  virtual ~NPME_KeywordInput ()  {   }

  //Read keywords from file, clear keywords, and print keywords
  bool ReadInputKeyword (const std::string& inputFile);
  void ClearKeyword ()  { _token.clear(); }
  void PrintKeywords (std::ostream& os)     const;
  void PrintKeywords (const std::string& filename) const;
  

  //add keywords manually
  void AddKeyword (const std::string& keyword, const std::string& value)
  {
    _token[keyword] = value;
  }
  template <class T1, class T2>
  void AddKeyword (const T1& keyword, const T2& value)
  {
    std::stringstream ss1;
    std::stringstream ss2;
    ss1 << keyword;  
    ss2 << value;
    _token[ss1.str()] = ss2.str();
  }

  //input:  keyword string
  //output: value (e.g. int, double, string, etc.)
  //returns true/false if keyword is present
  template <class T>
  bool GetValue (const std::string& keyword, T& value) const
  {
    std::map<std::string, std::string>::const_iterator it;
    it = _token.find (keyword);
    if (it == _token.end())
      return false;

    std::stringstream ss (it->second);
    ss >> value;
    return true;
  }
  template <class T>
  bool GetValue (const char *keyword, T& value) const
  {
    std::string sKeyword(keyword);
    return GetValue (sKeyword, value);
  }

private:
  std::map<std::string, std::string> _token;
};

//******************************Basic I/O Functions*****************************
bool NPME_ParseLine (const std::string& strLine, 
  std::vector<std::string>& strWords);


bool NPME_GoToLine (std::ifstream& ifs, const char *flag, 
  const std::string& filename, const bool printError);
//skips to line where the flag char string is found

bool NPME_SkipInput2Char (const char a, std::istream& ifs, 
  const std::string& filename, bool printError);
//skips input stream to character a

bool NPME_SkipInput2NextLine (std::istream& ifs, const std::string& filename,
  bool printError);
//skips input stream to next line

void NPME_PrintVec3 (std::ostream& os, const char *str, const double A[3]);

void NPME_RemoveExtension (const std::string& s1, std::string& s2);
//input:  s1 = "filename.ext"
//output: s2 = "filename"



//***************Read/Write Single Box Coords, Charges, Potential***************
void NPME_SetFilenames (std::string& logFile, 
        std::string& V_pme_File,      std::string& V_ref_File, 
  const std::string& coordFile, const std::string& keywordFile);
//input:  coordFile, keywordFile
//output: logFile, V_pme_File, V_ref_File


bool NPME_ReadSingleBoxChargeCoord (const std::string& filename,
  size_t& nCharge, bool& isChargeReal,
  std::vector<double>& coord, std::vector<double>& chargeReal, 
  std::vector<_Complex double>& chargeComplex);
//coord[3*nCharge]
//charge[nCharge]


bool NPME_ReadSingleBoxChargeCoordReal (const std::string& filename,
  size_t& nCharge, std::vector<double>& coord, std::vector<double>& charge);
//coord[3*nCharge]
//charge[nCharge]

bool NPME_ReadSingleBoxChargeCoordComplex (const std::string& filename,
  size_t& nCharge, std::vector<double>& coord, 
  std::vector<_Complex double>& charge);
//coord[3*nCharge]
//charge[nCharge]



void NPME_PrintSingleBoxChargeCoord (std::ostream& os,
  const size_t nCharge, 
  const bool isChargeReal, const std::vector<double>& coord, 
  const std::vector<double>& chargeReal,
  const std::vector<_Complex double>& chargeComplex);


void NPME_PrintBoxInfo (std::ostream& os, const size_t nCharge, 
  const bool isChargeReal);


void NPME_PrintSingleBox_V (const std::string& filename,
  const size_t nCharge, 
  const std::vector<double>& Vreal,  bool printHighPrecision);
void NPME_PrintSingleBox_V (const std::string& filename,
  const size_t nCharge, 
  const std::vector<_Complex double>& Vcomplex, bool printHighPrecision);
//V has size  V[nCharge][4]
//prints      V[nCharge][4]



bool NPME_ReadSingleBox_V (const std::string& filename,
  size_t& nCharge, std::vector<double>& V);
bool NPME_ReadSingleBox_V (const std::string& filename,
  size_t& nCharge, std::vector<_Complex double>& V);
bool NPME_ReadSingleBox_V (const std::string& filename,
  bool& isChargeReal, size_t& nCharge, 
  std::vector<double>& Vreal, std::vector<_Complex double>& Vcomplex);
//reads  V[nCharge][4]




}//end namespace NPME_Library


#endif // NPME_READ_PRINT_H



