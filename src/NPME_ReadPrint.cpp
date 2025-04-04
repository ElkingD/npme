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
#include <cstdio>

#include <iostream> 
#include <fstream> 
#include <sstream>
#include <vector>
#include <string>
#include <map>


#include "NPME_Constant.h"
#include "NPME_ReadPrint.h"

#include "NPME_SupportFunctions.h"
#include "NPME_RecSumSupportFunctions.h"




namespace NPME_Library
{
bool NPME_KeywordInput::ReadInputKeyword (const std::string& filename)
{
  using std::cout;
  using std::endl;
  using std::ifstream;
  using std::string;

  ifstream ifs(filename.c_str());
  if (ifs.fail())
  {
    cout << "Error opening " << filename << "\n";
    return false;
  }

  string strLine;
  std::vector<std::string> strWords;

  while (!ifs.eof())
  {
    getline (ifs, strLine);
    //cout << "strLine = " << strLine << endl;
    NPME_ParseLine (strLine, strWords);
    if (strWords.size() >= 2)
      _token[strWords[0]] = strWords[1];
  }

  return true;
}


void NPME_KeywordInput::PrintKeywords (std::ostream& os) const
{
  using std::ios;
  using std::endl;

  const size_t n  = 15;
  const char *s1  = "      ";
  const char *s2  = " ";

  os << "@Keyword\n";
  std::map<std::string, std::string>::const_iterator it;
  for (it = _token.begin(); it != _token.end(); it++)
  {
    os << s1; os.width(n); os.setf(ios::left); os << it->first;
    os << s2; os.width(n); os.setf(ios::left); os << it->second << endl;
  }
  os << "\n\n";
}

void NPME_KeywordInput::PrintKeywords (const std::string& filename) const
{
  std::cout << "printing " << filename << std::endl;
  std::ofstream ofs(filename.c_str());

  if (ofs.fail())
  {
    std::cout << "Error opening " << filename << std::endl;
    exit(0);
  }

  PrintKeywords (ofs);
}


//******************************************************************************
//******************************************************************************
//******************************Basic I/O Functions*****************************
//******************************************************************************
//******************************************************************************

bool NPME_SkipInput2Char (const char a, std::istream& ifs, 
  const std::string& filename, bool printError)
//skips input stream to character a
{
  while (!ifs.eof())
  {
    char b;
    ifs.get(b);
    if (a == b)
      return true;
  }

  if (printError)
  {
    std::cout << "Error.  unexpected eof in file " << filename << std::endl;
  }
  return false;
}
bool NPME_SkipInput2NextLine (std::istream& ifs, const std::string& filename,
  bool printError)
{
  return NPME_SkipInput2Char ('\n', ifs, filename, printError);
}

bool NPME_GoToLine (std::ifstream& ifs, const char *flag, 
  const std::string& filename, const bool printError)
//skips to line where the flag char string is found
{
  std::string sFlag (flag);

  while (!ifs.eof())
  {
    std::string str;
    ifs >> str;
    NPME_SkipInput2NextLine (ifs, filename, printError);
    if (str == sFlag)
      return true;
  }
  if (printError)
  {
    std::cout << "Unexpected end of file while search for " << flag;
    std::cout << " in file " << filename << std::endl;
  }
  return false;
}

bool NPME_ParseLine (const std::string& strLine, 
  std::vector<std::string>& strWords)
{
  using std::string;
  using std::stringstream;

  strWords.clear();

  stringstream sof(strLine);
  while (!sof.eof())
  {
    string strTmp;
    getline(sof, strTmp, ' ');
    if (strTmp.size() > 0)
      strWords.push_back(strTmp);
  }


//for (size_t i = 0; i < strWords.size(); i++)
//{
//  std::cout << "      " << i << " " << strWords[i].size() << " ";
//  std::cout << strWords[i] << std::endl;
//}


  return true;
}



void NPME_RemoveExtension (const std::string& s1, std::string& s2)
//input:  s1 = "filename.ext"
//output: s2 = "filename"
{
  for (size_t i = 0; i < s1.size(); i++)
  {
    if ( (s1[i] == '.') || (s1[i] == '\0') )
      break;
    s2.push_back(s1[i]);
  }
}

//******************************************************************************
//******************************************************************************
//***************Read/Write Single Box Coords, Charges, Potential***************
//******************************************************************************
//******************************************************************************

void NPME_SetFilenames (std::string& logFile, 
        std::string& V_pme_File,      std::string& V_ref_File, 
  const std::string& coordFile, const std::string& keywordFile)
//input:  coordFile, keywordFile
//output: logFile, V_pme_File, V_ref_File
{
  std::string initial1;
  std::string initial2;

  NPME_RemoveExtension (coordFile,   initial1);
  NPME_RemoveExtension (keywordFile, initial2);

  logFile    = initial1 + "_" + initial2 + ".log";
  V_pme_File = initial1 + "_" + initial2 + "_V_pme.output";
  V_ref_File = initial1 + "_" + initial2 + "_V_exact.output";
}


void NPME_PrintBoxInfo (std::ostream& os, const size_t nCharge, 
  const bool isChargeReal)
{
  char str[2000];

  os << "@BoxInfo\n";
  sprintf(str, "  nCharge       = %lu\n", nCharge);
  os << str;
  sprintf(str, "  isChargeReal  = %d\n", (int) isChargeReal);
  os << str;

  os.flush();
}

void NPME_PrintSingleBoxChargeCoord (std::ostream& os,
  const size_t nCharge, 
  const bool isChargeReal, const std::vector<double>& coord, 
  const std::vector<double>& chargeReal,
  const std::vector<_Complex double>& chargeComplex)
{
  char str[500];

  os << "@Coord\n";
  os << "  nCharge " << nCharge << "\n";
  for (size_t i = 0; i < nCharge; i++)
  {
    sprintf(str, "    %10.6f %10.6f %10.6f\n", coord[3*i], 
        coord[3*i+1], coord[3*i+2]);
    os << str;
  }
  os << "\n\n";

  const double *q;
  size_t nTot;
  if (isChargeReal)
  {
    os << "@ChargeReal\n";
    q     = &chargeReal[0];
    nTot  = nCharge;
  }
  else
  {
    os << "@ChargeComplex\n";
    q     = (double*) &chargeComplex[0];
    nTot  = 2*nCharge;
  }

  sprintf(str, "  nCharge %lu", nCharge);
  os << str;

  size_t Qindex0 = 0;
  for (size_t i = 0; i < nTot; i++)
  {
    if (Qindex0%5 == 0)
      os << "\n    ";
    sprintf(str, "%14.6le ", q[i]);
    os << str;
    Qindex0++;
  }
  os << "\n\n";   
}


bool NPME_ReadSingleBoxChargeCoord (const std::string& filename,
  size_t& nCharge, bool& isChargeReal,
  std::vector<double>& coord, std::vector<double>& chargeReal, 
  std::vector<_Complex double>& chargeComplex)
//coord[3*nCharge]
//charge[nCharge]
{
  using std::cout;
  using std::endl;
  using std::ifstream;
  using std::string;

  bool printError = 1;
  string sTmp;
  string strLine;

  ifstream ifs(filename.c_str());
  if (ifs.fail())
  {
    cout << "Error opening " << filename << "\n";
    return false;
  }

  if (!NPME_GoToLine (ifs, "@Coord", filename, printError)) return false;
  ifs >> sTmp;
  if (sTmp != "nCharge")
  {
    cout << "Error reading " << filename << ".\n";
    cout << "while reading @Coord, expecting:\n";
    cout << "  nCharge XX\n";
    return false;
  }

  ifs >> nCharge;
  if (!NPME_SkipInput2NextLine (ifs, filename, printError)) return false;

  coord.resize(3*nCharge);
  for (size_t i = 0; i < nCharge; i++)
  {
    if (ifs.eof())
    {
      cout << "Error reading " << filename << ".\n";
      cout << "  unexpected end of file while reading coordinates\n";
      return false;
    }
    for (size_t p = 0; p < 3; p++)
      ifs >> coord[3*i+p];
    if (!NPME_SkipInput2NextLine (ifs, filename, printError))
      return false;
  }

  bool foundCharge = 0;
  while (!ifs.eof())
  {
    getline (ifs, strLine);
    if (strLine[0] == '@')
    {
      std::stringstream sof(strLine);
      sof >> sTmp;
      if (sTmp == "@ChargeReal")
      {
        foundCharge  = 1;
        isChargeReal = 1;
        break;
      }
      else if (sTmp == "@ChargeComplex")
      {
        foundCharge  = 1;
        isChargeReal = 0;
        break;
      }
    }
  }
  if (!foundCharge)
  {
    cout << "Error reading " << filename << ".\n";
    cout << "  did not find @ChargeReal or @ChargeComplex flags\n";
    return false;
  }


  //read nCharge
  getline (ifs, strLine);
  std::vector<std::string> strWords;
  NPME_ParseLine (strLine, strWords);

  {
    bool foundError = 0;
    if (strWords.size() < 2)
      foundError = 1;
    else
    {
      if (strWords[0] != "nCharge")
        foundError = 1;
    }
    if (foundError)
    {
      cout << "Error reading " << filename << ".\n";
      cout << "  after finding @ChargeReal/@ChargeComplex, expecting to find:\n";
      cout << "  nCharge XX\n";
      cout << "found: " << strLine << "\n";
      return false;
    }

    std::stringstream sof1(strWords[1]);
    sof1 >> nCharge;
  }


  double *q;
  size_t nTot;
  if (isChargeReal)
  {
    chargeReal.resize(nCharge);
    q     = &chargeReal[0];
    nTot  = nCharge;
  }
  else
  {
    chargeComplex.resize(nCharge);
    q     = (double*) &chargeComplex[0];
    nTot  = 2*nCharge;
  }
  
  {
    const size_t remain = nTot%5;
    const size_t nLoop  = (nTot - remain)/5;
    double qTmp;

    size_t Qindex = 0;
    for (size_t i = 0; i < nLoop; i++)
    {
      if (ifs.eof())
      {
        cout << "Error reading " << filename << ".\n";
        cout << "  unexpected end of file while reading charges\n";
        return false;
      }

      for (size_t p = 0; p < 5; p++)
      {
        ifs >> qTmp;
        q[Qindex] = qTmp;
        Qindex++;
      }
        
      if (!NPME_SkipInput2NextLine (ifs, filename, printError))
        return false;
    }
    if (remain > 0)
    {
      for (size_t p = 0; p < remain; p++)
      {
        ifs >> qTmp;
        q[Qindex] = qTmp;
        Qindex++;
      }
      if (!NPME_SkipInput2NextLine (ifs, filename, printError))
        return false;
    }
  }

  return true;
}


bool NPME_ReadSingleBoxChargeCoordReal (const std::string& filename,
  size_t& nCharge, std::vector<double>& coord, std::vector<double>& charge)
//coord[3*nCharge]
//charge[nCharge]
{
  bool isChargeReal;
  std::vector<_Complex double> chargeComplex;

  if (!NPME_ReadSingleBoxChargeCoord (filename, nCharge, isChargeReal,
    coord, charge, chargeComplex))
    return false;

  if (!isChargeReal)
  {
    std::cout << "Error in NPME_ReadSingleBoxChargeCoordReal.  isChargeReal = 0\n";
    return false;
  }

  return true;
}

bool NPME_ReadSingleBoxChargeCoordComplex (const std::string& filename,
  size_t& nCharge, std::vector<double>& coord, 
  std::vector<_Complex double>& charge)
//coord[3*nCharge]
//charge[nCharge]
{
  bool isChargeReal;
  std::vector<double> chargeReal;

  if (!NPME_ReadSingleBoxChargeCoord (filename, nCharge, isChargeReal,
    coord, chargeReal, charge))
    return false;

  if (isChargeReal)
  {
    std::cout << "Error in NPME_ReadSingleBoxChargeCoordComplex.  isChargeReal = 1\n";
    return false;
  }

  return true;
}




void NPME_PrintSingleBox_V (const std::string& filename,
  const size_t nCharge, 
  const std::vector<double>& Vreal,  bool printHighPrecision)
//V has size  V[nCharge][4]
//prints      V[nCharge][4]

{
  char str[2000];
  using std::cout;
  using std::ofstream;

  ofstream os(filename.c_str());

  os << "@Vreal\n";
  sprintf(str, "  nCharge %lu", nCharge);
  os << str;

  size_t count = 0;
  for (size_t n = 0; n < nCharge;     n++)

  for (size_t p = 0; p < 4;           p++)
  {
    double Vloc = Vreal[n*4 + p];
    if (count%4 == 0)
      os << "\n    ";

    if (printHighPrecision)
      sprintf(str, "%24.16le ", Vloc);
    else
      sprintf(str, "%18.10le ", Vloc);
    os << str;

    count++;
  }

  os << "\n\n";   
}
void NPME_PrintSingleBox_V (const std::string& filename,
  const size_t nCharge, 
  const std::vector<_Complex double>& Vcomplex, bool printHighPrecision)
//V has size  V[nCharge][4]
//prints      V[nCharge][4]
{
  char str[2000];
  using std::cout;
  using std::ofstream;

  ofstream os(filename.c_str());

  os << "@Vcomplex\n";
  sprintf(str, "  nCharge %lu", nCharge);
  os << str;

  const double *Vtmp = (const double *) &Vcomplex[0];

  size_t count = 0;
  for (size_t n = 0; n < nCharge;     n++)
  for (size_t p = 0; p < 8;           p++)
  {
    double Vloc = Vtmp[n*8 + p];
    if (count%4 == 0)
      os << "\n    ";

    if (printHighPrecision)
      sprintf(str, "%24.16le ", Vloc);
    else
      sprintf(str, "%18.10le ", Vloc);
    os << str;

    count++;
  }

  os << "\n\n";   
}




bool NPME_ReadSingleBox_V (const std::string& filename,
  size_t& nCharge, std::vector<double>& V)
{
  using std::cout;
  using std::ifstream;
  using std::string;
  using std::vector;

  bool printError = 1;
  string sTmp;
  char str[2000];

  ifstream ifs(filename.c_str());
  if (ifs.fail())
  {
    cout << "Error opening " << filename << "\n";
    return false;
  }

  if (!NPME_GoToLine (ifs, "@Vreal", filename, printError)) return false;

  ifs >> sTmp;
  if (sTmp != "nCharge")
  {
    cout << "Error reading " << filename << ".\n";
    cout << "while reading @Vreal, expecting:\n";
    cout << "  nCharge XX\n";
    return false;
  }
  ifs >> nCharge;

  V.resize(nCharge*4);
  if (!NPME_SkipInput2NextLine (ifs, filename, printError)) return false;


  size_t count = 0;
  for (size_t n = 0; n < nCharge;    n++)
  for (size_t p = 0; p < 4;          p++)
  {
    ifs >> V[n*4 + p];
    count++;

    if (count%4 == 0)
    {
      if (!NPME_SkipInput2NextLine (ifs, filename, printError)) return false;
    }
  }

  return true;
}

bool NPME_ReadSingleBox_V (const std::string& filename,
  size_t& nCharge, std::vector<_Complex double>& V)
{
  using std::cout;
  using std::ifstream;
  using std::string;
  using std::vector;

  bool printError = 1;
  string sTmp;
  char str[2000];

  ifstream ifs(filename.c_str());
  if (ifs.fail())
  {
    cout << "Error opening " << filename << "\n";
    return false;
  }

  if (!NPME_GoToLine (ifs, "@Vcomplex", filename, printError)) return false;

  ifs >> sTmp;
  if (sTmp != "nCharge")
  {
    cout << "Error reading " << filename << ".\n";
    cout << "while reading @Vcomplex, expecting:\n";
    cout << "  nCharge XX\n";
    return false;
  }
  ifs >> nCharge;


  V.resize(nCharge*4);
  if (!NPME_SkipInput2NextLine (ifs, filename, printError)) return false;

  double *Vtmp = (double *) &V[0];
  size_t count = 0;
  for (size_t n = 0; n < nCharge;    n++)
  for (size_t p = 0; p < 8;          p++)
  {
    ifs >> Vtmp[n*8 + p];
    count++;

    if (count%4 == 0)
    {
      if (!NPME_SkipInput2NextLine (ifs, filename, printError)) return false;
    }
  }

  return true;
}


bool NPME_ReadSingleBox_V (const std::string& filename,
  bool& isChargeReal, size_t& nCharge, 
  std::vector<double>& Vreal, std::vector<_Complex double>& Vcomplex)
{
  using std::string;
  using std::cout;
  using std::vector;
  using std::ifstream;

  bool printError = 1;
  ifstream ifs(filename.c_str());
  if (ifs.fail())
  {
    cout << "Error opening " << filename << "\n";
    return false;
  }

  string sTmp;
  string strLine;
  

  bool foundV = 0;
  while (!ifs.eof())
  {
    getline (ifs, strLine);
    if (strLine[0] == '@')
    {
      std::stringstream sof(strLine);
      sof >> sTmp;
      if (sTmp == "@Vreal")
      {
        foundV       = 1;
        isChargeReal = 1;
        break;
      }
      else if (sTmp == "@Vcomplex")
      {
        foundV       = 1;
        isChargeReal = 0;
        break;
      }
    }
  }
  if (!foundV)
  {
    cout << "Error reading " << filename << ".\n";
    cout << "  did not find @Vreal or @Vcomplex flags\n";
    return false;
  }

  if (isChargeReal)
    return NPME_ReadSingleBox_V (filename, nCharge, Vreal);
  else
    return NPME_ReadSingleBox_V (filename, nCharge, Vcomplex);
}






}//end namespace NPME_Library



