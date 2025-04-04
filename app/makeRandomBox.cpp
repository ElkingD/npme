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
#include <cstdio>
#include <cstring>

#include <iostream> 
#include <fstream> 
#include <sstream>
#include <vector>
#include <string>


#include "NPME_Constant.h"
#include "NPME_ReadPrint.h"
#include "NPME_SupportFunctions.h"
#include "NPME_ExtLibrary.h"
using namespace NPME_Library;




int main (int argc, char *argv[])
{
  char str[2000];
  using std::cout;
  using std::string;
  using std::vector;
  using std::ofstream;

  bool PRINT_ALL = 0;   //verbose output in .log file

  if (argc != 2)
  {
    cout << "Error.\n";
    cout << argv[0] << " keywordFile\n";
    exit(0);
  }

  //1) read keywords and keep a copy of initial keywords
  string keywordFile(argv[1]);
  NPME_KeywordInput keyword0(keywordFile.c_str());
  NPME_KeywordInput keyword = keyword0;

  //2) number of charges (required)
  size_t nCharge;
  if (!keyword.GetValue ("nCharge", nCharge))
  {
    cout << "Error reading " << keywordFile << " . nCharge is not defined\n";
    exit(0);
  }


  //4) chargeType = 'real' or 'complex'  (default = 'real')
  bool isChargeReal;
  string chargeType;
  {
    if (!keyword.GetValue ("chargeType", chargeType))
    {
      chargeType = "real";
      keyword.AddKeyword ("chargeType", chargeType);
    }

    if      (chargeType == "real")    isChargeReal = 1;
    else if (chargeType == "complex") isChargeReal = 0;
    else
    {
      sprintf(str, "Error reading %s.  chargeType = %s is undefined.\n",
        keywordFile.c_str(), chargeType.c_str());
      cout << str;
      cout << "  chargeType must be 'real' or 'complex'\n";
      exit(0);
    }
  }
      
  //5) output coordinate filenames
  string crdFile;
  if (!keyword.GetValue ("crdFile", crdFile))
  {
    sprintf(str, "coord_%lu_%s.txt", nCharge, chargeType.c_str());
    crdFile = str;
    keyword.AddKeyword ("crdFile", crdFile);
  }

  //6) log file
  ofstream ofs_log;
  bool printLog;
  if (!keyword.GetValue ("printLog", printLog))
  {
    printLog = 1;
    keyword.AddKeyword ("printLog", printLog);
  }
  
  if (printLog)
  {
    string initial;
    string logFile;
    NPME_RemoveExtension (crdFile, initial);
    logFile = initial + ".log";

    cout << "printing " << logFile << "\n";
    ofs_log.open(logFile.c_str());
  }

  //7) box dimensions and charge scale
  double lx, ly, lz, chargeScale;
  if (!keyword.GetValue ("lx", lx))
  {
    lx = 10.0;
    keyword.AddKeyword ("lx", lx);
  }
  if (!keyword.GetValue ("ly", ly))
  {
    ly = 10.0;
    keyword.AddKeyword ("ly", ly);
  }
  if (!keyword.GetValue ("lz", lz))
  {
    lz = 10.0;
    keyword.AddKeyword ("lz", lz);
  }
  if (!keyword.GetValue ("chargeScale", chargeScale))
  {
    chargeScale = 1.0;
    keyword.AddKeyword ("chargeScale", chargeScale);
  }

  if ( (lx <= 0.0) || (ly <= 0.0) || (lz <= 0.0) )
  {
    sprintf(str, "Error.  lx ly lz = %f %f %f must be all be positive\n", 
      lx, ly, lz);
    cout << str;
    exit(0);
  }

  //8) random number seed
  size_t seed;
  if (!keyword.GetValue ("seed", seed))
  {
    seed = 1;
    keyword.AddKeyword ("seed", seed);
  }


  //9) print log file with initial keywords and initial+default keywords
  if (printLog)
  {
    ofs_log << "Initial Keywords:\n";
    keyword0.PrintKeywords (ofs_log);

    ofs_log << "\n\nKeywords with defaults added:\n";
    keyword.PrintKeywords (ofs_log);
  }



  //10) Generate random coordinates
  const double Rc[3] = {0,  0,  0};
  vector<double> coord(3*nCharge);
  {
    NPME_GenerateRandomCoord (nCharge, &coord[0], lx, ly, lz, Rc, seed);
  }

  //11) Generate random charges
  vector<double> chargeReal;
  vector<_Complex double> chargeComplex;
  if (chargeType == "real")
  {
    chargeReal.resize(nCharge);
    NPME_RandomNumberArray (nCharge, 
      &chargeReal[0], -chargeScale, chargeScale, seed+1);
  }
  else if (chargeType == "complex")
  {
    chargeComplex.resize(nCharge);
    NPME_RandomNumberArray (2*nCharge, 
      (double *) &chargeComplex[0], -chargeScale, chargeScale, seed+1);
  }
  else
  {
    sprintf(str, "Error.  chargeType = '%s' must be 'real' or 'complex\n", 
      chargeType.c_str());
    cout << str;
    exit(0);
  }

  //12) Print output
  ofstream ofs_out(crdFile.c_str());
  cout << "printing " << crdFile << "\n";
  NPME_PrintSingleBoxChargeCoord (ofs_out, nCharge, 
    isChargeReal, coord, chargeReal, chargeComplex);



  return 1;
}



