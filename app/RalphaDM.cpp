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


//This simpler application illustrates how NPME with the R^alpha kernel and
//DM Ewald splitting can be used.

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstddef>

#include <iostream> 
#include <fstream> 
#include <sstream>
#include <vector>
#include <string>





#include "NPME_Constant.h"
#include "NPME_ReadPrint.h"
#include "NPME_Interface.h"
#include "NPME_SupportFunctions.h"
using namespace NPME_Library;


int main (int argc, char *argv[])
{
  bool printHighPrecisionV = 0;
  using std::cout;
  using std::vector;
  using std::string;

  if (argc != 3)
  {
    cout << "Error.\n";
    cout << argv[0] << " coordChargeFile keywordFile\n";
    exit(0);
  }

  //1) get coordCharge, keyword, log, and output filenames
  string coordFile  (argv[1]);
  string keywordFile(argv[2]);
  string logFile;
  string V_pme_File;
  string V_ref_File;
  NPME_SetFilenames (logFile, V_pme_File, V_ref_File, coordFile, keywordFile);

  bool printLog = 1;
  std::ofstream ofs_log(logFile);
  cout << "printing " << logFile << "\n";

  //2) read coord, charges
  size_t nCharge;
  vector<double> coord;
  vector<double> charge;
  if (!NPME_ReadSingleBoxChargeCoordReal (coordFile.c_str(),
    nCharge, coord, charge))
  {
    cout << "Error reading " << coordFile << "\n";
    return 0;
  }

  //3) read keywords and override "funcType" and "EwaldSplit"
  NPME_KeywordInput keyword(keywordFile.c_str());
  keyword.AddKeyword ("funcType",   "Ralpha");
  keyword.AddKeyword ("EwaldSplit", "DerivMatch");

  //4) add default missing keywords and FFT sizes if not already present
  if (!NPME_AddMissingKeywords (keywordFile.c_str(), keyword, nCharge, 
            coord, charge, printLog, ofs_log))
  {
    cout << "Error in NPME_AddMissingKeywords\n";
    return 0;
  }

  string calcType;
  bool printV;
  if (!keyword.GetValue ("calcType", calcType))
  {
    cout << "Error.  'calcType' is missing from keyword\n";
    return 0;
  } 
  if (!keyword.GetValue ("printV", printV))
  {
    cout << "Error.  'printV' is missing from keyword\n";
    return 0;
  } 

  //5) Define Kernel functions, exact, SR = short range, and LR = long range
  double alpha;
  double Rdir;
  int Nder;
  if (!keyword.GetValue ("alpha", alpha))
  {
    cout << "Error.  'alpha' is missing from keyword\n";
    return 0;
  } 
  if (!keyword.GetValue ("Rdir", Rdir))
  {
    cout << "Error.  'Rdir' is missing from keyword\n";
    return 0;
  } 

  if (!keyword.GetValue ("nDeriv", Nder))
  {
    cout << "Error.  'Nder' is missing from keyword\n";
    return 0;
  } 

  NPME_Kfunc_Ralpha       func   (alpha);
  NPME_Kfunc_Ralpha_SR_DM funcSR (alpha, Nder, Rdir);
  NPME_Kfunc_Ralpha_LR_DM funcLR (alpha, Nder, Rdir);

  //5) SetUp main interface.  uses generic functions
  NPME_InterfaceReal_GenFunc npme;
  if (!npme.SetUp (keyword, nCharge, &coord[0], &func, &funcLR, &funcSR, 
      printLog, ofs_log))
  {
    cout << "Error.  npme.SetUp(..) failed\n";
    return 0;
  }

  if (printLog)
    npme.Print (ofs_log);

  //6) pme calc = main potential + potential gradient calculation
  vector<double> Vpme(nCharge*4);
  //Vpme[] = {V[0], dVdx[0], dVdx[0], dVdx[0],
  //          V[1], dVdx[1], dVdx[1], dVdx[1],
  //          V[2], dVdx[2], dVdx[2], dVdx[2],..
  if ( (calcType == "pme") || (calcType == "pme_exact") )
  {
    if (!npme.CalcV (&Vpme[0], nCharge, &charge[0], &coord[0], 
                    printLog, ofs_log))
    {
      printf("Error.  npme->CalcV failed\n");
      return 0;
    }
  }

  //7) do the exact ~N^2 calculation if asked
  vector<double> Vref;
  if ( (calcType == "exact") || (calcType == "pme_exact") )
  {
    Vref.resize(nCharge*4);
    if (!npme.CalcV_exact (&Vref[0], nCharge, &charge[0], &coord[0], 
                    printLog, ofs_log))
    {
      printf("Error.  npme->CalcV_exact failed\n");
      return 0;
    }
  }


  //8) calculate errors for pme_exact
  if (calcType == "pme_exact")
  {
    double errorV, errordVdr;
    NPME_CompareV1Arrays (errorV, errordVdr, 
      nCharge, 
      "pme", &Vpme[0], 
      "ref", &Vref[0], 0, ofs_log);
    
    //average magnitude of V and dVdr
    double V_mag, dVdr_mag;
    NPME_CalcV1AvgVecMag (V_mag, dVdr_mag, nCharge, &Vref[0]);

    if (printLog)
    {
      char str[2000];
      sprintf(str, "\n\nerrorV = %.2le errordVdr = %.2le (Absolute Error)\n", 
        errorV, errordVdr);
      ofs_log << str;
      sprintf(str, "errorV = %.2le errordVdr = %.2le (Relative Error)\n", 
        errorV/V_mag, errordVdr/dVdr_mag);
      ofs_log << str;
      ofs_log.flush();
    }
  }

  //9) print V to a file
  if (printV)
  {
    if ( (calcType == "pme") || (calcType == "pme_exact") )
    {
      cout << "printing " << V_pme_File << "\n";
      NPME_PrintSingleBox_V (V_pme_File, nCharge, Vpme, printHighPrecisionV);
    }
    if ( (calcType == "exact") || (calcType == "pme_exact") )
    {
      cout << "printing " << V_ref_File << "\n";
      NPME_PrintSingleBox_V (V_ref_File, nCharge, Vref, printHighPrecisionV);
    }
  }


  return 1;
}



