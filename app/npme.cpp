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
#include <cstddef>

#include <iostream> 
#include <fstream> 
#include <sstream>
#include <vector>
#include <string>



#include "NPME_Constant.h"
#include "NPME_ReadPrint.h"
#include "NPME_KernelFunction.h"
#include "NPME_KernelFunctionLaplace.h"
#include "NPME_KernelFunctionRalpha.h" 
#include "NPME_KernelFunctionHelmholtz.h"
#include "NPME_Interface.h"

#include "NPME_SupportFunctions.h"
using namespace NPME_Library;


int main (int argc, char *argv[])
{
  double timeStart = NPME_GetTime ();
  double time0, timeReadCoord, timeReadInput, timeProcessInput, timeSetUp;

  bool useGenericKernel    = 0;
  bool printHighPrecisionV = 0;


  bool printLev1  = 1;    //print minimum CPU time
  bool printLev2  = 1;    //print CPU times and SetUp info
  bool printLev3  = 0;    //print everything (only use on small systems)

  using std::cout;
  using std::string;
  using std::vector;


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


  //2) read coord, charges
  size_t nCharge;
  bool isChargeReal;
  vector<double> coord;
  vector<double> chargeReal;
  vector<_Complex double> chargeComplex;

  time0 = NPME_GetTime ();
  if (!NPME_ReadSingleBoxChargeCoord (coordFile.c_str(),
    nCharge, isChargeReal, coord,
    chargeReal, chargeComplex))
  {
    cout << "Error reading " << coordFile << "\n";
    return 0;
  }
  timeReadCoord = NPME_GetTime () - time0;


  //3) read keywords
  time0 = NPME_GetTime ();
  NPME_KeywordInput keyword(keywordFile.c_str());
  timeReadInput = NPME_GetTime () - time0;

  //4) print log file
  bool printLog = 1;
  std::ofstream ofs_log;
  keyword.GetValue ("printLog", printLog);


  if (!printLog)
  {
    printLev1  = 0;    //print CPU times
    printLev2  = 0;    //print CPU times and SetUp info
    printLev3  = 0;    //print everything (only use on small systems)
  }

  if (printLog)
  {
    cout << "printing " << logFile << "\n";
    ofs_log.open(logFile.c_str());
    if (printLev2)
      keyword.PrintKeywords (ofs_log);

    if (printLev3)
      NPME_PrintSingleBoxChargeCoord (ofs_log, 
          nCharge, isChargeReal, coord, chargeReal, chargeComplex);
    else if (printLev2)
      NPME_PrintBoxInfo (ofs_log, nCharge, isChargeReal);

  }

  //5) add default missing keywords
  time0 = NPME_GetTime ();
  if (!NPME_AddMissingKeywords (keywordFile.c_str(), keyword, nCharge, 
            isChargeReal, 
            coord, chargeReal, chargeComplex, printLev2, ofs_log))
  {
    cout << "Error in NPME_AddMissingKeywords\n";
    return 0;
  }
  timeProcessInput = NPME_GetTime () - time0;

  if (printLev1)
  {
    char str[2000];
    sprintf(str, "  time = %6.2le (read charge+coord)\n", timeReadCoord);
    ofs_log << str;
    sprintf(str, "  time = %6.2le (read input)\n", timeReadInput);
    ofs_log << str;
    sprintf(str, "  time = %6.2le (process input)\n", timeProcessInput);
    ofs_log << str;
  }



  string calcType;
  if (!keyword.GetValue ("calcType", calcType))
  {
    cout << "Error.  'calcType' is missing from keyword\n";
    return 0;
  } 

  bool printV;
  if (!keyword.GetValue ("printV", printV))
  {
    cout << "Error.  'printV' is missing from keyword\n";
    return 0;
  } 


  NPME_KernelList kernelList;
  //container for all pre-defined kernel function (classes)

  if (isChargeReal)
  {
    time0 = NPME_GetTime ();
    //func = func_LR + func_SR
    NPME_KfuncReal *func    = NULL;    //exact kernel
    NPME_KfuncReal *func_LR = NULL;    //smooth     long  range kernel
    NPME_KfuncReal *func_SR = NULL;    //non-smooth short range kernel

    //6) select kernel pointers
    if (!NPME_Interface_SelectKernelPtr (func, func_LR, func_SR,
      kernelList, keyword, 0, ofs_log))
    //a) using 'keyword', selects correct kernels from 'kernelList'
    //b) set correct kernels with appropriate parameters from 'keyword'
    //c) set kernel function pointers ('func', 'func_LR', 'func_SR')
    //   to the correct kernel functions contained in kernelList
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      return 0;
    }

    //7) set up npme interface
    NPME_InterfaceReal *npme = NULL;
    if (!NPME_Interface_SetUpRealKernel (npme, 
      func, func_LR, func_SR,
      nCharge, &coord[0], keyword, useGenericKernel,
      printLev2, ofs_log))
    //a) allocates and sets up the appropriate derived npme interface class
    //b) sets 'NPME_InterfaceReal' pointer to the the derived npme interface 
    //   class
    {
      cout << "Error in NPME_Interface_SetUp\n";
      return 0;
    }
    timeSetUp = NPME_GetTime () - time0;

    if (printLev1)
    {
      char str[2000];
      sprintf(str, "  time = %6.2le (setup)\n", timeSetUp);
      ofs_log << str;
    }

    //8) pme calc
    vector<double> Vpme;
    vector<double> Vref;
    if ( (calcType == "pme") || (calcType == "pme_exact") )
    {
      Vpme.resize(nCharge*4);
      memset(&Vpme[0], 0, nCharge*4*sizeof(double)); //not necessary

    //npme->ResetCoord (nCharge, &coord[0], printLev1, ofs_log);

      if (!npme->CalcV (&Vpme[0], nCharge, 
                       &chargeReal[0], &coord[0], printLev1, ofs_log))
      {
        printf("Error.  npme->CalcV failed\n");
        return 0;
      }

      double V_mag, dVdr_mag;
      NPME_CalcV1AvgVecMag (V_mag, dVdr_mag, nCharge, &Vpme[0]);

      if (printLev1)
      {
        char str[2000];
        sprintf(str, "V_mag  = %.2le dVdr_mag  = %.2le (Average magnitude)\n", 
          V_mag, dVdr_mag);
        ofs_log << str;
        ofs_log.flush();
      }
    }

    if ( (calcType == "exact") || (calcType == "pme_exact") )
    {
      Vref.resize(nCharge*4);
      memset(&Vref[0], 0, nCharge*4*sizeof(double)); //not necessary



      double timeVstart = NPME_GetTime ();
      if (!npme->CalcV_exact (&Vref[0], nCharge, 
                       &chargeReal[0], &coord[0], 0, ofs_log))
      {
        printf("Error.  npme->CalcV_exact failed\n");
        return 0;
      }
      double timeV = NPME_GetTime () - timeVstart;
      {
        char str[2000];
        sprintf(str, " time_exact = %.2le\n", timeV);
        ofs_log << str;
        ofs_log.flush();
      }
    }

    if (calcType == "pme_exact")
    {
      double errorV, errordVdr;
      NPME_CompareV1Arrays (errorV, errordVdr, 
        nCharge, 
        "pme", &Vpme[0], 
        "ref", &Vref[0], printLev3, ofs_log);
      
      double V_mag, dVdr_mag;
      NPME_CalcV1AvgVecMag (V_mag, dVdr_mag, nCharge, &Vref[0]);

      if (printLev1)
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
  }
  else
  {
    time0 = NPME_GetTime ();

    //func = func_LR + func_SR
    NPME_KfuncComplex *func    = NULL;    //exact kernel
    NPME_KfuncComplex *func_LR = NULL;    //smooth     long  range kernel
    NPME_KfuncComplex *func_SR = NULL;    //non-smooth short range kernel
                                              
    //6) select kernel pointers
    if (!NPME_Interface_SelectKernelPtr (func, func_LR, func_SR,
      kernelList, keyword, printLev2, ofs_log))
    //a) using 'keyword', selects correct kernels from 'kernelList'
    //b) set correct kernels with appropriate parameters from 'keyword'
    //c) set kernel function pointers ('func', 'func_LR', 'func_SR')
    //   to the correct kernel functions contained in kernelList
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      return 0;
    }

    //7) set up npme interface
    NPME_InterfaceComplex *npme = NULL;
    if (!NPME_Interface_SetUpComplexKernel (npme, 
      func, func_LR, func_SR,
      nCharge, &coord[0], keyword, useGenericKernel,
      printLev2, ofs_log))
    //a) allocates and sets up the appropriate derived npme interface class
    //b) sets 'NPME_InterfaceComplex' pointer to the the derived npme interface 
    //   class
    {
      cout << "Error in NPME_Interface_SetUp\n";
      return 0;
    }
    timeSetUp = NPME_GetTime () - time0;


    if (printLev1)
    {
      char str[2000];
      sprintf(str, "  time = %6.2le (setup)\n", timeSetUp);
      ofs_log << str;
    }

    vector<_Complex double> Vpme;
    vector<_Complex double> Vref;
    if ( (calcType == "pme") || (calcType == "pme_exact") )
    {
      Vpme.resize(nCharge*4);
      memset(&Vpme[0], 0, nCharge*4*sizeof(_Complex double));

    //npme->ResetCoord (nCharge, &coord[0], printLev1, ofs_log);
      if (!npme->CalcV (&Vpme[0], nCharge, 
                       &chargeComplex[0], &coord[0], printLev1, ofs_log))
      {
        printf("Error.  npme->CalcV failed\n");
        return 0;
      }

      double V_mag, dVdr_mag;
      NPME_CalcV1AvgVecMag (V_mag, dVdr_mag, nCharge, &Vpme[0]);
      if (printLev1)
      {
        char str[2000];
        sprintf(str, "V_mag  = %.2le dVdr_mag  = %.2le (Average magnitude)\n", 
          V_mag, dVdr_mag);
        ofs_log << str;
        ofs_log.flush();
      }
    }



    if ( (calcType == "exact") || (calcType == "pme_exact") )
    {
      Vref.resize(nCharge*4);
      memset(&Vref[0], 0, nCharge*4*sizeof(_Complex double));

      double timeVstart = NPME_GetTime ();
      if (!npme->CalcV_exact (&Vref[0], nCharge, 
                       &chargeComplex[0], &coord[0], 0, ofs_log))
      {
        printf("Error.  npme->CalcV_exact failed\n");
        return 0;
      }

      double timeV = NPME_GetTime () - timeVstart;
      {
        char str[2000];
        sprintf(str, " time_exact = %.2le\n", timeV);
        ofs_log << str;
        ofs_log.flush();
      }

    }

    if (calcType == "pme_exact")
    {
      double errorV, errordVdr;
      NPME_CompareV1Arrays (errorV, errordVdr, 
        nCharge, 
        "pme", &Vpme[0], 
        "ref", &Vref[0], printLev3, ofs_log);
      
      double V_mag, dVdr_mag;
      NPME_CalcV1AvgVecMag (V_mag, dVdr_mag, nCharge, &Vref[0]);

      if (printLev1)
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
  }





  double timeTot = NPME_GetTime () - timeStart;

  if (printLev1)
  {
    char str[2000];
    sprintf(str, "\n\nTotal_Run_Time = %.2f\n", timeTot);
    ofs_log << str;
    ofs_log.flush();
  }



  return 1;
}



