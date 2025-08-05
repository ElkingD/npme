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
#include <cstddef>

#include <iostream> 
#include <fstream> 
#include <sstream>
#include <vector>
#include <string>
#include <map>


#include "Constant.h"
#include "ReadPrint.h"
#include "Interface.h"
#include "RecSumInterface.h"


#include "SupportFunctions.h"
#include "EstimateParm.h"
#include "PartitionBox.h"
#include "PartitionEmbeddedBox.h"
#include "PermuteArray.h"
#include "RecSumSupportFunctions.h"

#include "FunctionDerivMatch.h"
#include "KernelFunctionLaplace.h"
#include "KernelFunctionRalpha.h"
#include "KernelFunctionHelmholtz.h"

#include "PotentialGenFunc.h"
#include "PotentialLaplace.h"
#include "PotentialHelmholtz.h"


namespace NPME_Library
{
bool NPME_AddMissingKeywords (const char *keywordFile, 
  NPME_KeywordInput& keyword, 
  const size_t nCharge, const bool isChargeReal, 
  const std::vector<double>& coord, const std::vector<double>& chargeReal, 
  const std::vector<_Complex double>& chargeComplex, 
  bool printLog, std::ostream& ofs_log)
{
  bool PRINT_ALL = 0;

  if (!printLog)
    PRINT_ALL = 0;

  using std::string;
  using std::cout;

  char str[2000];
  string sTmp;
  string funcType;
  string calcType;
  string ewaldType;
  double Rdir;
  double k0_r, k0_i;
  double tol;
  long int BnOrder;
  size_t nCellClust1D;

  //simple default values
  if (!keyword.GetValue ("nProc", sTmp))
    keyword.AddKeyword ("nProc", omp_get_max_threads());

  if (!keyword.GetValue ("FFTmemGB", sTmp))
    keyword.AddKeyword ("FFTmemGB", NPME_Default_FFTmemGB);

  if (!keyword.GetValue ("vecOption", sTmp))
    keyword.AddKeyword ("vecOption", "avx");

  if (!keyword.GetValue ("BnOrder", BnOrder))
  {
    BnOrder = NPME_Default_BnOrder;
    keyword.AddKeyword ("BnOrder", BnOrder);
  }

  if (!keyword.GetValue ("a1", sTmp)) keyword.AddKeyword ("a1", 4.0);
  if (!keyword.GetValue ("a2", sTmp)) keyword.AddKeyword ("a2", 4.0);
  if (!keyword.GetValue ("a3", sTmp)) keyword.AddKeyword ("a3", 4.0);

  if (!keyword.GetValue ("tol", tol))
    keyword.AddKeyword ("tol", NPME_Default_tol);


  if (!keyword.GetValue ("nCellClust1D", nCellClust1D))
    keyword.AddKeyword ("nCellClust1D", NPME_Default_nCellClust1D);


  if (keyword.GetValue ("calcType", calcType))
  {
    if ( (calcType != "pme")    && 
         (calcType != "exact")  && 
         (calcType != "pme_exact") )
    {
      sprintf(str, "Error reading %s\n", keywordFile);
      cout << str;
      sprintf(str, "  'calcType' = %s is undefined\n", calcType.c_str());
      cout << str;
      cout << "options are: 'pme', 'exact', or 'pme_exact'\n";
      cout << "  pme        = perform pme calculation only\n";
      cout << "  exact      = perform exact (~N^2) calculation only\n";
      cout << "  pme_exact  = perform both pme and exact calculations\n";
      return false;
    }
  }
  else
  {
    cout << "Missing required keyword 'calcType' from " << keywordFile << "\n";
    cout << "options are: 'pme', 'exact', or 'pme_exact'\n";
    cout << "  pme        = perform pme calculation only\n";
    cout << "  exact      = perform exact (~N^2) calculation only\n";
    cout << "  pme_exact  = perform both pme and exact calculations\n";
    return false;
  }

  if (!keyword.GetValue ("funcType", funcType))
  {
    cout << "Missing required keyword 'funcType' from " << keywordFile << "\n";
    cout << "options are: 'Laplace', 'Helmholtz', or 'Ralpha'\n";
    cout << "  Laplace    = 1/r           kernel\n";
    cout << "  Helmholtz  = exp(I*k0*r)/r kernel (k0    = complex)\n";
    cout << "  Ralpha     = r^alpha       kernel (alpha = real)\n";
    return false;
  }

  if (funcType == "Helmholtz")
  {
    if (!keyword.GetValue ("k0_r", k0_r))
    {
      cout << "Missing required keyword 'k0_r' from " << keywordFile << "\n";
      cout << "  funcType   = " << funcType  << "\n";
      return false;
    }

    if (!keyword.GetValue ("k0_i", k0_i))
    {
      cout << "Missing required keyword 'k0_i' from " << keywordFile << "\n";
      cout << "  funcType   = " << funcType  << "\n";
      return false;
    }

    if (k0_i < -1.0E-12)
    {
      cout << "Error.  k0_i = " << k0_i << " is negative\n";
      cout << "  k0_i should be positive or zero\n";
      return false;
    }


    if (isChargeReal != 0)
    {
      cout << "Error reading " << keywordFile << "\n";
      cout << "  input charges are real, but kernel is complex\n";
      cout << "  funcType   = " << funcType  << "\n";
      return false;
    }
  }
  else if (funcType == "Ralpha")
  {
    if (!keyword.GetValue ("alpha", sTmp))
    {
      cout << "Missing required keyword 'alpha' from " << keywordFile << "\n";
      cout << "  funcType   = " << funcType  << "\n";
      return false;
    }

    if (isChargeReal != 1)
    {
      cout << "Error reading " << keywordFile << "\n";
      cout << "  input charges are complex, but kernel is real\n";
      cout << "  funcType   = " << funcType  << "\n";
      return false;
    }
  }
  else if (funcType == "Laplace")
  {
    if (isChargeReal != 1)
    {
      cout << "Error reading " << keywordFile << "\n";
      cout << "  input charges are complex, but kernel is real\n";
      cout << "  funcType   = " << funcType  << "\n";
      return false;
    }
  }
  else 
  {
    cout << "Error reading " << keywordFile << "\n";
    cout << "  funcType   = " << funcType  << " is undefined\n";
    return false;
  }
    


  if (!keyword.GetValue ("Rdir", Rdir))
  {
    cout << "Missing required keyword 'Rdir' from " << keywordFile << "\n";
    cout << "  Rdir = direct space cutoff\n";
    return false;
  }

  if (!keyword.GetValue ("EwaldSplit", ewaldType))
    keyword.AddKeyword ("EwaldSplit", "DerivMatch");
  else
  {
    //only 2 possibilities for 'EwaldSplit' are 'DerivMatch' and 'LaplaceOrig'
    if (ewaldType == "LaplaceOrig")
    {
      if (funcType != "Laplace")
      {
        cout << "Keywords for EwaldSplit and funcType are incompatible\n";
        cout << "  EwaldSplit = " << ewaldType << "\n";
        cout << "  funcType   = " << funcType  << "\n";
        return false;
      }
    }
    else if (ewaldType != "DerivMatch")
    {
      cout << "Keywords for EwaldSplit are 'DerivMatch' or 'LaplaceOrig'\n";
      cout << "  EwaldSplit = " << ewaldType << " is undefined\n";
      return false;
    }
  }


  if (!keyword.GetValue ("del1", sTmp))   keyword.AddKeyword ("del1", Rdir);
  if (!keyword.GetValue ("del2", sTmp))   keyword.AddKeyword ("del2", Rdir);
  if (!keyword.GetValue ("del3", sTmp))   keyword.AddKeyword ("del3", Rdir);

  if (!keyword.GetValue ("printV",   sTmp)) keyword.AddKeyword ("printV",   1);
  if (!keyword.GetValue ("printLog", sTmp)) keyword.AddKeyword ("printLog", 1);


  if (!keyword.GetValue ("nNeigh", sTmp))
    keyword.AddKeyword ( "nNeigh", NPME_Default_nNeigh);

  if (!keyword.GetValue ("useCompactF", sTmp))
    keyword.AddKeyword ( "useCompactF", 1);

  //set FFT sizes and block sizes if any components are missing
  bool setFFT = 0;
  if (!keyword.GetValue ("N1", sTmp))   setFFT = 1;
  if (!keyword.GetValue ("N2", sTmp))   setFFT = 1;
  if (!keyword.GetValue ("N3", sTmp))   setFFT = 1;
  if (!keyword.GetValue ("n1", sTmp))   setFFT = 1;
  if (!keyword.GetValue ("n2", sTmp))   setFFT = 1;
  if (!keyword.GetValue ("n3", sTmp))   setFFT = 1;





  if (setFFT)
  {
    long int N1, N2, N3;
    long int n1, n2, n3;

    double tol;
    if (!keyword.GetValue ("tol", tol))
    {
      cout << "Error in NPME_AddMissingKeywords\n";
      cout << "  tol is not already set\n";
      return false;
    }


    if ( (funcType == "Ralpha") || (funcType == "Laplace") )
    {
      double alpha;
      if (funcType == "Laplace")
        alpha = -1.0;
      else
      {
        if (!keyword.GetValue ("alpha", alpha))
        {
          cout << "Error in NPME_AddMissingKeywords\n";
          cout << "  alpha is not already set\n";
          return false;
        }
      }

      if (chargeReal.size() != nCharge)
      {
        cout << "Error in NPME_AddMissingKeywords\n";
        sprintf(str, "  chargeReal.size() = %lu != %lu = nCharge\n",
          chargeReal.size(), nCharge);
        return false;
      }

      NPME_EstimateParmSingleBox_StaticModel parm;

      int nDeriv;
      if (!keyword.GetValue ("nDeriv", nDeriv))
      {
        //nDeriv is not given, choose optimal
        if (!parm.Get_FFT_outputNder (
                  N1, N2, N3,
                  n1, n2, n3,
                  nDeriv, nCharge, &chargeReal[0], &coord[0],
                  BnOrder, alpha, tol, Rdir, PRINT_ALL, ofs_log))
        {
          cout << "Error in NPME_AddMissingKeywords\n";
          cout << "  parm.Get_FFT failed\n";
          return false;
        }
        keyword.AddKeyword ("nDeriv", nDeriv);
      }
      else
      {
        //nDeriv is given
        if (!parm.Get_FFT_inputNder (
                  N1, N2, N3,
                  n1, n2, n3,
                  nCharge, &chargeReal[0], &coord[0],
                  BnOrder, alpha, nDeriv, tol, Rdir, PRINT_ALL, ofs_log))
        {
          cout << "Error in NPME_AddMissingKeywords\n";
          cout << "  parm.Get_FFT failed\n";
          return false;
        }
      }


      double eps_V;
      bool reliablePred;
      if (!parm.CalcPredError_V (eps_V, reliablePred,
          N1, N2, N3, 
          nCharge, &chargeReal[0], &coord[0],
          BnOrder, alpha, nDeriv, Rdir, PRINT_ALL, ofs_log))
      {
        cout << "Error in NPME_AddMissingKeywords\n";
        cout << "  parm.CalcPredError_V failed\n";
        return false;
      }

      if (printLog)
      {
        sprintf(str, "eps_V_pred = %.2le  (%d)\n", eps_V, (int) reliablePred);
        ofs_log << str;
        ofs_log.flush();
      }
    }
    else if (funcType == "Helmholtz")
    {
      if (chargeComplex.size() != nCharge)
      {
        cout << "Error in NPME_AddMissingKeywords\n";
        sprintf(str, "  chargeComplex.size() = %lu != %lu = nCharge\n",
          chargeComplex.size(), nCharge);
        return false;
      }

      NPME_EstimateParmSingleBox_HelmholtzModel parm;

      int nDeriv;
      if (!keyword.GetValue ("nDeriv", nDeriv))
      {
        //nDeriv is not given, choose optimal
        if (!parm.Get_FFT_outputNder (
                  N1, N2, N3,
                  n1, n2, n3,
                  nDeriv, nCharge, &chargeComplex[0], &coord[0],
                  BnOrder, k0_r + I*k0_i, tol, Rdir, PRINT_ALL, ofs_log))
        {
          cout << "Error in NPME_AddMissingKeywords\n";
          cout << "  parm.Get_FFT failed\n";
          return false;
        }
        keyword.AddKeyword ("nDeriv", nDeriv);
      }
      else
      {
        //nDeriv is given
        if (!parm.Get_FFT_inputNder (
                  N1, N2, N3,
                  n1, n2, n3,
                  nCharge, &chargeComplex[0], &coord[0],
                  BnOrder, k0_r + I*k0_i, nDeriv, tol, Rdir, 
                  1, ofs_log))
        {
          cout << "Error in NPME_AddMissingKeywords\n";
          cout << "  parm.Get_FFT failed\n";
          return false;
        }
      }


      double eps_V;
      bool reliablePred;
      if (!parm.CalcPredError_V (eps_V, reliablePred,
          N1, N2, N3, 
          nCharge, &chargeComplex[0], &coord[0],
          BnOrder, k0_r + I*k0_i, nDeriv, Rdir, PRINT_ALL, ofs_log))
      {
        cout << "Error in NPME_AddMissingKeywords\n";
        cout << "  parm.CalcPredError_V failed\n";
        return false;
      }

      if (printLog)
      {
        sprintf(str, "eps_V_pred = %.2le  (%d)\n", eps_V, (int) reliablePred);
        ofs_log << str;
        ofs_log.flush();
      }
    }
    else
    {
      cout << "Error in NPME_AddMissingKeywords\n";
      cout << "  funcType   = " << funcType  << " is undefined\n";
      return false;
    }

    keyword.AddKeyword ("N1", N1);
    keyword.AddKeyword ("N2", N2);
    keyword.AddKeyword ("N3", N3);
    keyword.AddKeyword ("n1", n1);
    keyword.AddKeyword ("n2", n2);
    keyword.AddKeyword ("n3", n3);
  }


  if (ewaldType == "DerivMatch")
  {
    int nDeriv;
    if (!keyword.GetValue ("nDeriv", nDeriv))
    {
      if (setFFT)
      {
        cout << "Error in NPME_AddMissingKeywords\n";
        cout << "  nDeriv is missing, but setFFT = 1\n";
        return false;
      }

      if      (BnOrder ==  4)   nDeriv = 3;
      else if (BnOrder ==  6)   nDeriv = 6;
      else                      nDeriv = 8;
      keyword.AddKeyword ("nDeriv", nDeriv);
    }
  }

  //calculate FFT memory
  {
    long int N1, N2, N3;
    long int n1, n2, n3;
    bool setFFT = 0;

    if (!keyword.GetValue ("N1", N1))   setFFT = 1;
    if (!keyword.GetValue ("N2", N2))   setFFT = 1;
    if (!keyword.GetValue ("N3", N3))   setFFT = 1;
    if (!keyword.GetValue ("n1", n1))   setFFT = 1;

    if (setFFT)
    {
      cout << "Error in NPME_AddMissingKeywords\n";
      cout << "  missing FFT values after setting\n";
      return false;
    }

    bool useSphereSymmCompactF;
    double maxFFTmemGB;
    if (!keyword.GetValue ("FFTmemGB", maxFFTmemGB))
    {
      cout << "Error in NPME_AddMissingKeywords\n";
      cout << "  FFTmemGB is missing\n";
      return false;
    }

    if (!keyword.GetValue ("useCompactF", useSphereSymmCompactF))
    {
      cout << "Error in NPME_AddMissingKeywords\n";
      cout << "  useCompactF is missing\n";
      return false;
    }

    double FFTmemGB = NPME_CalcFFTmemGB (
                        N1, N2, N3,
                        n1, useSphereSymmCompactF);

    if (PRINT_ALL)
    {
      sprintf(str, "  FFTmemGB    = %.2f\n", FFTmemGB);
      ofs_log << str;

      sprintf(str, "  maxFFTmemGB = %.2f\n", maxFFTmemGB);
      ofs_log << str;

      ofs_log.flush();
    }      
  }

  if (printLog)
  {
    ofs_log << "\n\nAfter Adding Missing Keywords\n";
    keyword.PrintKeywords (ofs_log);
    ofs_log << "\n\n";
  }

  return true;
}

bool NPME_AddMissingKeywords (const char *keywordFile, 
  NPME_KeywordInput& keyword, const size_t nCharge, 
  const std::vector<double>& coord, const std::vector<double>& chargeReal, 
  bool printLog, std::ostream& ofs_log)
{
  bool isChargeReal = 1;
  std::vector<_Complex double> chargeComplex;

  return NPME_AddMissingKeywords (keywordFile, keyword, nCharge, 
            isChargeReal, coord, chargeReal, chargeComplex, printLog, ofs_log);
}


bool NPME_AddMissingKeywords (const char *keywordFile, 
  NPME_KeywordInput& keyword, const size_t nCharge, 
  const std::vector<double>& coord, 
  const std::vector<_Complex double>& chargeComplex, 
  bool printLog, std::ostream& ofs_log)
{
  bool isChargeReal = 0;
  std::vector<double> chargeReal;

  return NPME_AddMissingKeywords (keywordFile, keyword, nCharge, 
            isChargeReal, coord, chargeReal, chargeComplex, printLog, ofs_log);
}

bool NPME_Interface_SelectKernelPtr (
  NPME_Library::NPME_KfuncReal*& funcReal, 
  NPME_Library::NPME_KfuncReal*& funcReal_LR, 
  NPME_Library::NPME_KfuncReal*& funcReal_SR,
  NPME_Library::NPME_KernelList& kernelList, 
  const NPME_Library::NPME_KeywordInput& keyword, 
  bool printLog, std::ostream& os)
//input:  kernelList, keyword
//output: funcReal, funcReal_LR, funcReal_SR
//a) using 'keyword', selects correct kernels from 'kernelList'
//b) set correct kernels with appropriate parameters from 'keyword'
//c) set kernel function pointers ('funcReal', 'funcReal_LR', 'funcReal_SR')
//   to the correct kernel functions contained in kernelList
{
  using std::string;
  using std::cout;
  char str[2000];
  int nDeriv;
  double Rdir;
  double beta;
  double tol;

  string funcType;
  string ewaldSplit;

  funcReal    = NULL;
  funcReal_LR = NULL;
  funcReal_SR = NULL;

  if (!keyword.GetValue ("funcType", funcType))
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    cout << "  funcType is not defined in keyword\n";
    return false;
  }

  if (!keyword.GetValue ("EwaldSplit", ewaldSplit))
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    cout << "  EwaldSplit is not defined in keyword\n";
    return false;
  }

  if (ewaldSplit == "DerivMatch")
  {
    if (!keyword.GetValue ("nDeriv", nDeriv))
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      cout << "  EwaldSplit = 'DerivMatch'";
      cout << "  nDeriv is not defined in keyword\n";
      return false;
    }
    if (!keyword.GetValue ("Rdir", Rdir))
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      cout << "  EwaldSplit = 'DerivMatch'";
      cout << "  Rdir is not defined in keyword\n";
      return false;
    }
  }
  else if (ewaldSplit == "LaplaceOrig")
  {
    if (!keyword.GetValue ("Rdir", Rdir))
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      cout << "  EwaldSplit = 'LaplaceOrig'";
      cout << "  Rdir is not defined in keyword\n";
      return false;
    }
    if (!keyword.GetValue ("tol", tol))
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      cout << "  EwaldSplit = 'LaplaceOrig'";
      cout << "  tol is not defined in keyword\n";
      return false;
    }
    beta = NPME_EwaldSplitOrig_Rdir2Beta (Rdir, tol);
  }
  else
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    sprintf(str, "  EwaldSplit = %s is not defined\n", ewaldSplit.c_str());
    cout << str;
    return false;
  }

  if (funcType == "Laplace")
  {
    funcReal = &kernelList.funcLaplace;

    if (ewaldSplit == "DerivMatch")
    {
      kernelList.funcLaplace_LR_DM.SetParm (nDeriv, Rdir, printLog, os);
      kernelList.funcLaplace_SR_DM.SetParm (nDeriv, Rdir, printLog, os);

      funcReal_LR = &kernelList.funcLaplace_LR_DM;
      funcReal_SR = &kernelList.funcLaplace_SR_DM;
    }
    else if (ewaldSplit == "LaplaceOrig")
    {
      kernelList.funcLaplace_LR_Orig.SetParm (beta);
      kernelList.funcLaplace_SR_Orig.SetParm (beta);

      funcReal_LR = &kernelList.funcLaplace_LR_Orig;
      funcReal_SR = &kernelList.funcLaplace_SR_Orig;
    }
    else
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      sprintf(str, "  EwaldSplit = %s is not defined for funcType = %s\n",
        ewaldSplit.c_str(), funcType.c_str());
      cout << str;
      return false;
    }
  }
  else if (funcType == "Ralpha")
  {
    double alpha;
    if (!keyword.GetValue ("alpha", alpha))
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      sprintf(str, "  funcType = %s but 'alpha' keyword is not defined\n",
        funcType.c_str());
      cout << str;
      return false;
    }

    kernelList.funcRalpha.SetParm (alpha);
    funcReal = &kernelList.funcRalpha;

    if (ewaldSplit == "DerivMatch")
    {
      kernelList.funcRalpha_LR_DM.SetParm (alpha, nDeriv, Rdir, printLog, os);
      kernelList.funcRalpha_SR_DM.SetParm (alpha, nDeriv, Rdir, printLog, os);

      funcReal_LR = &kernelList.funcRalpha_LR_DM;
      funcReal_SR = &kernelList.funcRalpha_SR_DM;
    }
    else
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      sprintf(str, "  EwaldSplit = %s is not defined for funcType = %s\n",
        ewaldSplit.c_str(), funcType.c_str());
      cout << str;
      return false;
    }
  }
  else
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    sprintf(str, "  funcType = %s is not defined for real charges\n",
      funcType.c_str());
    cout << str;
    return false;
  }


  if (funcReal == NULL)
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    cout << "  funcReal = NULL\n";
    return false;
  }
  if (funcReal_LR == NULL)
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    cout << "  funcReal_LR = NULL\n";
    return false;
  }
  if (funcReal_SR == NULL)
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    cout << "  funcReal_SR = NULL\n";
    return false;
  }

  if (printLog)
  {
    os << "\n\nNPME_Interface_SelectKernelPtr\n";
    funcReal->Print(os);
    funcReal_LR->Print(os);
    funcReal_SR->Print(os);
  }



  return true;
}

bool NPME_Interface_SelectKernelPtr (
  NPME_Library::NPME_KfuncComplex*& funcComplex, 
  NPME_Library::NPME_KfuncComplex*& funcComplex_LR, 
  NPME_Library::NPME_KfuncComplex*& funcComplex_SR,
  NPME_Library::NPME_KernelList& kernelList, 
  const NPME_Library::NPME_KeywordInput& keyword, 
  bool printLog, std::ostream& os)
//input:  kernelList, keyword
//output: funcComplex, funcComplex_LR, funcComplex_SR
//a) using 'keyword', selects correct kernels from 'kernelList'
//b) set correct kernels with appropriate parameters from 'keyword'
//c) set kernel function pointers ('funcComplex', 'funcComplex_LR', 
//   'funcComplex_SR') to the correct kernel functions contained in kernelList
{
  using std::string;
  using std::cout;
  char str[2000];
  int nDeriv;
  double Rdir;
  double beta;
  double tol;

  string funcType;
  string ewaldSplit;

  funcComplex    = NULL;
  funcComplex_LR = NULL;
  funcComplex_SR = NULL;

  if (!keyword.GetValue ("funcType", funcType))
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    cout << "  funcType is not defined in keyword\n";
    return false;
  }

  if (!keyword.GetValue ("EwaldSplit", ewaldSplit))
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    cout << "  EwaldSplit is not defined in keyword\n";
    return false;
  }

  if (ewaldSplit == "DerivMatch")
  {
    if (!keyword.GetValue ("nDeriv", nDeriv))
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      cout << "  EwaldSplit = 'DerivMatch'";
      cout << "  nDeriv is not defined in keyword\n";
      return false;
    }
    if (!keyword.GetValue ("Rdir", Rdir))
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      cout << "  EwaldSplit = 'DerivMatch'";
      cout << "  Rdir is not defined in keyword\n";
      return false;
    }
  }
  else
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    sprintf(str, "  EwaldSplit = %s is not defined\n", ewaldSplit.c_str());
    cout << str;
    return false;
  }

  if (funcType == "Helmholtz")
  {
    double k0_r, k0_i;
    if (!keyword.GetValue ("k0_r", k0_r))
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      sprintf(str, "  funcType = %s but k0_r is not defined\n",
        funcType.c_str());
      cout << str;
      return false;
    }
    if (!keyword.GetValue ("k0_i", k0_i))
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      sprintf(str, "  funcType = %s but k0_i is not defined\n",
        funcType.c_str());
      cout << str;
      return false;
    }
    _Complex double k0 = k0_r + I*k0_i;

    kernelList.funcHelmholtz.SetParm(k0);
    funcComplex = &kernelList.funcHelmholtz;

    if (ewaldSplit == "DerivMatch")
    {
      kernelList.funcHelmholtz_LR_DM.SetParm (k0, nDeriv, Rdir, printLog, os);
      kernelList.funcHelmholtz_SR_DM.SetParm (k0, nDeriv, Rdir, printLog, os);

      funcComplex_LR = &kernelList.funcHelmholtz_LR_DM;
      funcComplex_SR = &kernelList.funcHelmholtz_SR_DM;
    }
    else
    {
      cout << "Error in NPME_Interface_SelectKernelPtr\n";
      sprintf(str, "  EwaldSplit = %s is not defined for funcType = %s\n",
        ewaldSplit.c_str(), funcType.c_str());
      cout << str;
      return false;
    }
  }
  else
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    sprintf(str, "  funcType = %s is not defined for complex charges\n",
      funcType.c_str());
    cout << str;
    return false;
  }

  if (funcComplex == NULL)
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    cout << "  funcComplex = NULL\n";
    return false;
  }
  if (funcComplex_LR == NULL)
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    cout << "  funcComplex_LR = NULL\n";
    return false;
  }
  if (funcComplex_SR == NULL)
  {
    cout << "Error in NPME_Interface_SelectKernelPtr\n";
    cout << "  funcComplex_SR = NULL\n";
    return false;
  }

  if (printLog)
  {
    os << "\n\nNPME_Interface_SelectKernelPtr\n";
    funcComplex->Print(os);
    funcComplex_LR->Print(os);
    funcComplex_SR->Print(os);
  }



  return true;
}

bool NPME_Interface_SetUpRealKernel (
  NPME_Library::NPME_InterfaceReal*& npme, 
  NPME_Library::NPME_KfuncReal *func, 
  NPME_Library::NPME_KfuncReal *funcLR, 
  NPME_Library::NPME_KfuncReal *funcSR,
  const size_t nCharge, const double *coord,
  const NPME_Library::NPME_KeywordInput& keyword, 
  bool useGenericKernel, bool printLog, std::ostream& ofs_log)
//input:  nCharge, coord[nCharge*3], keyword,
//        func, funcLR, funcSR
//output: npme = base pointer to npme implementation class
//  a) allocates and sets up the appropriate derived npme interface class
//  b) sets 'NPME_InterfaceReal' pointer to the the derived npme interface 
//     class
//there are (optimized) specific kernel implementations for 
//  1) Laplace   + Deriv Match Ewald Splitting
//  2) Laplace   + Original    Ewald Splitting
//  3) Helmholtz + Deriv Match Ewald Splitting
//useGenericKernel = 0 uses the optimized specific kernel implementations
//useGenericKernel = 1 uses the generic            kernel implementations
{
  using std::string;
  using std::cout;
  int nDeriv;
  double Rdir;
  double beta;
  double tol;

  string funcType;
  string ewaldSplit;

  if (func == NULL)
  {
    cout << "Error in NPME_Interface_SetUpRealKernel\n";
    cout << "  func == NULL\n";
    return false;
  }
  if (funcLR == NULL)
  {
    cout << "Error in NPME_Interface_SetUpRealKernel\n";
    cout << "  funcLR == NULL\n";
    return false;
  }
  if (funcSR == NULL)
  {
    cout << "Error in NPME_Interface_SetUpRealKernel\n";
    cout << "  funcSR == NULL\n";
    return false;
  }

  if (!keyword.GetValue ("funcType", funcType))
  {
    cout << "Error in NPME_Interface_SetUpRealKernel\n";
    cout << "  funcType is not defined in keyword\n";
    return false;
  }

  if (!keyword.GetValue ("EwaldSplit", ewaldSplit))
  {
    cout << "Error in NPME_Interface_SetUpRealKernel\n";
    cout << "  EwaldSplit is not defined in keyword\n";
    return false;
  }

  char strDesc[2000];
  sprintf(strDesc, "funcType = %s and EwaldSplit = %s useGenericKernel = %d\n", 
    funcType.c_str(), ewaldSplit.c_str(), (int) useGenericKernel);

  if (funcType == "Laplace")
  {
    if (ewaldSplit == "DerivMatch")
    {
      if (useGenericKernel)
      {
        NPME_InterfaceReal_GenFunc *npmeTmp = new NPME_InterfaceReal_GenFunc;
        if (!npmeTmp->SetUp (keyword, nCharge, coord, 
          func, funcLR, funcSR, printLog, ofs_log))
        {
          cout << "Error in NPME_Interface_SetUpRealKernel\n";
          cout << "  npmeTmp->SetUp failed for ";
          cout << strDesc;
          return false;
        }
        npme = npmeTmp;
      }
      else
      {
        NPME_InterfaceReal_Laplace_DM *npmeTmp = 
                                            new NPME_InterfaceReal_Laplace_DM;
        if (!npmeTmp->SetUp (keyword, nCharge, coord, printLog, ofs_log))
        {
          cout << "Error in NPME_Interface_SetUpRealKernel\n";
          cout << "  npmeTmp->SetUp failed for ";
          cout << strDesc;
          return false;
        }
        npme = npmeTmp;
      }
    }
    else if (ewaldSplit == "LaplaceOrig")
    {
      if (useGenericKernel)
      {
        NPME_InterfaceReal_GenFunc *npmeTmp = new NPME_InterfaceReal_GenFunc;
        if (!npmeTmp->SetUp (keyword, nCharge, coord, 
          func, funcLR, funcSR, printLog, ofs_log))
        {
          cout << "Error in NPME_Interface_SetUpRealKernel\n";
          cout << "  npmeTmp->SetUp failed for ";
          cout << strDesc;
          return false;
        }
        npme = npmeTmp;
      }
      else
      {
        NPME_InterfaceReal_Laplace_Original *npmeTmp = 
                                      new NPME_InterfaceReal_Laplace_Original;
        if (!npmeTmp->SetUp (keyword, nCharge, coord, printLog, ofs_log))
        {
          cout << "Error in NPME_Interface_SetUpRealKernel\n";
          cout << "  npmeTmp->SetUp failed for ";
          cout << strDesc;
          return false;
        }
        npme = npmeTmp;


      }
    }
    else
    {
      cout << "Error in NPME_Interface_SetUpRealKernel for \n";
      cout << strDesc;
      return false;
    }
  }
  else if (funcType == "Ralpha")
  {
    if (ewaldSplit == "DerivMatch")
    {
      NPME_InterfaceReal_GenFunc *npmeTmp = new NPME_InterfaceReal_GenFunc;
      if (!npmeTmp->SetUp (keyword, nCharge, coord, 
        func, funcLR, funcSR, printLog, ofs_log))
      {
        cout << "Error in NPME_Interface_SetUpRealKernel\n";
        cout << "  npmeTmp->SetUp failed for ";
        cout << strDesc;
        return false;
      }
      npme = npmeTmp;
    }
    else
    {
      cout << "Error in NPME_Interface_SetUpRealKernel for \n";
      cout << strDesc;
      return false;
    }
  }
  else 
  {
    cout << "Error in NPME_Interface_SetUpRealKernel for \n";
    cout << strDesc;
    return false;
  }

  if (npme == NULL)
  {
    cout << "Error in NPME_Interface_SetUpRealKernel for \n";
    cout << "  npme ptr is not set\n";
    return 0;
  }

  if (printLog)
  {
    ofs_log << "\n\nNPME_Interface_SetUpRealKernel\n";
    npme->Print(ofs_log);
  }

  return true;
}

bool NPME_Interface_SetUpComplexKernel (
  NPME_Library::NPME_InterfaceComplex*& npme, 
  NPME_Library::NPME_KfuncComplex *func, 
  NPME_Library::NPME_KfuncComplex *funcLR, 
  NPME_Library::NPME_KfuncComplex *funcSR,
  const size_t nCharge, const double *coord,
  const NPME_Library::NPME_KeywordInput& keyword, 
  bool useGenericKernel, bool printLog, std::ostream& ofs_log)
//input:  nCharge, coord[nCharge*3], keyword,
//        func, funcLR, funcSR
//output: npme = base pointer to npme implementation class
//  a) allocates and sets up the appropriate derived npme interface class
//  b) sets 'NPME_InterfaceComplex' pointer to the the derived npme interface 
//     class
//there are (optimized) specific kernel implementations for 
//  1) Helmholtz + Deriv Match Ewald Splitting
//useGenericKernel = 0 uses the optimized specific kernel implementations
//useGenericKernel = 1 uses the generic            kernel implementations
{
  using std::string;
  using std::cout;
  int nDeriv;
  double Rdir;
  double beta;
  double tol;

  string funcType;
  string ewaldSplit;

  if (func == NULL)
  {
    cout << "Error in NPME_Interface_SetUpComplexKernel\n";
    cout << "  func == NULL\n";
    return false;
  }
  if (funcLR == NULL)
  {
    cout << "Error in NPME_Interface_SetUpComplexKernel\n";
    cout << "  funcLR == NULL\n";
    return false;
  }
  if (funcSR == NULL)
  {
    cout << "Error in NPME_Interface_SetUpComplexKernel\n";
    cout << "  funcSR == NULL\n";
    return false;
  }

  if (!keyword.GetValue ("funcType", funcType))
  {
    cout << "Error in NPME_Interface_SetUpComplexKernel\n";
    cout << "  funcType is not defined in keyword\n";
    return false;
  }

  if (!keyword.GetValue ("EwaldSplit", ewaldSplit))
  {
    cout << "Error in NPME_Interface_SetUpComplexKernel\n";
    cout << "  EwaldSplit is not defined in keyword\n";
    return false;
  }

  char strDesc[2000];
  sprintf(strDesc, "funcType = %s and EwaldSplit = %s useGenericKernel = %d\n", 
    funcType.c_str(), ewaldSplit.c_str(), (int) useGenericKernel);

  if (funcType == "Helmholtz")
  {
    if (ewaldSplit == "DerivMatch")
    {
      if (useGenericKernel)
      {
        NPME_InterfaceComplex_GenFunc *npmeTmp 
            = new NPME_InterfaceComplex_GenFunc;
        if (!npmeTmp->SetUp (keyword, nCharge, coord, 
          func, funcLR, funcSR, printLog, ofs_log))
        {
          cout << "Error in NPME_Interface_SetUpComplexKernel\n";
          cout << "  npmeTmp->SetUp failed for ";
          cout << strDesc;
          return false;
        }
        npme = npmeTmp;
      }
      else
      {
        NPME_InterfaceComplex_Helmholtz_DM *npmeTmp = 
                                         new NPME_InterfaceComplex_Helmholtz_DM;
        if (!npmeTmp->SetUp (keyword, nCharge, coord, printLog, ofs_log))
        {
          cout << "Error in NPME_Interface_SetUpComplexKernel\n";
          cout << "  npmeTmp->SetUp failed for ";
          cout << strDesc;
          return false;
        }
        npme = npmeTmp;
      }
    }
    else
    {
      cout << "Error in NPME_Interface_SetUpComplexKernel for \n";
      cout << strDesc;
      return false;
    }
  }
  else 
  {
    cout << "Error in NPME_Interface_SetUpComplexKernel for \n";
    cout << strDesc;
    return false;
  }
  if (printLog)
  {
    ofs_log << "\n\nNPME_Interface_SetUpComplexKernel\n";
    npme->Print(ofs_log);
  }

  return true;
}

NPME_InterfaceBase::NPME_InterfaceBase (const NPME_InterfaceBase& rhs)
{
  #if NPME_INTERFACE_DEBUG
    std::cout << "  NPME_InterfaceBase copy constructor\n";
  #endif

  _isBaseSet    = rhs._isBaseSet;
  _nProc        = rhs._nProc;
  _FFTmemGB     = rhs._FFTmemGB;
  _vecOption    = rhs._vecOption;

  //rec sum parameters
  _useCompactF  = rhs._useCompactF;
  _BnOrder      = rhs._BnOrder;
  _a1           = rhs._a1;
  _a2           = rhs._a2;
  _a3           = rhs._a3;
  _del1         = rhs._del1;
  _del2         = rhs._del2;
  _del3         = rhs._del3;
  _N1           = rhs._N1;
  _N2           = rhs._N2;
  _N3           = rhs._N3;
  _n1           = rhs._n1;
  _n2           = rhs._n2;
  _n3           = rhs._n3;
  _nCharge      = rhs._nCharge;

  //direct sum parameters
  _P            = rhs._P;
  _nNeigh       = rhs._nNeigh;
  _Rdir         = rhs._Rdir;

  _nAvgChgPerCell     = rhs._nAvgChgPerCell;
  _maxChgPerCell      = rhs._maxChgPerCell;
  _nAvgChgPerCluster  = rhs._nAvgChgPerCluster;
  _maxChgPerCluster   = rhs._maxChgPerCluster;
  _nAvgCellPerCluster = rhs._nAvgCellPerCluster;
  _maxCellPerCluster  = rhs._maxCellPerCluster;
  _nCellClust1D       = rhs._nCellClust1D;
  _cluster            = rhs._cluster;

  _kcycle       = rhs._kcycle;
}

NPME_InterfaceBase& NPME_InterfaceBase::operator= 
                                          (const NPME_InterfaceBase& rhs)
{
  #if NPME_INTERFACE_DEBUG
    std::cout << "  NPME_InterfaceBase assignment operator\n";
  #endif

  if (this != &rhs)
  {
    _isBaseSet    = rhs._isBaseSet;
    _nProc        = rhs._nProc;
    _FFTmemGB     = rhs._FFTmemGB;
    _vecOption    = rhs._vecOption;
    _nCharge      = rhs._nCharge;

    //direct sum parameters
    _P            = rhs._P;
    _nNeigh       = rhs._nNeigh;
    _Rdir         = rhs._Rdir;

    _nAvgChgPerCell     = rhs._nAvgChgPerCell;
    _maxChgPerCell      = rhs._maxChgPerCell;
    _nAvgChgPerCluster  = rhs._nAvgChgPerCluster;
    _maxChgPerCluster   = rhs._maxChgPerCluster;
    _nAvgCellPerCluster = rhs._nAvgCellPerCluster;
    _maxCellPerCluster  = rhs._maxCellPerCluster;
    _nCellClust1D       = rhs._nCellClust1D;
    _cluster            = rhs._cluster;

    _kcycle       = rhs._kcycle;

    //rec sum parameters
    _useCompactF  = rhs._useCompactF;
    _BnOrder      = rhs._BnOrder;
    _a1           = rhs._a1;
    _a2           = rhs._a2;
    _a3           = rhs._a3;
    _del1         = rhs._del1;
    _del2         = rhs._del2;
    _del3         = rhs._del3;
    _N1           = rhs._N1;
    _N2           = rhs._N2;
    _N3           = rhs._N3;
    _n1           = rhs._n1;
    _n2           = rhs._n2;
    _n3           = rhs._n3;
  }

  return *this;
}



bool NPME_InterfaceBase::ResetCoordBase (
    const size_t nCharge, const double *coord, 
    std::vector<double>& coordPermute,
    bool printLog, std::ostream& ofs_log)
{
  using std::cout;
  char str[2000];

  //direct sum set up
  _nCharge  = nCharge;
  coordPermute.resize(_nCharge*3);
  memcpy(&coordPermute[0], coord, 3*_nCharge*sizeof(double));
  _P.resize(_nCharge);

  if (!NPME_ClusterInterface (_cluster, _P, coordPermute, 
          _nAvgChgPerCell,     _maxChgPerCell,
          _nAvgChgPerCluster,  _maxChgPerCluster,
          _nAvgCellPerCluster, _maxCellPerCluster,
          _nCharge, coord, _nNeigh, _Rdir, _nCellClust1D, 
          NPME_InteractListType, _nProc))
  {
    cout << "Error in NPME_InterfaceBase::ResetCoordBase\n";
    cout << "NPME_ClusterInterface failed\n";
    return false;
  }

  if (printLog)
  {
    sprintf(str, "  nAvgChgPerCell      = %3lu (avg num of charges per cell)\n",
      _nAvgChgPerCell);
    ofs_log << str;
    sprintf(str, "  maxChgPerCell       = %3lu (max num of charges per cell)\n",
      _maxChgPerCell);
    ofs_log << str;
    sprintf(str, "  nAvgChgPerCluster   = %3lu (avg num of charges per cluster)\n",
      _nAvgChgPerCluster);
    ofs_log << str;
    sprintf(str, "  maxChgPerCluster    = %3lu (max num of charges per cluster)\n",
      _maxChgPerCluster);
    ofs_log << str;
    sprintf(str, "  nAvgCellPerCluster  = %3lu (avg num of cells per cluster)\n",
      _nAvgCellPerCluster);
    ofs_log << str;
    sprintf(str, "  maxCellPerCluster   = %3lu (max num of cells per cluster)\n",
      _maxCellPerCluster);
    ofs_log << str;

    const size_t nInteract = NPME_CountNumDirectInteract (_cluster);
    sprintf(str, "  nInteract           = %lu (tot number of dir interactions - cluster)\n",
      nInteract);
    ofs_log << str;
  }

  _kcycle.SetPermuteArray (_nCharge, &_P[0]);

  return true;
}


bool NPME_InterfaceBase::SetUpBase (
    const NPME_Library::NPME_KeywordInput& keyword,
    const size_t nCharge, const double *coord, 
    std::vector<double>& coordPermute,
    bool printLog, std::ostream& ofs_log)
//assumes ExtractKeywords is already set
{
  using std::cout;
  using std::string;

  bool PRINT_ALL = 0;

  char str[2000];
  string sTmp;
  if (!printLog)
    PRINT_ALL = 0;

  if (printLog)
  {
    ofs_log << "\n\nNPME_InterfaceBase::SetUpBase\n";
  }

  //system parameters
  {
    if (!keyword.GetValue ("nProc", _nProc))
    {
      cout << "Error in NPME_InterfaceBase::SetUpBase.\n";
      cout << "'nProc' keyword is missing \n";
      return false;
    }
    if (!keyword.GetValue ("FFTmemGB", _FFTmemGB))
    {
      cout << "Error in NPME_InterfaceBase::SetUpBase.\n";
      cout << "'FFTmemGB' keyword is missing \n";
      return false;
    }
    if (!keyword.GetValue ("vecOption", sTmp))
    {
      cout << "Error in NPME_InterfaceBase::SetUpBase.\n";
      cout << "'vecOption' keyword is missing \n";
      return false;
    }

    if      (sTmp == "none")    _vecOption = 0;
    else if (sTmp == "avx")     _vecOption = 1;
    else if (sTmp == "avx-512") _vecOption = 2;
    else
    {
      cout << "Error in NPME_InterfaceBase::SetUpBase.\n";
      sprintf(str, "  vecOption = %s is undefined\n", sTmp.c_str());
      cout << str;
      cout << "  options are 'none', 'avx', or 'avx-512'\n";
      return false;
    }
  }


  if (!keyword.GetValue ("Rdir", _Rdir))
  {
    cout << "Error in NPME_InterfaceBase::SetUpBase.\n";
    cout << "'Rdir' keyword is missing \n";
    return false;
  }

  if (!keyword.GetValue ("nNeigh", _nNeigh))
  {
    cout << "Error in NPME_InterfaceBase::SetUpBase.\n";
    cout << "'nNeigh' keyword is missing \n";
    return false;
  }
  if (!keyword.GetValue ("nCellClust1D", _nCellClust1D))
  {
    cout << "Error in NPME_InterfaceBase::SetUpBase.\n";
    cout << "'nCellClust1D' keyword is missing \n";
    return false;
  }




  if (PRINT_ALL)
  {
    sprintf(str, "  nProc         = %d\n",    _nProc);      ofs_log << str;
    sprintf(str, "  FFTmemGB      = %.2f\n",  _FFTmemGB);   ofs_log << str;
    sprintf(str, "  vecOption     = %d\n",    _vecOption);  ofs_log << str;

    sprintf(str, "  nCharge       = %lu\n", _nCharge);      ofs_log << str;
    sprintf(str, "  Rdir          = %f\n",  _Rdir);         ofs_log << str;
    sprintf(str, "  nNeigh        = %lu\n", _nNeigh);       ofs_log << str;
    sprintf(str, "  nCellClust1D  = %lu\n", _nCellClust1D); ofs_log << str;
  }

/*
  //direct sum set up
  _nCharge  = nCharge;
  coordPermute.resize(_nCharge*3);
  memcpy(&coordPermute[0], coord, 3*_nCharge*sizeof(double));
  _P.resize(_nCharge);
  {
    if (!NPME_ClusterInterface (_cluster, _P, coordPermute, 
            _nAvgChgPerCell,     _maxChgPerCell,
            _nAvgChgPerCluster,  _maxChgPerCluster,
            _nAvgCellPerCluster, _maxCellPerCluster,
            _nCharge, coord, _nNeigh, _Rdir, _nCellClust1D, 
            NPME_InteractListType, _nProc))
    {
      cout << "Error in NPME_InterfaceBase::SetUpBase\n";
      cout << "NPME_ClusterInterface failed\n";
      return false;
    }

    if (printLog)
    {
      sprintf(str, "  nAvgChgPerCell      = %3lu (avg num of charges per cell)\n",
        _nAvgChgPerCell);
      ofs_log << str;
      sprintf(str, "  maxChgPerCell       = %3lu (max num of charges per cell)\n",
        _maxChgPerCell);
      ofs_log << str;
      sprintf(str, "  nAvgChgPerCluster   = %3lu (avg num of charges per cluster)\n",
        _nAvgChgPerCluster);
      ofs_log << str;
      sprintf(str, "  maxChgPerCluster    = %3lu (max num of charges per cluster)\n",
        _maxChgPerCluster);
      ofs_log << str;
      sprintf(str, "  nAvgCellPerCluster  = %3lu (avg num of cells per cluster)\n",
        _nAvgCellPerCluster);
      ofs_log << str;
      sprintf(str, "  maxCellPerCluster   = %3lu (max num of cells per cluster)\n",
        _maxCellPerCluster);
      ofs_log << str;

      const size_t nInteract = NPME_CountNumDirectInteract (_cluster);
      sprintf(str, "  nInteract           = %lu (tot number of dir interactions - cluster)\n",
        nInteract);
      ofs_log << str;
    }
  }
  _kcycle.SetPermuteArray (_nCharge, &_P[0]);
*/

  if (!ResetCoordBase (nCharge, coord, coordPermute, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceBase::SetUpBase\n";
    cout << "ResetCoordBase failed\n";
    return false;
  }
  

  //get rec sum parameters
  {
    bool error = 0;
    if (!keyword.GetValue ("N1",      _N1))       error = 1;
    if (!keyword.GetValue ("N2",      _N2))       error = 1;
    if (!keyword.GetValue ("N3",      _N3))       error = 1;

    if (!keyword.GetValue ("del1",    _del1))     error = 1;
    if (!keyword.GetValue ("del2",    _del2))     error = 1;
    if (!keyword.GetValue ("del3",    _del3))     error = 1;

    if (!keyword.GetValue ("n1",      _n1))       error = 1;
    if (!keyword.GetValue ("n2",      _n2))       error = 1;
    if (!keyword.GetValue ("n3",      _n3))       error = 1;

    if (!keyword.GetValue ("a1",      _a1))       error = 1;
    if (!keyword.GetValue ("a2",      _a2))       error = 1;
    if (!keyword.GetValue ("a3",      _a3))       error = 1;

    if (!keyword.GetValue ("BnOrder", _BnOrder))  error = 1;
    if (!keyword.GetValue ("useCompactF", _useCompactF))
      error = 1;

    if (error)
    {
      cout << "Error in NPME_InterfaceBase::SetUpBase\n";
      cout << "  missing rec sum parameters\n";
      return false;
    }
  }


  if (printLog)
    PrintBase (ofs_log);


  return true;
}

void NPME_InterfaceBase::PrintBase (std::ostream& ofs_log) const
{
  char str[2000];
  char strOut[2000];

  char strVecOption[3][20];
  sprintf(strVecOption[0], "none");
  sprintf(strVecOption[1], "avx");
  sprintf(strVecOption[2], "avx-512");

  ofs_log << "\n\nNPME_InterfaceBase::PrintBase\n";

  sprintf(strOut, "  nProc       = %d\n", _nProc);
  ofs_log << strOut;

  sprintf(strOut, "  MaxFFTmemGB = %.2f\n", _FFTmemGB);
  ofs_log << strOut;

  sprintf(strOut, "  vecOption   = %s\n",   strVecOption[_vecOption]); 
  ofs_log << strOut;

  ofs_log << "\n";

  sprintf(strOut, "  nCharge     = %lu\n", _nCharge);
  ofs_log << strOut;

  sprintf(str, "  Rdir        = %f", _Rdir);
  sprintf(strOut, "%-50s (Direct Space Cutoff)\n", str);
  ofs_log << strOut;

  double FFTmemGB = NPME_CalcFFTmemGB (
                      _N1, _N2, _N3,
                      _n1, _useCompactF);


  sprintf(str, "  FFTmemGB    = %.3le GB", FFTmemGB);
  sprintf(strOut, "%-50s (FFT memory)\n", str);
  ofs_log << strOut;

  sprintf(str, "  N1 N2 N3    = %4ld %4ld %4ld",  _N1, _N2, _N3);
  sprintf(strOut, "%-50s (FFT Sizes)\n", str);
  ofs_log << strOut;

  sprintf(str, "  n1 n2 n3    = %4ld %4ld %4ld", _n1, _n2, _n3);
  sprintf(strOut, "%-50s (FFT Block Sizes)\n", str);
  ofs_log << strOut;

  sprintf(str, "  BnOrder     = %4ld", _BnOrder);
  sprintf(strOut, "%-50s (B-Spline Order)\n", str);
  ofs_log << strOut;

  sprintf(str, "  a           = %8.4f %8.4f %8.4f", _a1, _a2, _a3);
  sprintf(strOut, "%-50s (Four. Ext. Parm)\n", str);
  ofs_log << strOut;

  sprintf(str, "  del         = %8.4f %8.4f %8.4f", _del1, _del2, _del3);
  sprintf(strOut, "%-50s (Four. Ext. Parm)\n", str);
  ofs_log << strOut;

  ofs_log.flush();
}






//******************************************************************************
//******************************************************************************
//***************************NPME_InterfaceReal_GenFunc*************************
//******************************************************************************
//******************************************************************************



bool NPME_InterfaceReal_GenFunc::SetUp (
  const NPME_Library::NPME_KeywordInput& keyword,
  const size_t nCharge, const double *coord, 
  NPME_Library::NPME_KfuncReal *func, 
  NPME_Library::NPME_KfuncReal *funcLR,
  NPME_Library::NPME_KfuncReal *funcSR,
  bool printLog, std::ostream& ofs_log)
{
  _func   = func;
  _funcLR = funcLR;
  _funcSR = funcSR;

  using std::cout;
  using std::string;

  if (func == NULL)
  {
    cout << "Error in NPME_InterfaceReal_GenFunc::SetUp\n";
    cout << "  func == NULL\n";
    return false;
  }
  if (funcLR == NULL)
  {
    cout << "Error in NPME_InterfaceReal_GenFunc::SetUp\n";
    cout << "  funcLR == NULL\n";
    return false;
  }
  if (funcSR == NULL)
  {
    cout << "Error in NPME_InterfaceReal_GenFunc::SetUp\n";
    cout << "  funcSR == NULL\n";
    return false;
  }

  std::vector<double> coordPermute;
  if (!SetUpBase (keyword, nCharge, coord, coordPermute, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_GenFunc::SetUp.\n";
    cout << "  SetUpBase failed\n";
    return false;
  }

  if (!_recSum.SetUp (nCharge, &coordPermute[0], _BnOrder, _useCompactF,
                _N1,    _N2,    _N3,
                _n1,    _n2,    _n3,
                _del1,  _del2,  _del3,
                _a1,    _a2,    _a3,
                _funcLR, _nProc, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_GenFunc::SetUp.\n";
    cout << "  recSum.SetUp (..) failed\n";
    return false;
  }

  //self interaction smooth kernel _fself = funcLR(0,0,0)
  {
    double x = 0;
    double y = 0;
    double z = 0;
    (*_funcLR).Calc (1, &x, &y, &z);
    _fself = x;
  }


  return true;
}



bool NPME_InterfaceReal_GenFunc::ResetCoord (
  const size_t nCharge, const double *coord, 
  bool printLog, std::ostream& ofs_log)
{
  using std::cout;
  using std::string;

  std::vector<double> coordPermute;

  if (!ResetCoordBase (nCharge, coord, coordPermute, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_GenFunc::ResetCoord\n";
    cout << "ResetCoordBase failed\n";
    return false;
  }

  if (!_recSum.SetUp (nCharge, &coordPermute[0], _BnOrder, _useCompactF,
                _N1,    _N2,    _N3,
                _n1,    _n2,    _n3,
                _del1,  _del2,  _del3,
                _a1,    _a2,    _a3,
                _funcLR, _nProc, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_GenFunc::ResetCoord.\n";
    cout << "  recSum.SetUp (..) failed\n";
    return false;
  }

  return true;
}




void NPME_InterfaceReal_GenFunc::Print (std::ostream& ofs_log)
{
  ofs_log << "\n\nNPME_InterfaceReal_GenFunc::Print\n";
  ofs_log << "  fself = " << _fself << "\n";
  PrintBase (ofs_log);
  _recSum.Print (ofs_log);
}



bool NPME_InterfaceReal_GenFunc::CalcV (double *V1, const size_t nCharge, 
              double *charge, double *coord, 
              bool printLog, std::ostream& ofs_log)
//PME V1 functions for real charges
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]
{
  double time0, time;
  double timeStart, timeTot;
  char str[2000];
  bool zeroVarray = 0;
  bool PRINT_ALL = 0;

  timeStart = NPME_GetTime ();
  if (printLog)
  {
    ofs_log << "\n\nNPME_InterfaceReal_GenFunc::CalcV\n";
    ofs_log.flush();
  }

  //1) permute charge[_nCharge]
  time0 = NPME_GetTime ();
  NPME_Kcycle_InversePermuteArray_MN (1, _nCharge, _kcycle, 
    charge, _nProc);
  NPME_Kcycle_InversePermuteArray_MN (3, _nCharge, _kcycle, 
    coord, _nProc);
  time = NPME_GetTime () - time0;
  if (printLog)
  {
    sprintf(str, " time = %6.2le (permute charge, coord)\n", time);
    ofs_log << str;
    ofs_log.flush();
  }

  //2) direct sum + self
  {
    time0 = NPME_GetTime ();
    //direct
    NPME_PotGenFunc_DirectSum_V1 (*_funcSR, _nCharge, coord,
      charge, V1, _nProc, _vecOption, _cluster);
    //Vself
    for (size_t i = 0; i < _nCharge; i++)
      V1[4*i] -= _fself*charge[i];
    time = NPME_GetTime () - time0;
    if (printLog)
    {
      sprintf(str, " time = %6.2le (direct sum)\n", time);
      ofs_log << str;
      ofs_log.flush();
    }
  }


  //3) rec sum
  time0 = NPME_GetTime ();
  _recSum.CalcV1 (V1, nCharge, charge, coord, zeroVarray, PRINT_ALL, ofs_log);
  time = NPME_GetTime () - time0;
  if (printLog)
  {
    sprintf(str, " time = %6.2le (rec sum)\n", time);
    ofs_log << str;
    ofs_log.flush();
  }

  //4) permute V1 to original frame 
  NPME_Kcycle_PermuteArray_MN (4, _nCharge, _kcycle, V1, _nProc);

  //5) permute charge[_nCharge] back to original frame 
  NPME_Kcycle_PermuteArray_MN (1, _nCharge, _kcycle, charge, _nProc);
  NPME_Kcycle_PermuteArray_MN (3, _nCharge, _kcycle, coord,  _nProc);

  timeTot = NPME_GetTime () - timeStart;
  if (printLog)
  {
    sprintf(str, "--------------------------------------\n");
    sprintf(str, " time = %6.2le (total V1)\n\n\n", timeTot);
    ofs_log << str;
    ofs_log.flush();
  }

  return true;
}


bool NPME_InterfaceReal_GenFunc::CalcV_exact (double *V1, const size_t nCharge, 
              double *charge, double *coord, 
              bool printLog, std::ostream& ofs_log)
//exact V1 functions (brute force ~N^2 sum)
//input:  charge[nCharge] (real)
//output: V1[nCharge][4]
{
  if (nCharge != _nCharge)
  {
    char str[2000];
    std::cout << "Error in NPME_InterfaceReal_GenFunc::CalcV_exact.\n";
    sprintf(str, "  nCharge = %lu != %lu\n", nCharge, _nCharge);
    std::cout << str;
    return false;
  }

  double time0 = NPME_GetTime ();
  NPME_PotGenFunc_MacroSelf_V1 (*_func, _nCharge, coord, charge, 
              V1, _nProc, _vecOption);
  double time = NPME_GetTime () - time0;

  if (printLog)
  {
    char str[500];
    ofs_log << "\n\nNPME_InterfaceReal_GenFunc::CalcV_exact\n";
    sprintf(str, "  time_exact = %.2f\n", time);
    ofs_log << str;
  }

  return 1;
}





//******************************************************************************
//******************************************************************************
//************************NPME_InterfaceReal_Laplace_DM*************************
//******************************************************************************
//******************************************************************************



bool NPME_InterfaceReal_Laplace_DM::SetUp (
  const NPME_Library::NPME_KeywordInput& keyword,
  const size_t nCharge, const double *coord,
  bool printLog, std::ostream& ofs_log)
{
  using std::cout;
  using std::string;


  string funcType;
  string ewaldType;
  if (!keyword.GetValue ("funcType", funcType))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::SetUp\n";
    cout << "  Missing required keyword 'funcType'\n";
    return false;
  }

  if (!keyword.GetValue ("EwaldSplit", ewaldType))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::SetUp\n";
    cout << "  Missing required keyword 'EwaldSplit'\n";
    return false;
  }
  if (!keyword.GetValue ("Rdir", _Rdir))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::SetUp\n";
    cout << "  Missing required keyword 'Rdir'\n";
    return false;
  }
  if (!keyword.GetValue ("nDeriv", _Nder))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::SetUp\n";
    cout << "  Missing required keyword 'nDeriv'\n";
    return false;
  }

  if (funcType != "Laplace")
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::SetUp\n";
    char str[2000];
    sprintf(str, "  funcType = %s != 'Laplace'\n", funcType.c_str());
    cout << str;
    return false;
  }
  if (ewaldType != "DerivMatch")
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::SetUp\n";
    char str[2000];
    sprintf(str, "  ewaldType = %s != 'DerivMatch'\n", ewaldType.c_str());
    cout << str;
    return false;
  }


  std::vector<double> coordPermute;
  if (!SetUpBase (keyword, nCharge, coord, coordPermute, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::SetUp.\n";
    cout << "  SetUpBase failed\n";
    return false;
  }

  double f[NPME_MaxDerivMatchOrder+1];
  NPME_FunctionDerivMatch_RalphaRadialDeriv (f, _Nder, _Rdir, -1.0);
  if (!NPME_FunctionDerivMatch_CalcEvenSeries (&_a[0], &_b[0], 
    &f[0], _Nder, _Rdir))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::SetUp.\n";
    cout << "  NPME_FunctionDerivMatch_CalcEvenSeries failed\n";
    return false;
  }

  if (!_funcLR.SetParm (_Nder, _Rdir, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::SetUp.\n";
    cout << "  _funcLR.SetParm failed\n";
    return false;
  }
  if (!_funcSR.SetParm (_Nder, _Rdir, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::SetUp.\n";
    cout << "  _funcSR.SetParm failed\n";
    return false;
  }

  if (!_recSum.SetUp (nCharge, &coordPermute[0], _BnOrder, _useCompactF,
                _N1,    _N2,    _N3,
                _n1,    _n2,    _n3,
                _del1,  _del2,  _del3,
                _a1,    _a2,    _a3,
                &_funcLR, _nProc, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::SetUp.\n";
    cout << "  recSum.SetUp (..) failed\n";
    return false;
  }

  //self interaction smooth kernel _fself = funcLR(0,0,0)
  {
    double x = 0;
    double y = 0;
    double z = 0;
    _funcLR.Calc (1, &x, &y, &z);
    _fself = x;
  }


  return true;
}

bool NPME_InterfaceReal_Laplace_DM::ResetCoord (
  const size_t nCharge, const double *coord, 
  bool printLog, std::ostream& ofs_log)
{
  using std::cout;
  using std::string;

  std::vector<double> coordPermute;

  if (!ResetCoordBase (nCharge, coord, coordPermute, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::ResetCoord\n";
    cout << "ResetCoordBase failed\n";
    return false;
  }

  if (!_recSum.SetUp (nCharge, &coordPermute[0], _BnOrder, _useCompactF,
                _N1,    _N2,    _N3,
                _n1,    _n2,    _n3,
                _del1,  _del2,  _del3,
                _a1,    _a2,    _a3,
                &_funcLR, _nProc, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_DM::ResetCoord.\n";
    cout << "  recSum.SetUp (..) failed\n";
    return false;
  }

  return true;
}

void NPME_InterfaceReal_Laplace_DM::Print (std::ostream& ofs_log)
{
  ofs_log << "\n\nNPME_InterfaceReal_Laplace_DM::Print\n";
  ofs_log << "  fself = " << _fself << "\n";
  
  char str[2000];
  sprintf(str, "  Nder  = %d\n", _Nder);   ofs_log << str;
  sprintf(str, "  Rdir  = %f\n", _Rdir);   ofs_log << str;

  ofs_log << "\n";
  for (int i = 0; i <= _Nder; i++)
  {
    sprintf(str, "      a[%4d] = %15.6le\n", i, _a[i]);
    ofs_log << str;
  }
  ofs_log << "\n";
  for (int i = 0; i <= _Nder; i++)
  {
    sprintf(str, "      b[%4d] = %15.6le\n", i, _b[i]);
    ofs_log << str;
  }

  PrintBase (ofs_log);
  _recSum.Print (ofs_log);
}



bool NPME_InterfaceReal_Laplace_DM::CalcV (double *V1, const size_t nCharge, 
              double *charge, double *coord, 
              bool printLog, std::ostream& ofs_log)
//PME V1 functions for real charges
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]
{
  double time0, time;
  double timeStart, timeTot;
  char str[2000];
  bool zeroVarray = 0;
  bool PRINT_ALL = 0;

  timeStart = NPME_GetTime ();
  if (printLog)
  {
    ofs_log << "\n\nNPME_InterfaceReal_Laplace_DM::CalcV\n";
    ofs_log.flush();
  }

  //1) permute charge[_nCharge]
  time0 = NPME_GetTime ();
  NPME_Kcycle_InversePermuteArray_MN (1, _nCharge, _kcycle, 
    charge, _nProc);
  NPME_Kcycle_InversePermuteArray_MN (3, _nCharge, _kcycle, 
    coord, _nProc);
  time = NPME_GetTime () - time0;
  if (printLog)
  {
    sprintf(str, " time = %6.2le (permute charge, coord)\n", time);
    ofs_log << str;
    ofs_log.flush();
  }

  //2) direct sum + self
  {
    time0 = NPME_GetTime ();
    //direct
    NPME_PotLaplace_SR_DM_DirectSum_V1 (
      _Nder, _a, _b, _Rdir,
      _nCharge, coord,
      charge, V1, _nProc, _vecOption, _cluster);

    //Vself
    for (size_t i = 0; i < _nCharge; i++)
      V1[4*i] -= _fself*charge[i];
    time = NPME_GetTime () - time0;
    if (printLog)
    {
      sprintf(str, " time = %6.2le (direct sum)\n", time);
      ofs_log << str;
      ofs_log.flush();
    }
  }

  //3) rec sum
  time0 = NPME_GetTime ();
  _recSum.CalcV1 (V1, nCharge, charge, coord, zeroVarray, PRINT_ALL, ofs_log);
  time = NPME_GetTime () - time0;
  if (printLog)
  {
    sprintf(str, " time = %6.2le (rec sum)\n", time);
    ofs_log << str;
    ofs_log.flush();
  }

  //4) permute V1 to original frame 
  NPME_Kcycle_PermuteArray_MN (4, _nCharge, _kcycle, V1, _nProc);

  //5) permute charge[_nCharge] back to original frame 
  NPME_Kcycle_PermuteArray_MN (1, _nCharge, _kcycle, charge, _nProc);
  NPME_Kcycle_PermuteArray_MN (3, _nCharge, _kcycle, coord,  _nProc);

  timeTot = NPME_GetTime () - timeStart;
  if (printLog)
  {
    sprintf(str, "--------------------------------------\n");
    sprintf(str, " time = %6.2le (total V1)\n\n\n", timeTot);
    ofs_log << str;
    ofs_log.flush();
  }

  return true;
}










bool NPME_InterfaceReal_Laplace_DM::CalcV_exact (double *V1, const size_t nCharge, 
              double *charge, double *coord, 
              bool printLog, std::ostream& ofs_log)
//exact V1 functions (brute force ~N^2 sum)
//input:  charge[nCharge] (real)
//output: V1[nCharge][4]
{
  if (nCharge != _nCharge)
  {
    char str[2000];
    std::cout << "Error in NPME_InterfaceReal_Laplace_DM::CalcV_exact.\n";
    sprintf(str, "  nCharge = %lu != %lu\n", nCharge, _nCharge);
    std::cout << str;
    return false;
  }
  double time0 = NPME_GetTime ();
  NPME_PotLaplace_MacroSelf_V1 (_nCharge, coord, charge, 
              V1, _nProc, _vecOption);
  double time = NPME_GetTime () - time0;

  if (printLog)
  {
    char str[500];
    ofs_log << "\n\nNPME_InterfaceReal_Laplace_DM::CalcV_exact\n";
    sprintf(str, "  time_exact = %.2f\n", time);
    ofs_log << str;
  }

  return 1;
}



//******************************************************************************
//******************************************************************************
//************************NPME_InterfaceReal_Laplace_Original*******************
//******************************************************************************
//******************************************************************************



bool NPME_InterfaceReal_Laplace_Original::SetUp (
  const NPME_Library::NPME_KeywordInput& keyword,
  const size_t nCharge, const double *coord,
  bool printLog, std::ostream& ofs_log)
{
  using std::cout;
  using std::string;


  string funcType;
  string ewaldType;
  double Rdir, tol;

  if (!keyword.GetValue ("funcType", funcType))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_Original::SetUp\n";
    cout << "  Missing required keyword 'funcType'\n";
    return false;
  }
  if (!keyword.GetValue ("EwaldSplit", ewaldType))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_Original::SetUp\n";
    cout << "  Missing required keyword 'EwaldSplit'\n";
    return false;
  }
  if (!keyword.GetValue ("Rdir", Rdir))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_Original::SetUp\n";
    cout << "  Missing required keyword 'Rdir'\n";
    return false;
  }
  if (!keyword.GetValue ("tol", tol))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_Original::SetUp\n";
    cout << "  Missing required keyword 'tol'\n";
    return false;
  }

  if (funcType != "Laplace")
  {
    cout << "Error in NPME_InterfaceReal_Laplace_Original::SetUp\n";
    char str[2000];
    sprintf(str, "  funcType = %s != 'Laplace'\n", funcType.c_str());
    cout << str;
    return false;
  }
  if (ewaldType != "LaplaceOrig")
  {
    cout << "Error in NPME_InterfaceReal_Laplace_Original::SetUp\n";
    char str[2000];
    sprintf(str, "  ewaldType = %s != 'LaplaceOrig'\n", ewaldType.c_str());
    cout << str;
    return false;
  }

  _beta = NPME_EwaldSplitOrig_Rdir2Beta (Rdir, tol);

  std::vector<double> coordPermute;
  if (!SetUpBase (keyword, nCharge, coord, coordPermute, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_Original::SetUp.\n";
    cout << "  SetUpBase failed\n";
    return false;
  }


  _funcLR.SetParm (_beta);
  _funcSR.SetParm (_beta);

  if (!_recSum.SetUp (nCharge, &coordPermute[0], _BnOrder, _useCompactF,
                _N1,    _N2,    _N3,
                _n1,    _n2,    _n3,
                _del1,  _del2,  _del3,
                _a1,    _a2,    _a3,
                &_funcLR, _nProc, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_Original::SetUp.\n";
    cout << "  recSum.SetUp (..) failed\n";
    return false;
  }

  //self interaction smooth kernel _fself = funcLR(0,0,0)
  {
    double x = 0;
    double y = 0;
    double z = 0;
    _funcLR.Calc (1, &x, &y, &z);
    _fself = x;
  }


  return true;
}

bool NPME_InterfaceReal_Laplace_Original::ResetCoord (
  const size_t nCharge, const double *coord, 
  bool printLog, std::ostream& ofs_log)
{
  using std::cout;
  using std::string;

  std::vector<double> coordPermute;

  if (!ResetCoordBase (nCharge, coord, coordPermute, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_Original::ResetCoord\n";
    cout << "ResetCoordBase failed\n";
    return false;
  }

  if (!_recSum.SetUp (nCharge, &coordPermute[0], _BnOrder, _useCompactF,
                _N1,    _N2,    _N3,
                _n1,    _n2,    _n3,
                _del1,  _del2,  _del3,
                _a1,    _a2,    _a3,
                &_funcLR, _nProc, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceReal_Laplace_Original::ResetCoord.\n";
    cout << "  recSum.SetUp (..) failed\n";
    return false;
  }

  return true;
}



void NPME_InterfaceReal_Laplace_Original::Print (std::ostream& ofs_log)
{
  ofs_log << "\n\nNPME_InterfaceReal_Laplace_Original::Print\n";
  ofs_log << "  fself = " << _fself << "\n";
  ofs_log << "  beta  = " << _beta  << "\n";

  PrintBase (ofs_log);
  _recSum.Print (ofs_log);
}



bool NPME_InterfaceReal_Laplace_Original::CalcV (double *V1, const size_t nCharge, 
              double *charge, double *coord, 
              bool printLog, std::ostream& ofs_log)
//PME V1 functions for real charges
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]
{
  double time0, time;
  double timeStart, timeTot;
  char str[2000];
  bool zeroVarray = 0;
  bool PRINT_ALL = 0;

  timeStart = NPME_GetTime ();
  if (printLog)
  {
    ofs_log << "\n\nNPME_InterfaceReal_Laplace_Original::CalcV\n";
    ofs_log.flush();
  }

  //1) permute charge[_nCharge]
  time0 = NPME_GetTime ();
  NPME_Kcycle_InversePermuteArray_MN (1, _nCharge, _kcycle, 
    charge, _nProc);
  NPME_Kcycle_InversePermuteArray_MN (3, _nCharge, _kcycle, 
    coord, _nProc);
  time = NPME_GetTime () - time0;
  if (printLog)
  {
    sprintf(str, " time = %6.2le (permute charge, coord)\n", time);
    ofs_log << str;
    ofs_log.flush();
  }

  //2) direct sum + self
  {
    time0 = NPME_GetTime ();
    //direct
    NPME_PotLaplace_SR_Original_DirectSum_V1 (_beta,
      _nCharge, coord,
      charge, V1, _nProc, _vecOption, _cluster);
    //Vself
    for (size_t i = 0; i < _nCharge; i++)
      V1[4*i] -= _fself*charge[i];
    time = NPME_GetTime () - time0;
    if (printLog)
    {
      sprintf(str, " time = %6.2le (direct sum)\n", time);
      ofs_log << str;
      ofs_log.flush();
    }
  }


  //3) rec sum
  time0 = NPME_GetTime ();
  _recSum.CalcV1 (V1, nCharge, charge, coord, zeroVarray, PRINT_ALL, ofs_log);
  time = NPME_GetTime () - time0;
  if (printLog)
  {
    sprintf(str, " time = %6.2le (rec sum)\n", time);
    ofs_log << str;
    ofs_log.flush();
  }

  //4) permute V1 to original frame 
  NPME_Kcycle_PermuteArray_MN (4, _nCharge, _kcycle, V1, _nProc);

  //5) permute charge[_nCharge] back to original frame 
  NPME_Kcycle_PermuteArray_MN (1, _nCharge, _kcycle, charge, _nProc);
  NPME_Kcycle_PermuteArray_MN (3, _nCharge, _kcycle, coord,  _nProc);

  timeTot = NPME_GetTime () - timeStart;
  if (printLog)
  {
    sprintf(str, "--------------------------------------\n");
    sprintf(str, " time = %6.2le (total V1)\n\n\n", timeTot);
    ofs_log << str;
    ofs_log.flush();
  }

  return true;
}

bool NPME_InterfaceReal_Laplace_Original::CalcV_exact (double *V1, 
              const size_t nCharge, 
              double *charge, double *coord, 
              bool printLog, std::ostream& ofs_log)
//exact V1 functions (brute force ~N^2 sum)
//input:  charge[nCharge] (real)
//output: V1[nCharge][4]
{
  if (nCharge != _nCharge)
  {
    using std::cout;
    char str[2000];
    cout << "Error in NPME_InterfaceReal_Laplace_Original::CalcV_exact.\n";
    sprintf(str, "  nCharge = %lu != %lu\n", nCharge, _nCharge);
    cout << str;
    return false;
  }
  double time0 = NPME_GetTime ();
  NPME_PotLaplace_MacroSelf_V1 (_nCharge, coord, charge, 
              V1, _nProc, _vecOption);
  double time = NPME_GetTime () - time0;

  if (printLog)
  {
    char str[500];
    ofs_log << "\n\nNPME_InterfaceReal_Laplace_Original::CalcV_exact\n";
    sprintf(str, "  time_exact = %.2f\n", time);
    ofs_log << str;
  }

  return 1;
}




//******************************************************************************
//******************************************************************************
//************************NPME_InterfaceComplex_GenFunc*************************
//******************************************************************************
//******************************************************************************



bool NPME_InterfaceComplex_GenFunc::SetUp (
  const NPME_Library::NPME_KeywordInput& keyword,
  const size_t nCharge, const double *coord,
  NPME_Library::NPME_KfuncComplex *func, 
  NPME_Library::NPME_KfuncComplex *funcLR,
  NPME_Library::NPME_KfuncComplex *funcSR,
  bool printLog, std::ostream& ofs_log)
{
  _func   = func;
  _funcLR = funcLR;
  _funcSR = funcSR;

  using std::cout;
  using std::string;

  if (func == NULL)
  {
    cout << "Error in NPME_InterfaceComplex_GenFunc::SetUp\n";
    cout << "  func == NULL\n";
    return false;
  }
  if (funcLR == NULL)
  {
    cout << "Error in NPME_InterfaceComplex_GenFunc::SetUp\n";
    cout << "  funcLR == NULL\n";
    return false;
  }
  if (funcSR == NULL)
  {
    cout << "Error in NPME_InterfaceComplex_GenFunc::SetUp\n";
    cout << "  funcSR == NULL\n";
    return false;
  }

  std::vector<double> coordPermute;
  if (!SetUpBase (keyword, nCharge, coord, coordPermute, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceComplex_GenFunc::SetUp.\n";
    cout << "  SetUpBase failed\n";
    return false;
  }

  if (!_recSum.SetUp (nCharge, &coordPermute[0], _BnOrder, _useCompactF,
                _N1,    _N2,    _N3,
                _n1,    _n2,    _n3,
                _del1,  _del2,  _del3,
                _a1,    _a2,    _a3,
                _funcLR, _nProc, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceComplex_GenFunc::SetUp.\n";
    cout << "  recSum.SetUp (..) failed\n";
    return false;
  }

  //self interaction smooth kernel _fself = funcLR(0,0,0)
  {
    double f0_r_x = 0;
    double f0_i   = 0;
    double y      = 0;
    double z      = 0;
    (*_funcLR).Calc (1, &f0_r_x, &f0_i, &y, &z);
    _fself = f0_r_x + I*f0_i;
  }

  return true;
}

bool NPME_InterfaceComplex_GenFunc::ResetCoord (
  const size_t nCharge, const double *coord, 
  bool printLog, std::ostream& ofs_log)
{
  using std::cout;
  using std::string;

  std::vector<double> coordPermute;

  if (!ResetCoordBase (nCharge, coord, coordPermute, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceComplex_GenFunc::ResetCoord\n";
    cout << "ResetCoordBase failed\n";
    return false;
  }

  if (!_recSum.SetUp (nCharge, &coordPermute[0], _BnOrder, _useCompactF,
                _N1,    _N2,    _N3,
                _n1,    _n2,    _n3,
                _del1,  _del2,  _del3,
                _a1,    _a2,    _a3,
                _funcLR, _nProc, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceComplex_GenFunc::ResetCoord.\n";
    cout << "  recSum.SetUp (..) failed\n";
    return false;
  }

  return true;
}

void NPME_InterfaceComplex_GenFunc::Print (std::ostream& ofs_log)
{
  ofs_log << "\n\nNPME_InterfaceComplex_GenFunc::Print\n";
  ofs_log << "  fself = " << _fself << "\n";
  PrintBase (ofs_log);
  _recSum.Print (ofs_log);
}



bool NPME_InterfaceComplex_GenFunc::CalcV (
              _Complex double *V1, const size_t nCharge, 
              _Complex double *charge, double *coord, 
              bool printLog, std::ostream& ofs_log)
//PME V1 functions for real charges
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]
{
  double time0, time;
  double timeStart, timeTot;
  char str[2000];
  bool zeroVarray = 0;
  bool PRINT_ALL = 0;

  timeStart = NPME_GetTime ();
  if (printLog)
  {
    ofs_log << "\n\nNPME_InterfaceComplex_GenFunc::CalcV\n";
    ofs_log.flush();
  }

  //1) permute charge[_nCharge]
  time0 = NPME_GetTime ();
  NPME_Kcycle_InversePermuteArray_MN (2, _nCharge, _kcycle, 
    (double *) charge, _nProc);
  NPME_Kcycle_InversePermuteArray_MN (3, _nCharge, _kcycle, 
    coord, _nProc);
  time = NPME_GetTime () - time0;
  if (printLog)
  {
    sprintf(str, " time = %6.2le (permute charge, coord)\n", time);
    ofs_log << str;
    ofs_log.flush();
  }

  //2) direct sum + self
  {
    time0 = NPME_GetTime ();
    //direct
    NPME_PotGenFunc_DirectSum_V1 (*_funcSR, _nCharge, coord,
      charge, V1, _nProc, _vecOption, _cluster);

    //Vself
    for (size_t i = 0; i < _nCharge; i++)
      V1[4*i] -= _fself*charge[i];
    time = NPME_GetTime () - time0;
    if (printLog)
    {
      sprintf(str, " time = %6.2le (direct sum)\n", time);
      ofs_log << str;
      ofs_log.flush();
    }
  }


  //3) rec sum
  time0 = NPME_GetTime ();
  _recSum.CalcV1 (V1, nCharge, charge, coord, zeroVarray, PRINT_ALL, ofs_log);
  time = NPME_GetTime () - time0;
  if (printLog)
  {
    sprintf(str, " time = %6.2le (rec sum)\n", time);
    ofs_log << str;
    ofs_log.flush();
  }

  //4) permute V1 to original frame 
  NPME_Kcycle_PermuteArray_MN (8, _nCharge, _kcycle, (double *) V1, _nProc);

  //5) permute charge[_nCharge] back to original frame 
  NPME_Kcycle_PermuteArray_MN (2, _nCharge, _kcycle, (double *) charge, _nProc);
  NPME_Kcycle_PermuteArray_MN (3, _nCharge, _kcycle, coord,  _nProc);

  timeTot = NPME_GetTime () - timeStart;
  if (printLog)
  {
    sprintf(str, "--------------------------------------\n");
    sprintf(str, " time = %6.2le (total V1)\n\n\n", timeTot);
    ofs_log << str;
    ofs_log.flush();
  }

  return true;
}








bool NPME_InterfaceComplex_GenFunc::CalcV_exact (
              _Complex double *V1, const size_t nCharge, 
              _Complex double *charge, double *coord, 
              bool printLog, std::ostream& ofs_log)
//exact V1 functions (brute force ~N^2 sum)
//input:  charge[nCharge] (real)
//output: V1[nCharge][4]
{
  if (nCharge != _nCharge)
  {
    char str[2000];
    std::cout << "Error in NPME_InterfaceComplex_GenFunc::CalcV_exact.\n";
    sprintf(str, "  nCharge = %lu != %lu\n", nCharge, _nCharge);
    std::cout << str;
    return false;
  }
  double time0 = NPME_GetTime ();
  NPME_PotGenFunc_MacroSelf_V1 (*_func, _nCharge, coord, charge, 
              V1, _nProc, _vecOption);
  double time = NPME_GetTime () - time0;

  if (printLog)
  {
    char str[500];
    ofs_log << "\n\nNPME_InterfaceComplex_GenFunc::CalcV_exact\n";
    sprintf(str, "  time_exact = %.2f\n", time);
    ofs_log << str;
  }


  return 1;
}


//******************************************************************************
//******************************************************************************
//*******************NPME_InterfaceComplex_Helmholtz_DM*************************
//******************************************************************************
//******************************************************************************



bool NPME_InterfaceComplex_Helmholtz_DM::SetUp (
  const NPME_Library::NPME_KeywordInput& keyword,
  const size_t nCharge, const double *coord,
  bool printLog, std::ostream& ofs_log)
{
  using std::cout;
  using std::string;

  string funcType;
  string ewaldType;
  if (!keyword.GetValue ("funcType", funcType))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp\n";
    cout << "  Missing required keyword 'funcType'\n";
    return false;
  }

  if (!keyword.GetValue ("EwaldSplit", ewaldType))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp\n";
    cout << "  Missing required keyword 'EwaldSplit'\n";
    return false;
  }
  if (!keyword.GetValue ("Rdir", _Rdir))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp\n";
    cout << "  Missing required keyword 'Rdir'\n";
    return false;
  }
  if (!keyword.GetValue ("nDeriv", _Nder))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp\n";
    cout << "  Missing required keyword 'nDeriv'\n";
    return false;
  }

  double k0_r, k0_i;
  if (!keyword.GetValue ("k0_r", k0_r))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp\n";
    cout << "  Missing required keyword 'k0_r'\n";
    return false;
  }
  if (!keyword.GetValue ("k0_i", k0_i))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp\n";
    cout << "  Missing required keyword 'k0_i'\n";
    return false;
  }
  _k0 = k0_r + I*k0_i;

  if (funcType != "Helmholtz")
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp\n";
    char str[2000];
    sprintf(str, "  funcType = %s != 'Helmholtz'\n", funcType.c_str());
    cout << str;
    return false;
  }
  if (ewaldType != "DerivMatch")
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp\n";
    char str[2000];
    sprintf(str, "  ewaldType = %s != 'DerivMatch'\n", ewaldType.c_str());
    cout << str;
    return false;
  }


  std::vector<double> coordPermute;
  if (!SetUpBase (keyword, nCharge, coord, coordPermute, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp.\n";
    cout << "  SetUpBase failed\n";
    return false;
  }

  _Complex double fHelm[NPME_MaxDerivMatchOrder+1];
  NPME_FunctionDerivMatch_HelmholtzRadialDeriv (&fHelm[0], _Nder, _Rdir, _k0);
  if (!NPME_FunctionDerivMatch_CalcEvenSeries (&_a[0], &_b[0], 
    &fHelm[0], _Nder, _Rdir))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp.\n";
    cout << "  NPME_FunctionDerivMatch_CalcEvenSeries failed\n";
    return false;
  }

  if (!_func.SetParm (_k0))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp.\n";
    cout << "  _funcLR.SetParm failed\n";
    return false;
  }
  if (!_funcLR.SetParm (_k0, _Nder, _Rdir, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp.\n";
    cout << "  _funcLR.SetParm failed\n";
    return false;
  }
  if (!_funcSR.SetParm (_k0, _Nder, _Rdir, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp.\n";
    cout << "  _funcSR.SetParm failed\n";
    return false;
  }

  if (!_recSum.SetUp (nCharge, &coordPermute[0], _BnOrder, _useCompactF,
                _N1,    _N2,    _N3,
                _n1,    _n2,    _n3,
                _del1,  _del2,  _del3,
                _a1,    _a2,    _a3,
                &_funcLR, _nProc, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::SetUp.\n";
    cout << "  recSum.SetUp (..) failed\n";
    return false;
  }

  //self interaction smooth kernel _fself = funcLR(0,0,0)
  {
    double f0_r_x = 0;
    double f0_i   = 0;
    double y      = 0;
    double z      = 0;
    _funcLR.Calc (1, &f0_r_x, &f0_i, &y, &z);
    _fself = f0_r_x + I*f0_i;
  }


  return true;
}

bool NPME_InterfaceComplex_Helmholtz_DM::ResetCoord (
  const size_t nCharge, const double *coord, 
  bool printLog, std::ostream& ofs_log)
{
  using std::cout;
  using std::string;

  std::vector<double> coordPermute;

  if (!ResetCoordBase (nCharge, coord, coordPermute, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::ResetCoord\n";
    cout << "ResetCoordBase failed\n";
    return false;
  }

  if (!_recSum.SetUp (nCharge, &coordPermute[0], _BnOrder, _useCompactF,
                _N1,    _N2,    _N3,
                _n1,    _n2,    _n3,
                _del1,  _del2,  _del3,
                _a1,    _a2,    _a3,
                &_funcLR, _nProc, printLog, ofs_log))
  {
    cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::ResetCoord.\n";
    cout << "  recSum.SetUp (..) failed\n";
    return false;
  }

  return true;
}

void NPME_InterfaceComplex_Helmholtz_DM::Print (std::ostream& ofs_log)
{
  ofs_log << "\n\nNPME_InterfaceComplex_Helmholtz_DM::Print\n";
  ofs_log << "  fself = " << _fself << "\n";
  
  char str[2000];
  sprintf(str, "  Nder  = %d\n", _Nder);   ofs_log << str;
  sprintf(str, "  Rdir  = %f\n", _Rdir);   ofs_log << str;

  ofs_log << "\n";
  for (int i = 0; i <= _Nder; i++)
  {
    sprintf(str, "      a[%4d] = %15.6le + %15.6lei\n", 
      i, creal(_a[i]), cimag(_a[i]));
    ofs_log << str;
  }
  ofs_log << "\n";
  for (int i = 0; i <= _Nder; i++)
  {
    sprintf(str, "      b[%4d] = %15.6le + %15.6lei\n", 
      i, creal(_b[i]), cimag(_b[i]));
    ofs_log << str;
  }

  PrintBase (ofs_log);
  _recSum.Print (ofs_log);
}


bool NPME_InterfaceComplex_Helmholtz_DM::CalcV (
              _Complex double *V1, const size_t nCharge, 
              _Complex double *charge, double *coord, 
              bool printLog, std::ostream& ofs_log)
//PME V1 functions for real charges
  //input:  charge[nCharge] (real)
  //output: V1[nCharge][4]
{
  double time0, time;
  double timeStart, timeTot;
  char str[2000];
  bool zeroVarray = 0;
  bool PRINT_ALL = 0;

  timeStart = NPME_GetTime ();
  if (printLog)
  {
    ofs_log << "\n\nNPME_InterfaceComplex_Helmholtz_DM::CalcV\n";
    ofs_log.flush();
  }

  //1) permute charge[_nCharge]
  time0 = NPME_GetTime ();
  NPME_Kcycle_InversePermuteArray_MN (2, _nCharge, _kcycle, 
    (double *) charge, _nProc);
  NPME_Kcycle_InversePermuteArray_MN (3, _nCharge, _kcycle, 
    coord, _nProc);
  time = NPME_GetTime () - time0;
  if (printLog)
  {
    sprintf(str, " time = %6.2le (permute charge, coord)\n", time);
    ofs_log << str;
    ofs_log.flush();
  }

  //2) direct sum + self
  {
    time0 = NPME_GetTime ();
    //direct
    NPME_PotHelmholtz_SR_DM_DirectSum_V1 (_k0,
      _Nder, _a, _b, _Rdir,
      _nCharge, coord,
      charge, V1, _nProc, _vecOption, _cluster);

    //Vself
    for (size_t i = 0; i < _nCharge; i++)
      V1[4*i] -= _fself*charge[i];
    time = NPME_GetTime () - time0;
    if (printLog)
    {
      sprintf(str, " time = %6.2le (direct sum)\n", time);
      ofs_log << str;
      ofs_log.flush();
    }
  }


  //3) rec sum
  time0 = NPME_GetTime ();
  _recSum.CalcV1 (V1, nCharge, charge, coord, zeroVarray, PRINT_ALL, ofs_log);
  time = NPME_GetTime () - time0;
  if (printLog)
  {
    sprintf(str, " time = %6.2le (rec sum)\n", time);
    ofs_log << str;
    ofs_log.flush();
  }

  //4) permute V1 to original frame 
  NPME_Kcycle_PermuteArray_MN (8, _nCharge, _kcycle, (double *) V1, _nProc);

  //5) permute charge[_nCharge] back to original frame 
  NPME_Kcycle_PermuteArray_MN (2, _nCharge, _kcycle, (double *) charge, _nProc);
  NPME_Kcycle_PermuteArray_MN (3, _nCharge, _kcycle, coord,  _nProc);

  timeTot = NPME_GetTime () - timeStart;
  if (printLog)
  {
    sprintf(str, "--------------------------------------\n");
    sprintf(str, " time = %6.2le (total V1)\n\n\n", timeTot);
    ofs_log << str;
    ofs_log.flush();
  }

  return true;
}





bool NPME_InterfaceComplex_Helmholtz_DM::CalcV_exact (
              _Complex double *V1, const size_t nCharge, 
              _Complex double *charge, double *coord, 
              bool printLog, std::ostream& ofs_log)
//exact V1 functions (brute force ~N^2 sum)
//input:  charge[nCharge] (real)
//output: V1[nCharge][4]
{
  if (nCharge != _nCharge)
  {
    char str[2000];
    std::cout << "Error in NPME_InterfaceComplex_Helmholtz_DM::CalcV_exact.\n";
    sprintf(str, "  nCharge = %lu != %lu\n", nCharge, _nCharge);
    std::cout << str;
    return false;
  }
  double time0 = NPME_GetTime ();
  NPME_PotHelmholtz_MacroSelf_V1 (_k0, _nCharge, coord, charge, 
              V1, _nProc, _vecOption);
  double time = NPME_GetTime () - time0;

  if (printLog)
  {
    char str[500];
    ofs_log << "\n\nNPME_InterfaceComplex_Helmholtz_DM::CalcV_exact\n";
    sprintf(str, "  time_exact = %.2f\n", time);
    ofs_log << str;
  }

  return 1;
}




}//end namespace NPME_Library





