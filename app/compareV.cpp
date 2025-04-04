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
using namespace NPME_Library;




int main (int argc, char *argv[])
{
  using std::string;
  using std::cout;
  using std::vector;

  if (argc != 3)
  {
    cout << "Error.\n";
    cout << argv[0] << " Vfile1 Vfile2\n";
    exit(0);
  }



  string file1(argv[1]);
  string file2(argv[2]);

  bool isChargeReal1;
  bool isChargeReal2;
  size_t nCharge1;
  size_t nCharge2;

  vector<double> Vr1;
  vector<double> Vr2;
  vector<_Complex double> Vc1;
  vector<_Complex double> Vc2;



  if (!NPME_ReadSingleBox_V (file1, isChargeReal1, nCharge1, 
                            Vr1, Vc1))
  {
    cout << "Error reading " << file1 << "\n";
    exit(0);
  }
  if (!NPME_ReadSingleBox_V (file2, isChargeReal2, nCharge2, 
                            Vr2, Vc2))
  {
    cout << "Error reading " << file2 << "\n";
    exit(0);
  }

  char str[2000];
  {
    sprintf(str, "Error reading %s and %s\n", file1.c_str(), file2.c_str());

    if (isChargeReal1 != isChargeReal2)
    {
      cout << str;
      cout << "  isChargeReal do not match\n";
      exit(0);
    }

    if (nCharge1 != nCharge2)
    {
      cout << str;
      cout << "  nCharge do not match\n";
      exit(0);
    }
  }


  double errorV, errordVdr;
  double V_mag, dVdr_mag;
  if (isChargeReal1)
  {
    NPME_CompareV1Arrays (errorV, errordVdr, 
      nCharge1, 
      "pme", &Vr1[0], 
      "ref", &Vr2[0], 0, cout);
    NPME_CalcV1AvgVecMag (V_mag, dVdr_mag, nCharge1, &Vr2[0]);
  }
  else
  {
    NPME_CompareV1Arrays (errorV, errordVdr, 
      nCharge1, 
      "pme", &Vc1[0], 
      "ref", &Vc2[0], 0, cout);
    NPME_CalcV1AvgVecMag (V_mag, dVdr_mag, nCharge1, &Vc2[0]);
  }



  sprintf(str, "error_V = %.3le error_dVrdr = %.3le (abs. error)\n", 
    errorV, errordVdr);
  cout << str;

  sprintf(str, "error_V = %.3le error_dVrdr = %.3le (rel. error)\n", 
    errorV/V_mag, errordVdr/dVdr_mag);
  cout << str;

  return 1;
}



