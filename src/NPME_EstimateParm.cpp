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
#include <cmath>
#include <cstdio>

#include <iostream> 
#include <vector>

#include "NPME_Constant.h"
#include "NPME_EstimateParm.h"
#include "NPME_SupportFunctions.h"
#include "NPME_RecSumSupportFunctions.h"

namespace NPME_Library
{

double NPME_EstimateParmCalcQsum (const size_t nCharge, 
  const double *charge)
{
  double Q = 0.0;
  for (size_t i = 0; i < nCharge; i++)
    Q += charge[i]*charge[i];
  return sqrt(Q);
}
double NPME_EstimateParmCalcQsum (const size_t nCharge, 
  const _Complex double *charge)
{
  double Q = 0.0;
  for (size_t i = 0; i < nCharge; i++)
    Q += creal( charge[i]*conj(charge[i]) );
  return sqrt(Q);
}

double NPME_EstimateParmSingleBox_Calc_Pu (const double *a, const double u)
{
  double y      = 0;
  double u_p    = 1;
  for (int p = 0; p <= 4; p++)
  {
    y   += a[p]*u_p;
    u_p *= u;
  }

  return y;
}


bool NPME_EstimateParmSingleBox_FindOptimal_u (double& u, 
  const double y0, const double a[5], const double uMin, const double uMax)
//simple grid search for y(u) ~ y0 for uMin <= u <= uMax
{
  const int NumTrial  = 1000;
  const double del    = (uMax - uMin)/(NumTrial-1);
  const double delTol = 0.1;

  double minDiff  = 1.0E6;
  double yPrev    = 1.0E6;
  for (int n = 0; n < NumTrial; n++)
  {
    double uTrial = uMin + n*del;

    double y      = 0;
    double u_p    = 1;
    for (int p = 0; p <= 4; p++)
    {
      y   += a[p]*u_p;
      u_p *= uTrial;
    }

    double diff = fabs(y - y0);
    if (minDiff > diff)
    {
      minDiff = diff;
      u       = uTrial;
    }
    if (minDiff < delTol)
      break;


    //y should be negative and decreasing.  if y starts increasing, stop
    if (y > yPrev)
      break;

    yPrev = y;
  }
    
  return true;
}







bool NPME_EstimateParmSingleBox_StaticModel::Get_FFT_outputNder (
  long int& N1, long int& N2, long int& N3, 
  long int& n1, long int& n2, long int& n3,
  int& Nder,
  const size_t nCharge, const double *charge, const double *coord, 
  const long int BsplineOrder, const double alpha, 
  const double tol, const double Rdir, bool printLog, std::ostream& os) const
{
  using std::cout;


  std::vector<size_t> parmIndexList;
  if (!GetParmIndexList (parmIndexList, BsplineOrder, alpha))
  {
    cout << "Error in NPME_EstimateParmSingleBox_StaticModel::Get_FFT_outputNder\n";
    cout << "  GetParmIndexList failed\n";
    return false;
  }


  long int N1_trial, N2_trial, N3_trial;
  long int n1_trial, n2_trial, n3_trial;
  if (!Get_FFT_inputParmIndex (
          N1_trial, N2_trial, N3_trial,
          n1_trial, n2_trial, n3_trial,
          nCharge, charge, coord,
          parmIndexList[0], tol, Rdir, printLog, os))
  {
    cout << "Error in NPME_EstimateParmSingleBox_StaticModel::Get_FFT_outputNder\n";
    cout << "  GetParmIndexList failed\n";
    return false;
  }

  size_t minGridSizeParmIndex = parmIndexList[0];
  long int minGridSize        = N1_trial*N2_trial*N3_trial;
  N1                          = N1_trial;
  N2                          = N2_trial;
  N3                          = N3_trial;
  n1                          = n1_trial;
  n2                          = n2_trial;
  n3                          = n3_trial;

  for (size_t n = 1; n < parmIndexList.size(); n++)
  {
    if (!Get_FFT_inputParmIndex (
            N1_trial, N2_trial, N3_trial,
            n1_trial, n2_trial, n3_trial,
            nCharge, charge, coord,
            parmIndexList[n], tol, Rdir, printLog, os))
    {
      cout << "Error in NPME_EstimateParmSingleBox_StaticModel::Get_FFT_outputNder\n";
      cout << "  Get_FFT_inputParmIndex failed\n";
      return false;
    }

    if (minGridSize > N1_trial*N2_trial*N3_trial)
    {
      minGridSizeParmIndex  = parmIndexList[n];
      minGridSize           = N1_trial*N2_trial*N3_trial;
      N1                    = N1_trial;
      N2                    = N2_trial;
      N3                    = N3_trial;
      n1                    = n1_trial;
      n2                    = n2_trial;
      n3                    = n3_trial;
    }
  }

  Nder = _parm[minGridSizeParmIndex].Nder;

  return true;
}



bool NPME_EstimateParmSingleBox_StaticModel::Get_FFT_inputNder (
  long int& N1, long int& N2, long int& N3, 
  long int& n1, long int& n2, long int& n3,
  const size_t nCharge, const double *charge, const double *coord, 
  const long int BsplineOrder, const double alpha, const int Nder, 
  const double tol, const double Rdir, bool printLog, std::ostream& os) const
{
  using std::cout;

  size_t parmIndex;
  if (!GetParmIndex (parmIndex, BsplineOrder, alpha, Nder))
  {
    cout << "Error in NPME_EstimateParmSingleBox_StaticModel::Get_FFT_inputNder\n";
    cout << "  GetParmIndex failed\n";
    return false;
  }

  return Get_FFT_inputParmIndex (
            N1, N2, N3, 
            n1, n2, n3,
            nCharge, charge, coord, parmIndex,
            tol, Rdir, printLog, os);
}

bool NPME_EstimateParmSingleBox_StaticModel::Get_FFT_inputParmIndex (
  long int& N1, long int& N2, long int& N3, 
  long int& n1, long int& n2, long int& n3,
  const size_t nCharge, const double *charge, const double *coord, 
  const size_t parmIndex,
  const double tol, const double Rdir, bool printLog, std::ostream& os) const
{
  const size_t MaxIteration = 10;
  char str[2000];

  //1) assume del = Rdir
  const double del = Rdir;

  
  //2) get model parameters
  const int Nder              = _parm[parmIndex].Nder;
  const double uMin           = _parm[parmIndex].uMin;
  const double uMax           = _parm[parmIndex].uMax;
  const long int BsplineOrder = _parm[parmIndex].BsplineOrder;
  const double *a             = _parm[parmIndex].a;
  const double alpha          = _parm[parmIndex].alpha;



  if (printLog)
  {
    os << "\n\nNPME_EstimateParmSingleBox_StaticModel::Get_FFT_inputParmIndex\n";
    sprintf(str, "   uMin          = %f\n", uMin);
    os << str;
    sprintf(str, "   uMax          = %f\n", uMax);
    os << str;

    sprintf(str, "   a             = %f %f %f %f %f", a[0], a[1], a[2], a[3], a[4]);
    os << str;

    sprintf(str, "   Rdir          = %f\n", Rdir);
    os << str;
    sprintf(str, "   alpha         = %f\n", alpha);
    os << str;
    sprintf(str, "   BsplineOrder  = %ld\n", BsplineOrder);
    os << str;
    sprintf(str, "   Nder          = %d\n", Nder);
    os << str;
    sprintf(str, "   tol           = %.2le\n", tol);
    os << str;
    os.flush();
  }

  //physical dimension
  double X0, Y0, Z0;
  double R0[3];
  NPME_CalcBoxDimensionCenter (nCharge, coord, X0, Y0, Z0, R0);
  const double Q0 = NPME_EstimateParmCalcQsum (nCharge, charge);

  if (printLog)
  {
    sprintf(str, "   X0 Y0 Z0      = %10.6f %10.6f %10.6f\n", X0, Y0, Z0);
    os << str;
    sprintf(str, "   Q0            = %10.6f\n", Q0);
    os << str;
    os << "\n\n";
    os.flush();
  }

  N1 = 100;
  N2 = 100;
  N3 = 100;

  double eps_V_pred = 0;
  for (size_t n = 0; n < MaxIteration; n++)
  {
    if (printLog)
    {
      sprintf(str, "Iteration %lu\n", n);
      os << str;
    }
    double X, Y, Z;
    double R[3];
    if (n == 0)
    {
      X = X0;
      Y = Y0;
      Z = Z0;
    }
    else
    {
      NPME_RecSumInterface_GetPMECorrBox (
        X,  Y,  Z,  R, 
        X0, Y0, Z0, R0,
        N1, N2, N3,
        del, BsplineOrder);
    }
    double L1, L2, L3;
    L1 = 2*(X + del);
    L2 = 2*(Y + del);
    L3 = 2*(Z + del);

    const double Q1 = Q0*sqrt(L1*L1*L1/(L1*L2*L3));
    const double Q2 = Q0*sqrt(L2*L2*L2/(L1*L2*L3));
    const double Q3 = Q0*sqrt(L3*L3*L3/(L1*L2*L3));

    if (printLog)
    {
      sprintf(str, "   X  Y  Z       = %10.6f %10.6f %10.6f\n", X,  Y,  Z);
      os << str;
      sprintf(str, "   L1 L2 L3      = %10.6f %10.6f %10.6f\n", L1, L2, L3);
      os << str;
      sprintf(str, "   Q1 Q2 Q3      = %10.6f %10.6f %10.6f\n", Q1, Q2, Q3);
      os << str;
      os.flush();
    }

    //5) log(theta) reference values
    //for a cubic system,
    //  tol = eps_V 
    //      = Q*exp(P(u))*Rdir^alpha*(Rdir/L)^1.5
    //where
    //  u   = N1*Rdir/L
    //for a rectangular system
    //  tol         = Q1*exp(P(u1))*Rdir^alpha*(Rdir/L1)^1.5
    //  exp(P(u1))  = tol/Q1*Rdir^(-alpha)*(L1/Rdir)^1.5
    //  P(u1)       = log(tol*Rdir^(-alpha)/Q1*(L1/Rdir)^1.5)
    //  P(u1)       = log(t1)
    //  t1          = tol*Rdir^(-alpha)/Q1*(L1/Rdir)^1.5

    //eps_theta1 = tol
    double y1, y2, y3;
    {
      const double C  = tol*pow(Rdir, -alpha);
      const double t1 = C/Q1*pow(L1/Rdir, 1.5);
      const double t2 = C/Q2*pow(L2/Rdir, 1.5);
      const double t3 = C/Q3*pow(L3/Rdir, 1.5);

      y1 = log(t1);
      y2 = log(t2);
      y3 = log(t3);

      if (printLog)
      {
        sprintf(str, "   t1 t2 t3      = %.2le %.2le %.2le\n", t1, t2, t3);
        os << str;
        sprintf(str, "   y1 y2 y3      = %10.6f %10.6f %10.6f\n", y1, y2, y3);
        os << str;
        os.flush();
      }
    }

    double u1, u2, u3;
    if (!NPME_EstimateParmSingleBox_FindOptimal_u (u1, y1, 
      a, uMin, uMax))
    {
      std::cout << "Error in NPME_EstimateParmSingleBox_StaticModel::Get_FFT\n";
      std::cout << "NPME_EstimateParmSingleBox_FindOptimal_u failed\n";
      return false;
    }
    if (!NPME_EstimateParmSingleBox_FindOptimal_u (u2, y2, 
      a, uMin, uMax))
    {
      std::cout << "Error in NPME_EstimateParmSingleBox_StaticModel::Get_FFT\n";
      std::cout << "NPME_EstimateParmSingleBox_FindOptimal_u failed\n";
      return false;
    }
    if (!NPME_EstimateParmSingleBox_FindOptimal_u (u3, y3, 
      a, uMin, uMax))
    {
      std::cout << "Error in NPME_EstimateParmSingleBox_StaticModel::Get_FFT\n";
      std::cout << "NPME_EstimateParmSingleBox_FindOptimal_u failed\n";
      return false;
    }

    //u = N*Rdir/L
    N1 = (long int) (u1*L1/Rdir) + 1;
    N2 = (long int) (u2*L2/Rdir) + 1;
    N3 = (long int) (u3*L3/Rdir) + 1;

    if (N1 < 4*BsplineOrder + 1)  N1 = 4*BsplineOrder + 1;
    if (N2 < 4*BsplineOrder + 1)  N2 = 4*BsplineOrder + 1;
    if (N3 < 4*BsplineOrder + 1)  N3 = 4*BsplineOrder + 1;

    if (printLog)
    {
      sprintf(str, "   u1 u2 u3      = %10.6f %10.6f %10.6f\n", u1, u2, u3);
      os << str;
      sprintf(str, "   N1 N2 N3      = %10ld %10ld %10ld (Before)\n", 
        N1, N2, N3);
      os << str;
      os.flush();
    }

  }

  NPME_FindFFTSizeBlockSize (N1, n1, N1, NPME_IdealRecSumBlockSize);
  NPME_FindFFTSizeBlockSize (N2, n2, N2, NPME_IdealRecSumBlockSize);
  NPME_FindFFTSizeBlockSize (N3, n3, N3, NPME_IdealRecSumBlockSize);

  if (printLog)
  {
    os << "Final FFT + FFT Block Sizes\n";
    sprintf(str, "   N1 N2 N3      = %10ld %10ld %10ld\n", N1, N2, N3);
    os << str;
    sprintf(str, "   n1 n2 n3      = %10ld %10ld %10ld\n", n1, n2, n3);
    os << str;
    os << "\n";
    os.flush();
  }


  return true;
}



bool NPME_EstimateParmSingleBox_StaticModel::CalcPredError_V (
    double& eps_V, bool& reliablePred,
    const long int N1, const long int N2, const long int N3, 
    const size_t nCharge, const double *charge, const double *coord, 
    const long int BsplineOrder, const double alpha, const int Nder, 
    const double Rdir, bool printLog, std::ostream& os)
{
  using std::cout;

  size_t parmIndex;
  if (!GetParmIndex (parmIndex, BsplineOrder, alpha, Nder))
  {
    cout << "Error in NPME_EstimateParmSingleBox_StaticModel::CalcPredError_V\n";
    cout << "  GetParmIndex failed\n";
    return false;
  }


  char str[2000];

  //1) assume del = Rdir
  const double del = Rdir;

  //2) get model parameters
  const double uMin           = _parm[parmIndex].uMin;
  const double uMax           = _parm[parmIndex].uMax;
  const double *a             = _parm[parmIndex].a;

  if (printLog)
  {
    os << "\n\nNPME_EstimateParmSingleBox_StaticModel::CalcPredError_V\n";
    sprintf(str, "   uMin          = %f\n", uMin);
    os << str;
    sprintf(str, "   uMax          = %f\n", uMax);
    os << str;

    sprintf(str, "   a             = %f %f %f %f %f", a[0], a[1], a[2], a[3], a[4]);
    os << str;



    sprintf(str, "   Rdir          = %f\n", Rdir);
    os << str;
    sprintf(str, "   alpha         = %f\n", alpha);
    os << str;
    sprintf(str, "   BsplineOrder  = %ld\n", BsplineOrder);
    os << str;
    sprintf(str, "   Nder          = %d\n", Nder);
    os << str;
    os.flush();
  }

  //physical dimension
  double X0, Y0, Z0;
  double R0[3];
  NPME_CalcBoxDimensionCenter (nCharge, coord, X0, Y0, Z0, R0);
  const double Q0 = NPME_EstimateParmCalcQsum (nCharge, charge);

  if (printLog)
  {
    sprintf(str, "   X0 Y0 Z0      = %10.6f %10.6f %10.6f\n", X0, Y0, Z0);
    os << str;
    sprintf(str, "   Q0            = %10.6f\n", Q0);
    os << str;
    os << "\n\n";
    os.flush();
  }

  double X, Y, Z;
  double R[3];
  NPME_RecSumInterface_GetPMECorrBox (
    X,  Y,  Z,  R, 
    X0, Y0, Z0, R0,
    N1, N2, N3,
    del, BsplineOrder);

  double L1, L2, L3;
  L1 = 2*(X + del);
  L2 = 2*(Y + del);
  L3 = 2*(Z + del);


  const double u1 = N1*Rdir/L1;
  const double u2 = N2*Rdir/L2;
  const double u3 = N3*Rdir/L3;

  //prediction is reliable if uMin <= u1,u2,u3 <= uMax
  reliablePred = 1;
  if ((u1 < uMin) || (u1 > uMax)) reliablePred = 0;
  if ((u2 < uMin) || (u2 > uMax)) reliablePred = 0;
  if ((u3 < uMin) || (u3 > uMax)) reliablePred = 0;

  const double Q1     = Q0*sqrt(L1*L1*L1/(L1*L2*L3));
  const double Q2     = Q0*sqrt(L2*L2*L2/(L1*L2*L3));
  const double Q3     = Q0*sqrt(L3*L3*L3/(L1*L2*L3));

  const double P1     = NPME_EstimateParmSingleBox_Calc_Pu (a, u1);
  const double P2     = NPME_EstimateParmSingleBox_Calc_Pu (a, u2);
  const double P3     = NPME_EstimateParmSingleBox_Calc_Pu (a, u3);

  const double eps_V1 = Q1*pow(Rdir, alpha)*pow(Rdir/L1, 1.5)*exp(P1);
  const double eps_V2 = Q2*pow(Rdir, alpha)*pow(Rdir/L2, 1.5)*exp(P2);
  const double eps_V3 = Q3*pow(Rdir, alpha)*pow(Rdir/L3, 1.5)*exp(P3);

  eps_V  = NPME_Max (eps_V1, eps_V2, eps_V3);

  if (printLog)
  {
    sprintf(str, "   X  Y  Z              = %10.6f %10.6f %10.6f\n", 
      X,  Y,  Z);
    os << str;
    sprintf(str, "   L1 L2 L3             = %10.6f %10.6f %10.6f\n", 
      L1, L2, L3);
    os << str;
    sprintf(str, "   Q1 Q2 Q3             = %10.6f %10.6f %10.6f\n", 
      Q1, Q2, Q3);
    sprintf(str, "   N1 N2 N3             = %10ld %10ld %10ld\n", 
      N1, N2, N3);
    os << str;
    sprintf(str, "   u1 u2 u3             = %10.6f %10.6f %10.6f\n", 
      u1, u2, u3);
    os << str;
    sprintf(str, "   P1 P2 P3             = %10.6f %10.6f %10.6f\n", 
      P1, P2, P3);
    os << str;
    sprintf(str, "   eps_V1 eps_V1 eps_V1 = %.4le %.4le %.4le\n", 
      eps_V1, eps_V2, eps_V3);
    os << str;
    sprintf(str, "   reliablePred         = %d\n", (int) reliablePred);
    os << str;
    sprintf(str, "   eps_V                = %.4le\n", eps_V);
    os << str;
    os.flush();
  }


  return true;
}





bool NPME_EstimateParmSingleBox_StaticModel::GetParmIndex (size_t& parmIndex,
  const long int BsplineOrder, const double alpha, const int Nder) const
//finds parmIndex which haa the correct BsplineOrder, alpha, 
{
  using std::cout;

  for (size_t n = 0; n < _parm.size(); n++)
    if (BsplineOrder == _parm[n].BsplineOrder)
      if (Nder == _parm[n].Nder)
        if (fabs(alpha - _parm[n].alpha) < 1.0E-6)
        {
          parmIndex = n;
          return true;
        }

  char str[2000];
  cout << "NPME_EstimateParmSingleBox_StaticModel::GetParmIndex\n";
  sprintf(str, "parameters not defined for BsplineOrder = %ld alpha = %f Nder = %d\n",
    BsplineOrder, alpha, Nder);
  cout << str;
    
  return false;
}

bool NPME_EstimateParmSingleBox_StaticModel::GetParmIndexList (
  std::vector<size_t>& parmIndexList,
  const long int BsplineOrder, const double alpha) const
//finds list of parmIndexes which have the correct BsplineOrder and alpha
{
  parmIndexList.clear();
  for (size_t n = 0; n < _parm.size(); n++)
    if (BsplineOrder == _parm[n].BsplineOrder)
      if (fabs(alpha - _parm[n].alpha) < 1.0E-6)
      {
        parmIndexList.push_back(n);
      }

  if (parmIndexList.size() == 0)
  {
    using std::cout;
    cout << "Error in NPME_EstimateParmSingleBox_StaticModel::GetParmIndexList\n";
    cout << "  parmIndexList is empty.\n";
    char str[500];
    sprintf(str, "parameters not available for BsplineOrder = %ld and alpha = %f\n",
      BsplineOrder, alpha);
    return false;
  }

  return true;
}


bool NPME_EstimateParmSingleBox_StaticModel::GetParm (double& uMin, 
  double& uMax, double a[5], const long int BsplineOrder, 
  const double alpha, const int Nder) const
{
  for (size_t n = 0; n < _parm.size(); n++)
    if (BsplineOrder == _parm[n].BsplineOrder)
      if (Nder  == _parm[n].Nder)
        if (fabs(alpha - _parm[n].alpha) < 1.0E-6)
        {
          uMin = _parm[n].uMin;
          uMax = _parm[n].uMax;
          
          for (int p = 0; p <= 4; p++)
            a[p] = _parm[n].a[p];

          return true;
        }

  return false;
}



bool NPME_EstimateParmSingleBox_StaticModel::SetParm ()
{
  if (_isSet)
    return true;

  _isSet = 1;
  _parm.clear();

  NPME_ThetaParmStatic parmTmp;


  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          2;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -1.85283618e+00;
  parmTmp.a[1]         = -1.02692894e+00;
  parmTmp.a[2]         =  4.95827300e-02;
  parmTmp.a[3]         = -1.32807727e-03;
  parmTmp.a[4]         =  1.42669856e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          3;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -1.32334963e+00;
  parmTmp.a[1]         = -1.25739868e+00;
  parmTmp.a[2]         =  6.21842186e-02;
  parmTmp.a[3]         = -1.70439439e-03;
  parmTmp.a[4]         =  1.87090674e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -1.04843164e+00;
  parmTmp.a[1]         = -1.70796463e+00;
  parmTmp.a[2]         =  8.75293980e-02;
  parmTmp.a[3]         = -2.44163816e-03;
  parmTmp.a[4]         =  2.68734494e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  2.15308395e-01;
  parmTmp.a[1]         = -2.10343797e+00;
  parmTmp.a[2]         =  1.12079820e-01;
  parmTmp.a[3]         = -3.21679516e-03;
  parmTmp.a[4]         =  3.61247628e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  1.09886288e+00;
  parmTmp.a[1]         = -2.32041604e+00;
  parmTmp.a[2]         =  1.26814650e-01;
  parmTmp.a[3]         = -3.67310932e-03;
  parmTmp.a[4]         =  4.15025291e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -9.68745679e-01;
  parmTmp.a[1]         = -1.77340887e+00;
  parmTmp.a[2]         =  9.30866242e-02;
  parmTmp.a[3]         = -2.63920360e-03;
  parmTmp.a[4]         =  2.93862092e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  1.97831612e-01;
  parmTmp.a[1]         = -2.20261892e+00;
  parmTmp.a[2]         =  1.20872664e-01;
  parmTmp.a[3]         = -3.54629667e-03;
  parmTmp.a[4]         =  4.04480830e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  9.41489678e-01;
  parmTmp.a[1]         = -2.43607532e+00;
  parmTmp.a[2]         =  1.29333173e-01;
  parmTmp.a[3]         = -3.69740025e-03;
  parmTmp.a[4]         =  4.13412068e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          7;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  2.52323043e+00;
  parmTmp.a[1]         = -2.85737809e+00;
  parmTmp.a[2]         =  1.56006449e-01;
  parmTmp.a[3]         = -4.55450221e-03;
  parmTmp.a[4]         =  5.16908542e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  4.00003378e+00;
  parmTmp.a[1]         = -3.18360838e+00;
  parmTmp.a[2]         =  1.74241076e-01;
  parmTmp.a[3]         = -5.07171816e-03;
  parmTmp.a[4]         =  5.74688221e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         = -2.35331214e-01;
  parmTmp.a[1]         = -1.98988808e+00;
  parmTmp.a[2]         =  1.13634265e-01;
  parmTmp.a[3]         = -3.43725152e-03;
  parmTmp.a[4]         =  4.01590329e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.65742715e+00;
  parmTmp.a[1]         = -2.70053798e+00;
  parmTmp.a[2]         =  1.54895967e-01;
  parmTmp.a[3]         = -4.70548399e-03;
  parmTmp.a[4]         =  5.51238344e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  4.66697773e+00;
  parmTmp.a[1]         = -3.52900454e+00;
  parmTmp.a[2]         =  2.06610381e-01;
  parmTmp.a[3]         = -6.35548929e-03;
  parmTmp.a[4]         =  7.49863366e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         10;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  7.55789052e+00;
  parmTmp.a[1]         = -4.16855463e+00;
  parmTmp.a[2]         =  2.39744561e-01;
  parmTmp.a[3]         = -7.27483136e-03;
  parmTmp.a[4]         =  8.49477039e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         12;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.05372690e+01;
  parmTmp.a[1]         = -4.76600124e+00;
  parmTmp.a[2]         =  2.76183197e-01;
  parmTmp.a[3]         = -8.71540783e-03;
  parmTmp.a[4]         =  1.09080621e-04;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         14;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.00684113e+01;
  parmTmp.a[1]         = -4.06328117e+00;
  parmTmp.a[2]         =  1.66763552e-01;
  parmTmp.a[3]         = -3.75622758e-03;
  parmTmp.a[4]         =  4.10960701e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         16;
  parmTmp.alpha        = -1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  8.08272834e+00;
  parmTmp.a[1]         = -2.77369556e+00;
  parmTmp.a[2]         = -1.45573769e-02;
  parmTmp.a[3]         =  4.93545584e-03;
  parmTmp.a[4]         = -9.04536685e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          2;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -7.72809589e-01;
  parmTmp.a[1]         = -1.01155412e+00;
  parmTmp.a[2]         =  4.84312437e-02;
  parmTmp.a[3]         = -1.29004932e-03;
  parmTmp.a[4]         =  1.38097364e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          3;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -1.17965231e-01;
  parmTmp.a[1]         = -1.25761956e+00;
  parmTmp.a[2]         =  6.21030956e-02;
  parmTmp.a[3]         = -1.69898745e-03;
  parmTmp.a[4]         =  1.86067248e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  2.72991241e-01;
  parmTmp.a[1]         = -1.69162636e+00;
  parmTmp.a[2]         =  8.62540534e-02;
  parmTmp.a[3]         = -2.39846172e-03;
  parmTmp.a[4]         =  2.63456343e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  1.61596071e+00;
  parmTmp.a[1]         = -2.09000013e+00;
  parmTmp.a[2]         =  1.11097769e-01;
  parmTmp.a[3]         = -3.18523661e-03;
  parmTmp.a[4]         =  3.57483866e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  2.56956668e+00;
  parmTmp.a[1]         = -2.31126638e+00;
  parmTmp.a[2]         =  1.25591133e-01;
  parmTmp.a[3]         = -3.62300507e-03;
  parmTmp.a[4]         =  4.08282975e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  3.83856942e-01;
  parmTmp.a[1]         = -1.76441944e+00;
  parmTmp.a[2]         =  9.24479776e-02;
  parmTmp.a[3]         = -2.61917883e-03;
  parmTmp.a[4]         =  2.91562007e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  1.60116417e+00;
  parmTmp.a[1]         = -2.18505509e+00;
  parmTmp.a[2]         =  1.19505131e-01;
  parmTmp.a[3]         = -3.50014560e-03;
  parmTmp.a[4]         =  3.98844814e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  2.39973237e+00;
  parmTmp.a[1]         = -2.41425696e+00;
  parmTmp.a[2]         =  1.27609634e-01;
  parmTmp.a[3]         = -3.63868969e-03;
  parmTmp.a[4]         =  4.06212217e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          7;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  4.05527199e+00;
  parmTmp.a[1]         = -2.84031991e+00;
  parmTmp.a[2]         =  1.54791464e-01;
  parmTmp.a[3]         = -4.51608328e-03;
  parmTmp.a[4]         =  5.12435173e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  5.56501238e+00;
  parmTmp.a[1]         = -3.16223382e+00;
  parmTmp.a[2]         =  1.72541830e-01;
  parmTmp.a[3]         = -5.01640065e-03;
  parmTmp.a[4]         =  5.68074652e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.10910385e+00;
  parmTmp.a[1]         = -1.97763554e+00;
  parmTmp.a[2]         =  1.12642640e-01;
  parmTmp.a[3]         = -3.40266280e-03;
  parmTmp.a[4]         =  3.97253995e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  3.11842745e+00;
  parmTmp.a[1]         = -2.67756947e+00;
  parmTmp.a[2]         =  1.52976346e-01;
  parmTmp.a[3]         = -4.63724219e-03;
  parmTmp.a[4]         =  5.42583714e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  6.23826179e+00;
  parmTmp.a[1]         = -3.50526223e+00;
  parmTmp.a[2]         =  2.04732619e-01;
  parmTmp.a[3]         = -6.29147049e-03;
  parmTmp.a[4]         =  7.42001124e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         10;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  9.19417453e+00;
  parmTmp.a[1]         = -4.13695548e+00;
  parmTmp.a[2]         =  2.37182846e-01;
  parmTmp.a[3]         = -7.18597878e-03;
  parmTmp.a[4]         =  8.38458313e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         12;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.21897107e+01;
  parmTmp.a[1]         = -4.71206401e+00;
  parmTmp.a[2]         =  2.70670554e-01;
  parmTmp.a[3]         = -8.45860543e-03;
  parmTmp.a[4]         =  1.04583609e-04;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         14;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.19218504e+01;
  parmTmp.a[1]         = -4.07306991e+00;
  parmTmp.a[2]         =  1.70446732e-01;
  parmTmp.a[3]         = -3.99601737e-03;
  parmTmp.a[4]         =  4.51982209e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         16;
  parmTmp.alpha        = -2.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  9.51220099e+00;
  parmTmp.a[1]         = -2.56711505e+00;
  parmTmp.a[2]         = -4.20378901e-02;
  parmTmp.a[3]         =  6.37346985e-03;
  parmTmp.a[4]         = -1.13999828e-04;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          2;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -7.04230071e-02;
  parmTmp.a[1]         = -9.96696153e-01;
  parmTmp.a[2]         =  4.73188940e-02;
  parmTmp.a[3]         = -1.25328274e-03;
  parmTmp.a[4]         =  1.33670093e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          3;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  6.96732456e-01;
  parmTmp.a[1]         = -1.25470910e+00;
  parmTmp.a[2]         =  6.18184974e-02;
  parmTmp.a[3]         = -1.68779642e-03;
  parmTmp.a[4]         =  1.84463727e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  1.18855546e+00;
  parmTmp.a[1]         = -1.67575288e+00;
  parmTmp.a[2]         =  8.50345989e-02;
  parmTmp.a[3]         = -2.35764707e-03;
  parmTmp.a[4]         =  2.58512055e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  2.60288620e+00;
  parmTmp.a[1]         = -2.07609915e+00;
  parmTmp.a[2]         =  1.10097911e-01;
  parmTmp.a[3]         = -3.15338815e-03;
  parmTmp.a[4]         =  3.53714484e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  3.61839384e+00;
  parmTmp.a[1]         = -2.29956416e+00;
  parmTmp.a[2]         =  1.24240719e-01;
  parmTmp.a[3]         = -3.57023863e-03;
  parmTmp.a[4]         =  4.01344887e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  1.32540234e+00;
  parmTmp.a[1]         = -1.75459827e+00;
  parmTmp.a[2]         =  9.17513654e-02;
  parmTmp.a[3]         = -2.59732762e-03;
  parmTmp.a[4]         =  2.89047976e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  2.58840117e+00;
  parmTmp.a[1]         = -2.16707040e+00;
  parmTmp.a[2]         =  1.18111605e-01;
  parmTmp.a[3]         = -3.45327863e-03;
  parmTmp.a[4]         =  3.93138836e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  3.44241057e+00;
  parmTmp.a[1]         = -2.39347356e+00;
  parmTmp.a[2]         =  1.25988599e-01;
  parmTmp.a[3]         = -3.58392815e-03;
  parmTmp.a[4]         =  3.99537275e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          7;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  5.16705877e+00;
  parmTmp.a[1]         = -2.82391065e+00;
  parmTmp.a[2]         =  1.53637617e-01;
  parmTmp.a[3]         = -4.47987564e-03;
  parmTmp.a[4]         =  5.08236656e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  6.70863668e+00;
  parmTmp.a[1]         = -3.14177099e+00;
  parmTmp.a[2]         =  1.70948452e-01;
  parmTmp.a[3]         = -4.96485034e-03;
  parmTmp.a[4]         =  5.61926134e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  2.04327283e+00;
  parmTmp.a[1]         = -1.96474064e+00;
  parmTmp.a[2]         =  1.11609596e-01;
  parmTmp.a[3]         = -3.36684059e-03;
  parmTmp.a[4]         =  3.92779781e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  4.16217020e+00;
  parmTmp.a[1]         = -2.65523848e+00;
  parmTmp.a[2]         =  1.51125573e-01;
  parmTmp.a[3]         = -4.57170599e-03;
  parmTmp.a[4]         =  5.34287998e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  7.38641752e+00;
  parmTmp.a[1]         = -3.48238502e+00;
  parmTmp.a[2]         =  2.02942810e-01;
  parmTmp.a[3]         = -6.23089573e-03;
  parmTmp.a[4]         =  7.34600902e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         10;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.04044626e+01;
  parmTmp.a[1]         = -4.10601297e+00;
  parmTmp.a[2]         =  2.34610741e-01;
  parmTmp.a[3]         = -7.09295089e-03;
  parmTmp.a[4]         =  8.26197071e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         12;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.34262676e+01;
  parmTmp.a[1]         = -4.66560983e+00;
  parmTmp.a[2]         =  2.66366436e-01;
  parmTmp.a[3]         = -8.28221212e-03;
  parmTmp.a[4]         =  1.01912695e-04;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         14;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.31829366e+01;
  parmTmp.a[1]         = -4.01538182e+00;
  parmTmp.a[2]         =  1.65092500e-01;
  parmTmp.a[3]         = -3.78204449e-03;
  parmTmp.a[4]         =  4.21950908e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         16;
  parmTmp.alpha        = -3.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.09594765e+01;
  parmTmp.a[1]         = -2.60312446e+00;
  parmTmp.a[2]         = -2.97008201e-02;
  parmTmp.a[3]         =  5.46459070e-03;
  parmTmp.a[4]         = -9.68321619e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          2;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  4.54541968e-01;
  parmTmp.a[1]         = -9.82384113e-01;
  parmTmp.a[2]         =  4.62592324e-02;
  parmTmp.a[3]         = -1.21856365e-03;
  parmTmp.a[4]         =  1.29520236e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          3;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  1.32265167e+00;
  parmTmp.a[1]         = -1.24982147e+00;
  parmTmp.a[2]         =  6.14173833e-02;
  parmTmp.a[3]         = -1.67369559e-03;
  parmTmp.a[4]         =  1.82623179e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  1.90479155e+00;
  parmTmp.a[1]         = -1.66041512e+00;
  parmTmp.a[2]         =  8.38750914e-02;
  parmTmp.a[3]         = -2.31922714e-03;
  parmTmp.a[4]         =  2.53888136e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  3.38427726e+00;
  parmTmp.a[1]         = -2.06212292e+00;
  parmTmp.a[2]         =  1.09109401e-01;
  parmTmp.a[3]         = -3.12227702e-03;
  parmTmp.a[4]         =  3.50072239e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  4.45594651e+00;
  parmTmp.a[1]         = -2.28652227e+00;
  parmTmp.a[2]         =  1.22854272e-01;
  parmTmp.a[3]         = -3.51779783e-03;
  parmTmp.a[4]         =  3.94569592e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  2.06378152e+00;
  parmTmp.a[1]         = -1.74435984e+00;
  parmTmp.a[2]         =  9.10343067e-02;
  parmTmp.a[3]         = -2.57509749e-03;
  parmTmp.a[4]         =  2.86520639e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  3.36827870e+00;
  parmTmp.a[1]         = -2.14904119e+00;
  parmTmp.a[2]         =  1.16724410e-01;
  parmTmp.a[3]         = -3.40677567e-03;
  parmTmp.a[4]         =  3.87482544e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  4.27798699e+00;
  parmTmp.a[1]         = -2.37388105e+00;
  parmTmp.a[2]         =  1.24483782e-01;
  parmTmp.a[3]         = -3.53354337e-03;
  parmTmp.a[4]         =  3.93428130e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          7;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  6.06681316e+00;
  parmTmp.a[1]         = -2.80814928e+00;
  parmTmp.a[2]         =  1.52548242e-01;
  parmTmp.a[3]         = -4.44615732e-03;
  parmTmp.a[4]         =  5.04370918e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  7.63974630e+00;
  parmTmp.a[1]         = -3.12228713e+00;
  parmTmp.a[2]         =  1.69461663e-01;
  parmTmp.a[3]         = -4.91715347e-03;
  parmTmp.a[4]         =  5.56264430e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  2.77431582e+00;
  parmTmp.a[1]         = -1.95138275e+00;
  parmTmp.a[2]         =  1.10547929e-01;
  parmTmp.a[3]         = -3.33018051e-03;
  parmTmp.a[4]         =  3.88209629e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  4.99801080e+00;
  parmTmp.a[1]         = -2.63391328e+00;
  parmTmp.a[2]         =  1.49378631e-01;
  parmTmp.a[3]         = -4.51026121e-03;
  parmTmp.a[4]         =  5.26542576e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  8.32130145e+00;
  parmTmp.a[1]         = -3.46069076e+00;
  parmTmp.a[2]         =  2.01269608e-01;
  parmTmp.a[3]         = -6.17479219e-03;
  parmTmp.a[4]         =  7.27789702e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         10;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.14077848e+01;
  parmTmp.a[1]         = -4.07983008e+00;
  parmTmp.a[2]         =  2.32578736e-01;
  parmTmp.a[3]         = -7.02505557e-03;
  parmTmp.a[4]         =  8.18077178e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         12;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.45063182e+01;
  parmTmp.a[1]         = -4.64550073e+00;
  parmTmp.a[2]         =  2.65471547e-01;
  parmTmp.a[3]         = -8.28168313e-03;
  parmTmp.a[4]         =  1.02337777e-04;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         14;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.41565079e+01;
  parmTmp.a[1]         = -3.92906051e+00;
  parmTmp.a[2]         =  1.55721926e-01;
  parmTmp.a[3]         = -3.36095894e-03;
  parmTmp.a[4]         =  3.58825307e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         16;
  parmTmp.alpha        = -4.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.20189001e+01;
  parmTmp.a[1]         = -2.54084628e+00;
  parmTmp.a[2]         = -3.46192653e-02;
  parmTmp.a[3]         =  5.60981086e-03;
  parmTmp.a[4]         = -9.82515320e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          2;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  8.72200082e-01;
  parmTmp.a[1]         = -9.68492214e-01;
  parmTmp.a[2]         =  4.52398953e-02;
  parmTmp.a[3]         = -1.18532380e-03;
  parmTmp.a[4]         =  1.25556212e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          3;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  1.83198988e+00;
  parmTmp.a[1]         = -1.24358019e+00;
  parmTmp.a[2]         =  6.09328516e-02;
  parmTmp.a[3]         = -1.65723707e-03;
  parmTmp.a[4]         =  1.80539707e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  2.49692187e+00;
  parmTmp.a[1]         = -1.64576902e+00;
  parmTmp.a[2]         =  8.27878618e-02;
  parmTmp.a[3]         = -2.28363899e-03;
  parmTmp.a[4]         =  2.49641481e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  4.03627508e+00;
  parmTmp.a[1]         = -2.04836318e+00;
  parmTmp.a[2]         =  1.08149867e-01;
  parmTmp.a[3]         = -3.09233781e-03;
  parmTmp.a[4]         =  3.46591328e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  5.15912971e+00;
  parmTmp.a[1]         = -2.27273531e+00;
  parmTmp.a[2]         =  1.21469723e-01;
  parmTmp.a[3]         = -3.46672598e-03;
  parmTmp.a[4]         =  3.88064969e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  2.67356406e+00;
  parmTmp.a[1]         = -1.73365965e+00;
  parmTmp.a[2]         =  9.02879526e-02;
  parmTmp.a[3]         = -2.55197468e-03;
  parmTmp.a[4]         =  2.83886473e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  4.01618690e+00;
  parmTmp.a[1]         = -2.13091365e+00;
  parmTmp.a[2]         =  1.15333723e-01;
  parmTmp.a[3]         = -3.36018423e-03;
  parmTmp.a[4]         =  3.81816965e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  4.98266677e+00;
  parmTmp.a[1]         = -2.35556150e+00;
  parmTmp.a[2]         =  1.23104045e-01;
  parmTmp.a[3]         = -3.48799606e-03;
  parmTmp.a[4]         =  3.87965623e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          7;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  6.83111617e+00;
  parmTmp.a[1]         = -2.79303059e+00;
  parmTmp.a[2]         =  1.51514612e-01;
  parmTmp.a[3]         = -4.41437577e-03;
  parmTmp.a[4]         =  5.00742560e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  8.43502762e+00;
  parmTmp.a[1]         = -3.10387029e+00;
  parmTmp.a[2]         =  1.68082246e-01;
  parmTmp.a[3]         = -4.87320899e-03;
  parmTmp.a[4]         =  5.51063579e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  3.37845705e+00;
  parmTmp.a[1]         = -1.93803055e+00;
  parmTmp.a[2]         =  1.09501757e-01;
  parmTmp.a[3]         = -3.29441700e-03;
  parmTmp.a[4]         =  3.83785311e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  5.70133095e+00;
  parmTmp.a[1]         = -2.61343464e+00;
  parmTmp.a[2]         =  1.47718538e-01;
  parmTmp.a[3]         = -4.45220065e-03;
  parmTmp.a[4]         =  5.19246462e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  9.11901821e+00;
  parmTmp.a[1]         = -3.43987464e+00;
  parmTmp.a[2]         =  1.99672203e-01;
  parmTmp.a[3]         = -6.12128417e-03;
  parmTmp.a[4]         =  7.21290024e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         10;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.22727187e+01;
  parmTmp.a[1]         = -4.05496416e+00;
  parmTmp.a[2]         =  2.30631090e-01;
  parmTmp.a[3]         = -6.95851305e-03;
  parmTmp.a[4]         =  8.09829541e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         12;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.54354646e+01;
  parmTmp.a[1]         = -4.62156963e+00;
  parmTmp.a[2]         =  2.63857064e-01;
  parmTmp.a[3]         = -8.23466036e-03;
  parmTmp.a[4]         =  1.01799704e-04;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         14;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.52546306e+01;
  parmTmp.a[1]         = -3.95471322e+00;
  parmTmp.a[2]         =  1.61019650e-01;
  parmTmp.a[3]         = -3.66986703e-03;
  parmTmp.a[4]         =  4.10600132e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         16;
  parmTmp.alpha        = -5.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.30408034e+01;
  parmTmp.a[1]         = -2.51078782e+00;
  parmTmp.a[2]         = -3.72447798e-02;
  parmTmp.a[3]         =  5.73056882e-03;
  parmTmp.a[4]         = -1.00206550e-04;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          2;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  1.21703299e+00;
  parmTmp.a[1]         = -9.54974489e-01;
  parmTmp.a[2]         =  4.42561284e-02;
  parmTmp.a[3]         = -1.15343877e-03;
  parmTmp.a[4]         =  1.21773581e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          3;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  2.26070334e+00;
  parmTmp.a[1]         = -1.23651205e+00;
  parmTmp.a[2]         =  6.04083702e-02;
  parmTmp.a[3]         = -1.64006655e-03;
  parmTmp.a[4]         =  1.78443007e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  3.00175919e+00;
  parmTmp.a[1]         = -1.63162695e+00;
  parmTmp.a[2]         =  8.17537869e-02;
  parmTmp.a[3]         = -2.25011178e-03;
  parmTmp.a[4]         =  2.45666656e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  4.59628376e+00;
  parmTmp.a[1]         = -2.03466330e+00;
  parmTmp.a[2]         =  1.07199651e-01;
  parmTmp.a[3]         = -3.06271645e-03;
  parmTmp.a[4]         =  3.43150348e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  5.76757082e+00;
  parmTmp.a[1]         = -2.25872519e+00;
  parmTmp.a[2]         =  1.20120013e-01;
  parmTmp.a[3]         = -3.41786880e-03;
  parmTmp.a[4]         =  3.81906044e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  3.19315005e+00;
  parmTmp.a[1]         = -1.72272215e+00;
  parmTmp.a[2]         =  8.95300841e-02;
  parmTmp.a[3]         = -2.52858157e-03;
  parmTmp.a[4]         =  2.81226744e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  4.57240939e+00;
  parmTmp.a[1]         = -2.11328381e+00;
  parmTmp.a[2]         =  1.13996653e-01;
  parmTmp.a[3]         = -3.31572644e-03;
  parmTmp.a[4]         =  3.76436697e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  5.59408214e+00;
  parmTmp.a[1]         = -2.33815674e+00;
  parmTmp.a[2]         =  1.21807385e-01;
  parmTmp.a[3]         = -3.44538878e-03;
  parmTmp.a[4]         =  3.82861139e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          7;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  7.49870694e+00;
  parmTmp.a[1]         = -2.77853532e+00;
  parmTmp.a[2]         =  1.50534697e-01;
  parmTmp.a[3]         = -4.38442153e-03;
  parmTmp.a[4]         =  4.97332613e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  9.13361178e+00;
  parmTmp.a[1]         = -3.08650809e+00;
  parmTmp.a[2]         =  1.66804115e-01;
  parmTmp.a[3]         = -4.83286107e-03;
  parmTmp.a[4]         =  5.46319338e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  3.89283730e+00;
  parmTmp.a[1]         = -1.92458623e+00;
  parmTmp.a[2]         =  1.08460057e-01;
  parmTmp.a[3]         = -3.25904093e-03;
  parmTmp.a[4]         =  3.79427123e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  6.31115823e+00;
  parmTmp.a[1]         = -2.59391957e+00;
  parmTmp.a[2]         =  1.46155791e-01;
  parmTmp.a[3]         = -4.39794850e-03;
  parmTmp.a[4]         =  5.12462846e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  9.81961994e+00;
  parmTmp.a[1]         = -3.42034960e+00;
  parmTmp.a[2]         =  1.98200110e-01;
  parmTmp.a[3]         = -6.07262543e-03;
  parmTmp.a[4]         =  7.15439676e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         10;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.30397267e+01;
  parmTmp.a[1]         = -4.03197051e+00;
  parmTmp.a[2]         =  2.28848547e-01;
  parmTmp.a[3]         = -6.89770693e-03;
  parmTmp.a[4]         =  8.02245967e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         12;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.62043040e+01;
  parmTmp.a[1]         = -4.57649206e+00;
  parmTmp.a[2]         =  2.59711114e-01;
  parmTmp.a[3]         = -8.07806101e-03;
  parmTmp.a[4]         =  9.98643981e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         14;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.61682398e+01;
  parmTmp.a[1]         = -3.94544235e+00;
  parmTmp.a[2]         =  1.61355752e-01;
  parmTmp.a[3]         = -3.71726233e-03;
  parmTmp.a[4]         =  4.19427467e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         16;
  parmTmp.alpha        = -6.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.38025025e+01;
  parmTmp.a[1]         = -2.41724265e+00;
  parmTmp.a[2]         = -4.81306131e-02;
  parmTmp.a[3]         =  6.24996933e-03;
  parmTmp.a[4]         = -1.08327473e-04;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          2;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  1.50946487e+00;
  parmTmp.a[1]         = -9.42037771e-01;
  parmTmp.a[2]         =  4.33283808e-02;
  parmTmp.a[3]         = -1.12363066e-03;
  parmTmp.a[4]         =  1.18253867e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          3;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  2.63081457e+00;
  parmTmp.a[1]         = -1.22916623e+00;
  parmTmp.a[2]         =  5.98830959e-02;
  parmTmp.a[3]         = -1.62328499e-03;
  parmTmp.a[4]         =  1.76428571e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  3.44224472e+00;
  parmTmp.a[1]         = -1.61815309e+00;
  parmTmp.a[2]         =  8.07843333e-02;
  parmTmp.a[3]         = -2.21900060e-03;
  parmTmp.a[4]         =  2.42003422e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  5.08859702e+00;
  parmTmp.a[1]         = -2.02155102e+00;
  parmTmp.a[2]         =  1.06309558e-01;
  parmTmp.a[3]         = -3.03543471e-03;
  parmTmp.a[4]         =  3.40023339e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  6.30528017e+00;
  parmTmp.a[1]         = -2.24490895e+00;
  parmTmp.a[2]         =  1.18834135e-01;
  parmTmp.a[3]         = -3.37216486e-03;
  parmTmp.a[4]         =  3.76208123e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  3.64575194e+00;
  parmTmp.a[1]         = -1.71185244e+00;
  parmTmp.a[2]         =  8.87893692e-02;
  parmTmp.a[3]         = -2.50604411e-03;
  parmTmp.a[4]         =  2.78698134e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          5;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  5.05891133e+00;
  parmTmp.a[1]         = -2.09591423e+00;
  parmTmp.a[2]         =  1.12685622e-01;
  parmTmp.a[3]         = -3.27220487e-03;
  parmTmp.a[4]         =  3.71172646e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  6.13605457e+00;
  parmTmp.a[1]         = -2.32193900e+00;
  parmTmp.a[2]         =  1.20624383e-01;
  parmTmp.a[3]         = -3.40711439e-03;
  parmTmp.a[4]         =  3.78332224e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          7;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  8.09296017e+00;
  parmTmp.a[1]         = -2.76466564e+00;
  parmTmp.a[2]         =  1.49608687e-01;
  parmTmp.a[3]         = -4.35637107e-03;
  parmTmp.a[4]         =  4.94160292e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         =  9.75920242e+00;
  parmTmp.a[1]         = -3.07025724e+00;
  parmTmp.a[2]         =  1.65625515e-01;
  parmTmp.a[3]         = -4.79582502e-03;
  parmTmp.a[4]         =  5.41968568e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          4;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  4.33977283e+00;
  parmTmp.a[1]         = -1.91104596e+00;
  parmTmp.a[2]         =  1.07418002e-01;
  parmTmp.a[3]         = -3.22376522e-03;
  parmTmp.a[4]         =  3.75083743e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          6;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  6.85057354e+00;
  parmTmp.a[1]         = -2.57533169e+00;
  parmTmp.a[2]         =  1.44683082e-01;
  parmTmp.a[3]         = -4.34713281e-03;
  parmTmp.a[4]         =  5.06132154e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          8;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.04458346e+01;
  parmTmp.a[1]         = -3.40176146e+00;
  parmTmp.a[2]         =  1.96809095e-01;
  parmTmp.a[3]         = -6.02678636e-03;
  parmTmp.a[4]         =  7.09934331e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         10;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.37341621e+01;
  parmTmp.a[1]         = -4.01145879e+00;
  parmTmp.a[2]         =  2.27314607e-01;
  parmTmp.a[3]         = -6.84731037e-03;
  parmTmp.a[4]         =  7.96230577e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         12;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.69651692e+01;
  parmTmp.a[1]         = -4.55781769e+00;
  parmTmp.a[2]         =  2.58405229e-01;
  parmTmp.a[3]         = -8.03019691e-03;
  parmTmp.a[4]         =  9.90283382e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         14;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.68723629e+01;
  parmTmp.a[1]         = -3.88343873e+00;
  parmTmp.a[2]         =  1.54742012e-01;
  parmTmp.a[3]         = -3.42147277e-03;
  parmTmp.a[4]         =  3.74905962e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         16;
  parmTmp.alpha        = -7.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  1.45817891e+01;
  parmTmp.a[1]         = -2.35242020e+00;
  parmTmp.a[2]         = -5.72171742e-02;
  parmTmp.a[3]         =  6.77135769e-03;
  parmTmp.a[4]         = -1.17363649e-04;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          2;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -3.25970277e+00;
  parmTmp.a[1]         = -1.06297188e+00;
  parmTmp.a[2]         =  5.22474169e-02;
  parmTmp.a[3]         = -1.41573001e-03;
  parmTmp.a[4]         =  1.53208720e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          4;
  parmTmp.Nder         =          3;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  4.28572200e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -3.02246574e+00;
  parmTmp.a[1]         = -1.23948122e+00;
  parmTmp.a[2]         =  6.11495966e-02;
  parmTmp.a[3]         = -1.67903963e-03;
  parmTmp.a[4]         =  1.85211839e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          4;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -2.56499333e+00;
  parmTmp.a[1]         = -1.85225088e+00;
  parmTmp.a[2]         =  9.85588114e-02;
  parmTmp.a[3]         = -2.80054286e-03;
  parmTmp.a[4]         =  3.10787903e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          5;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -1.30036751e+00;
  parmTmp.a[1]         = -2.25147673e+00;
  parmTmp.a[2]         =  1.21553694e-01;
  parmTmp.a[3]         = -3.47316555e-03;
  parmTmp.a[4]         =  3.86256515e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          6;
  parmTmp.Nder         =          6;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  4.09524500e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -7.22390072e-01;
  parmTmp.a[1]         = -2.40948984e+00;
  parmTmp.a[2]         =  1.32329005e-01;
  parmTmp.a[3]         = -3.80580727e-03;
  parmTmp.a[4]         =  4.25446569e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          4;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -2.06529555e+00;
  parmTmp.a[1]         = -2.01794209e+00;
  parmTmp.a[2]         =  1.12460636e-01;
  parmTmp.a[3]         = -3.28892061e-03;
  parmTmp.a[4]         =  3.71985857e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          5;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -6.49541860e-01;
  parmTmp.a[1]         = -2.47598419e+00;
  parmTmp.a[2]         =  1.38136067e-01;
  parmTmp.a[3]         = -4.00966673e-03;
  parmTmp.a[4]         =  4.49213464e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          6;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -7.65332815e-01;
  parmTmp.a[1]         = -2.38358641e+00;
  parmTmp.a[2]         =  1.12125334e-01;
  parmTmp.a[3]         = -2.81244926e-03;
  parmTmp.a[4]         =  2.80071877e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          7;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -1.15558635e+00;
  parmTmp.a[1]         = -2.20007124e+00;
  parmTmp.a[2]         =  8.58749742e-02;
  parmTmp.a[3]         = -1.88007728e-03;
  parmTmp.a[4]         =  1.74116576e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          8;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  3.90476900e+00;
  parmTmp.uMax         =  3.20043040e+01;
  parmTmp.a[0]         = -9.99849711e-01;
  parmTmp.a[1]         = -2.22574979e+00;
  parmTmp.a[2]         =  8.80977989e-02;
  parmTmp.a[3]         = -2.18156755e-03;
  parmTmp.a[4]         =  2.42578505e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          4;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  2.05085314e+00;
  parmTmp.a[1]         = -3.08054502e+00;
  parmTmp.a[2]         =  2.06284606e-01;
  parmTmp.a[3]         = -6.72122265e-03;
  parmTmp.a[4]         =  8.14500062e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          6;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  3.20606426e+00;
  parmTmp.a[1]         = -3.23066062e+00;
  parmTmp.a[2]         =  1.73897451e-01;
  parmTmp.a[3]         = -4.75037389e-03;
  parmTmp.a[4]         =  5.01050851e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          8;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  2.11979583e+00;
  parmTmp.a[1]         = -2.75122141e+00;
  parmTmp.a[2]         =  1.13778645e-01;
  parmTmp.a[3]         = -2.68394106e-03;
  parmTmp.a[4]         =  2.68400507e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         10;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  3.25209374e+00;
  parmTmp.a[1]         = -3.14711996e+00;
  parmTmp.a[2]         =  1.59369765e-01;
  parmTmp.a[3]         = -4.83359480e-03;
  parmTmp.a[4]         =  5.98199560e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         12;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  3.84476629e+00;
  parmTmp.a[1]         = -3.31119513e+00;
  parmTmp.a[2]         =  1.74444257e-01;
  parmTmp.a[3]         = -5.39532685e-03;
  parmTmp.a[4]         =  6.70888176e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         14;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  4.71251704e+00;
  parmTmp.a[1]         = -3.52984359e+00;
  parmTmp.a[2]         =  1.93599304e-01;
  parmTmp.a[3]         = -6.09507192e-03;
  parmTmp.a[4]         =  7.61338047e-05;
  _parm.push_back(parmTmp);



  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         16;
  parmTmp.alpha        =  1.00000000e+00;
  parmTmp.uMin         =  3.14286300e+00;
  parmTmp.uMax         =  3.18228510e+01;
  parmTmp.a[0]         =  5.70158853e+00;
  parmTmp.a[1]         = -3.73579013e+00;
  parmTmp.a[2]         =  2.08625035e-01;
  parmTmp.a[3]         = -6.55880323e-03;
  parmTmp.a[4]         =  8.13422651e-05;
  _parm.push_back(parmTmp);





  return true;
}

//******************************************************************************
//******************************************************************************
//*********************************Helmholtz Model******************************
//******************************************************************************
//******************************************************************************







bool NPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_outputNder (
    long int& N1, long int& N2, long int& N3, 
    long int& n1, long int& n2, long int& n3,
    int& Nder,
    const size_t nCharge, const _Complex double *charge, const double *coord, 
    const long int BsplineOrder, const _Complex double k0, const double tol,
    const double Rdir, bool printLog, std::ostream& os) const
{
  using std::cout;


  std::vector<size_t> parmIndexList;
  if (!GetParmIndexList (parmIndexList, BsplineOrder))
  {
    cout << "Error in NPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_outputNder\n";
    cout << "  GetParmIndexList failed\n";
    return false;
  }

  long int N1_trial, N2_trial, N3_trial;
  long int n1_trial, n2_trial, n3_trial;
  if (!Get_FFT_inputParmIndex (
          N1_trial, N2_trial, N3_trial,
          n1_trial, n2_trial, n3_trial,
          nCharge, charge, coord,
          parmIndexList[0], k0, tol, Rdir, printLog, os))
  {
    cout << "Error in NPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_outputNder\n";
    cout << "  GetParmIndexList failed\n";
    return false;
  }

  size_t minGridSizeParmIndex = parmIndexList[0];
  long int minGridSize        = N1_trial*N2_trial*N3_trial;
  N1                          = N1_trial;
  N2                          = N2_trial;
  N3                          = N3_trial;
  n1                          = n1_trial;
  n2                          = n2_trial;
  n3                          = n3_trial;

  for (size_t n = 1; n < parmIndexList.size(); n++)
  {
    if (!Get_FFT_inputParmIndex (
            N1_trial, N2_trial, N3_trial,
            n1_trial, n2_trial, n3_trial,
            nCharge, charge, coord,
            parmIndexList[n], k0, tol, Rdir, printLog, os))
    {
      cout << "Error in NPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_outputNder\n";
      cout << "  Get_FFT_inputParmIndex failed\n";
      return false;
    }

    if (minGridSize > N1_trial*N2_trial*N3_trial)
    {
      minGridSizeParmIndex  = parmIndexList[n];
      minGridSize           = N1_trial*N2_trial*N3_trial;
      N1                    = N1_trial;
      N2                    = N2_trial;
      N3                    = N3_trial;
      n1                    = n1_trial;
      n2                    = n2_trial;
      n3                    = n3_trial;
    }
  }

  Nder = _parm[minGridSizeParmIndex].Nder;

  return true;
}



bool NPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_inputNder (
    long int& N1, long int& N2, long int& N3, 
    long int& n1, long int& n2, long int& n3,
    const size_t nCharge, const _Complex double *charge, const double *coord, 
    const long int BsplineOrder, const _Complex double k0, const int Nder, 
    const double tol, const double Rdir, bool printLog, std::ostream& os) const
{
  using std::cout;

  size_t parmIndex;
  if (!GetParmIndex (parmIndex, BsplineOrder, Nder))
  {
    cout << "Error in NPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_inputNder\n";
    cout << "  GetParmIndex failed\n";
    return false;
  }

  return Get_FFT_inputParmIndex (
            N1, N2, N3, 
            n1, n2, n3,
            nCharge, charge, coord, parmIndex,
            k0, tol, Rdir, printLog, os);
}

bool NPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_inputParmIndex (
    long int& N1, long int& N2, long int& N3, 
    long int& n1, long int& n2, long int& n3,
    const size_t nCharge, const _Complex double *charge, const double *coord, 
    const size_t parmIndex,
    const _Complex double k0, const double tol,
    const double Rdir, bool printLog, std::ostream& os) const
{
  if (!_isSet)
  {
    std::cout << "Error in NPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_inputParmIndex\n";
    std::cout << "_isSet = 0\n";
    return false;
  }

  const size_t MaxIteration = 10;
  char str[2000];

  //1) assume del = Rdir
  const double del = Rdir;

  //2) get model parameters
  const int Nder              = _parm[parmIndex].Nder;
  const double uMin           = _parm[parmIndex].uMin;
  const double uMax           = _parm[parmIndex].uMax;
  const double wMin           = _parm[parmIndex].wMin;
  const double wMax           = _parm[parmIndex].wMax;
  const long int BsplineOrder = _parm[parmIndex].BsplineOrder;
  const double *a             = _parm[parmIndex].a;


  if (printLog)
  {
    os << "\n\nNPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_inputParmIndex\n";
    sprintf(str, "   Nder          = %d\n", Nder);            os << str;
    sprintf(str, "   uMin          = %f\n", uMin);            os << str;
    sprintf(str, "   uMax          = %f\n", uMax);            os << str;
    sprintf(str, "   wMin          = %f\n", wMin);            os << str;
    sprintf(str, "   wMax          = %f\n", wMax);            os << str;
    sprintf(str, "   Rdir          = %f\n", Rdir);            os << str;
    sprintf(str, "   BsplineOrder  = %ld\n", BsplineOrder);   os << str;
    sprintf(str, "   tol           = %.2le\n", tol);          os << str;
    os.flush();
  }

  //y = log(theta) = a[p][q]*u^p*w^q
  //note that w is a constant w.r.t. u or Nfft
  //let b[p] = a[p][q]*w^q
  double b[5];
  {
    const double w = cabs(k0)*Rdir;
    if ( (w < wMin) || (w > wMax))
    {
      std::cout << "Error in NPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_inputParmIndex\n";
      sprintf(str, "w = %f is outside wMin = %f wMax = %f\n",
        w, wMin, wMax);
      std::cout << str;
      return false;
    }

    double w_q[5];
    for (size_t q = 0; q < 5; q++)
    {
      if (q == 0) w_q[q] = 1;
      else        w_q[q] = w_q[q-1]*w;
    }
  
    for (size_t p = 0; p < 5; p++)
    {
      b[p] = 0;
      for (size_t q = 0; q < 5; q++)
        b[p] += a[p*5+q]*w_q[q];
    }
  }
  

  //physical dimension
  double X0, Y0, Z0;
  double R0[3];
  NPME_CalcBoxDimensionCenter (nCharge, coord, X0, Y0, Z0, R0);
  const double Q0 = NPME_EstimateParmCalcQsum (nCharge, charge);

  if (printLog)
  {
    sprintf(str, "   X0 Y0 Z0      = %10.6f %10.6f %10.6f\n", X0, Y0, Z0);
    os << str;
    sprintf(str, "   Q0            = %10.6f\n", Q0);
    os << str;
    os.flush();
  }

  N1 = 100;
  N2 = 100;
  N3 = 100;

  for (size_t n = 0; n < MaxIteration; n++)
  {
    if (printLog)
    {
      sprintf(str, "Iteration %lu\n", n);
      os << str;
    }

    double X, Y, Z;
    double R[3];
    if (n == 0)
    {
      X = X0;
      Y = Y0;
      Z = Z0;
    }
    else
    {
      NPME_RecSumInterface_GetPMECorrBox (
        X,  Y,  Z,  R, 
        X0, Y0, Z0, R0,
        N1, N2, N3,
        del, BsplineOrder);
    }
    double L1, L2, L3;
    L1 = 2*(X + del);
    L2 = 2*(Y + del);
    L3 = 2*(Z + del);

    const double Q1 = Q0*sqrt(L1*L1*L1/(L1*L2*L3));
    const double Q2 = Q0*sqrt(L2*L2*L2/(L1*L2*L3));
    const double Q3 = Q0*sqrt(L3*L3*L3/(L1*L2*L3));

    if (printLog)
    {
      sprintf(str, "   X  Y  Z       = %10.6f %10.6f %10.6f\n", X,  Y,  Z);
      os << str;
      sprintf(str, "   L1 L2 L3      = %10.6f %10.6f %10.6f\n", L1, L2, L3);
      os << str;
      sprintf(str, "   Q1 Q2 Q3      = %10.6f %10.6f %10.6f\n", Q1, Q2, Q3);
      os << str;
      os.flush();
    }

    //5) log(theta) reference values
    double y1, y2, y3;
    {
      const double C  = tol*Rdir;
      const double t1 = C/Q1*pow(L1/Rdir, 1.5);
      const double t2 = C/Q2*pow(L2/Rdir, 1.5);
      const double t3 = C/Q3*pow(L3/Rdir, 1.5);

      y1 = log(t1);
      y2 = log(t2);
      y3 = log(t3);

      if (printLog)
      {
        sprintf(str, "   t1 t2 t3      = %.2le %.2le %.2le\n", t1, t2, t3);
        os << str;
        sprintf(str, "   y1 y2 y3      = %10.6f %10.6f %10.6f\n", y1, y2, y3);
        os << str;
      }
    }

    double u1, u2, u3;
    if (!NPME_EstimateParmSingleBox_FindOptimal_u (u1, y1, b, uMin, uMax))
    {
      std::cout << "Error in NPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_inputParmIndex\n";
      std::cout << "NPME_EstimateParmSingleBox_FindOptimal_u failed\n";
      return false;
    }
    if (!NPME_EstimateParmSingleBox_FindOptimal_u (u2, y2, b, uMin, uMax))
    {
      std::cout << "Error in NPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_inputParmIndex\n";
      std::cout << "NPME_EstimateParmSingleBox_FindOptimal_u failed\n";
      return false;
    }
    if (!NPME_EstimateParmSingleBox_FindOptimal_u (u3, y3, b, uMin, uMax))
    {
      std::cout << "Error in NPME_EstimateParmSingleBox_HelmholtzModel::Get_FFT_inputParmIndex\n";
      std::cout << "NPME_EstimateParmSingleBox_FindOptimal_u failed\n";
      return false;
    }

    //u = N*Rdir/L
    N1 = (long int) (u1*L1/Rdir) + 1;
    N2 = (long int) (u2*L2/Rdir) + 1;
    N3 = (long int) (u3*L3/Rdir) + 1;

    if (N1 < 4*BsplineOrder + 1)  N1 = 4*BsplineOrder + 1;
    if (N2 < 4*BsplineOrder + 1)  N2 = 4*BsplineOrder + 1;
    if (N3 < 4*BsplineOrder + 1)  N3 = 4*BsplineOrder + 1;

  //NPME_FindFFTSizeBlockSize (N1, n1, N1, NPME_IdealRecSumBlockSize);
  //NPME_FindFFTSizeBlockSize (N2, n2, N2, NPME_IdealRecSumBlockSize);
  //NPME_FindFFTSizeBlockSize (N3, n3, N3, NPME_IdealRecSumBlockSize);

    if (printLog)
    {
      sprintf(str, "   u1 u2 u3      = %10.6f %10.6f %10.6f\n", u1, u2, u3);
      os << str;
      sprintf(str, "   N1 N2 N3      = %10ld %10ld %10ld\n", 
        N1, N2, N3);
      os << str;
      os.flush();
    }
  }


  NPME_FindFFTSizeBlockSize (N1, n1, N1, NPME_IdealRecSumBlockSize);
  NPME_FindFFTSizeBlockSize (N2, n2, N2, NPME_IdealRecSumBlockSize);
  NPME_FindFFTSizeBlockSize (N3, n3, N3, NPME_IdealRecSumBlockSize);

  if (printLog)
  {
    sprintf(str, "   N1 N2 N3      = %10ld %10ld %10ld\n", N1, N2, N3);
    os << str;
    sprintf(str, "   n1 n2 n3      = %10ld %10ld %10ld\n", n1, n2, n3);
    os << str;
    os.flush();
  }

  return true;
}

bool NPME_EstimateParmSingleBox_HelmholtzModel::CalcPredError_V (
    double& eps_V, bool& reliablePred,
    const long int N1, const long int N2, const long int N3, 
    const size_t nCharge, const _Complex double *charge, const double *coord, 
    const long int BsplineOrder, const _Complex double k0, const int Nder, 
    const double Rdir, bool printLog, std::ostream& os)
{
  using std::cout;

  size_t parmIndex;
  if (!GetParmIndex (parmIndex, BsplineOrder, Nder))
  {
    cout << "Error in NPME_EstimateParmSingleBox_HelmholtzModel::CalcPredError_V\n";
    cout << "  GetParmIndex failed\n";
    return false;
  }


  char str[2000];

  //1) assume del = Rdir
  const double del = Rdir;

  //2) get model parameters
  const double uMin           = _parm[parmIndex].uMin;
  const double uMax           = _parm[parmIndex].uMax;
  const double wMin           = _parm[parmIndex].wMin;
  const double wMax           = _parm[parmIndex].wMax;
  const double *a             = _parm[parmIndex].a;


  if (printLog)
  {
    os << "\n\nNPME_EstimateParmSingleBox_HelmholtzModel::CalcPredError_V\n";
    sprintf(str, "   Nder          = %d\n", Nder);            os << str;
    sprintf(str, "   uMin          = %f\n", uMin);            os << str;
    sprintf(str, "   uMax          = %f\n", uMax);            os << str;
    sprintf(str, "   wMin          = %f\n", wMin);            os << str;
    sprintf(str, "   wMax          = %f\n", wMax);            os << str;
    sprintf(str, "   Rdir          = %f\n", Rdir);            os << str;
    sprintf(str, "   BsplineOrder  = %ld\n", BsplineOrder);   os << str;
    os.flush();
  }

  //physical dimension
  double X0, Y0, Z0;
  double R0[3];
  NPME_CalcBoxDimensionCenter (nCharge, coord, X0, Y0, Z0, R0);
  const double Q0 = NPME_EstimateParmCalcQsum (nCharge, charge);

  if (printLog)
  {
    sprintf(str, "   X0 Y0 Z0      = %10.6f %10.6f %10.6f\n", X0, Y0, Z0);
    os << str;
    sprintf(str, "   Q0            = %10.6f\n", Q0);
    os << str;
    os << "\n\n";
    os.flush();
  }

  double X, Y, Z;
  double R[3];
  NPME_RecSumInterface_GetPMECorrBox (
    X,  Y,  Z,  R, 
    X0, Y0, Z0, R0,
    N1, N2, N3,
    del, BsplineOrder);

  double L1, L2, L3;
  L1 = 2*(X + del);
  L2 = 2*(Y + del);
  L3 = 2*(Z + del);


  //y = log(theta) = a[p][q]*u^p*w^q
  //note that w is a constant w.r.t. u or Nfft
  //let b[p] = a[p][q]*w^q
  double b[5];
  {
    const double w = cabs(k0)*Rdir;
    if ( (w < wMin) || (w > wMax))
    {
      std::cout << "Error in NPME_EstimateParmSingleBox_HelmholtzModel::CalcPredError_V\n";
      sprintf(str, "w = %f is outside wMin = %f wMax = %f\n",
        w, wMin, wMax);
      std::cout << str;
      return false;
    }

    double w_q[5];
    for (size_t q = 0; q < 5; q++)
    {
      if (q == 0) w_q[q] = 1;
      else        w_q[q] = w_q[q-1]*w;
    }
  
    for (size_t p = 0; p < 5; p++)
    {
      b[p] = 0;
      for (size_t q = 0; q < 5; q++)
        b[p] += a[p*5+q]*w_q[q];
    }
  }


  const double u1 = N1*Rdir/L1;
  const double u2 = N2*Rdir/L2;
  const double u3 = N3*Rdir/L3;

  //prediction is reliable if uMin <= u1,u2,u3 <= uMax
  reliablePred = 1;
  if ((u1 < uMin) || (u1 > uMax)) reliablePred = 0;
  if ((u2 < uMin) || (u2 > uMax)) reliablePred = 0;
  if ((u3 < uMin) || (u3 > uMax)) reliablePred = 0;

  const double Q1     = Q0*sqrt(L1*L1*L1/(L1*L2*L3));
  const double Q2     = Q0*sqrt(L2*L2*L2/(L1*L2*L3));
  const double Q3     = Q0*sqrt(L3*L3*L3/(L1*L2*L3));

  const double P1     = NPME_EstimateParmSingleBox_Calc_Pu (b, u1);
  const double P2     = NPME_EstimateParmSingleBox_Calc_Pu (b, u2);
  const double P3     = NPME_EstimateParmSingleBox_Calc_Pu (b, u3);

  const double eps_V1 = Q1*pow(Rdir, -1.0)*pow(Rdir/L1, 1.5)*exp(P1);
  const double eps_V2 = Q2*pow(Rdir, -1.0)*pow(Rdir/L2, 1.5)*exp(P2);
  const double eps_V3 = Q3*pow(Rdir, -1.0)*pow(Rdir/L3, 1.5)*exp(P3);

  eps_V  = NPME_Max (eps_V1, eps_V2, eps_V3);

  if (printLog)
  {
    sprintf(str, "   X  Y  Z              = %10.6f %10.6f %10.6f\n", 
      X,  Y,  Z);
    os << str;
    sprintf(str, "   L1 L2 L3             = %10.6f %10.6f %10.6f\n", 
      L1, L2, L3);
    os << str;
    sprintf(str, "   Q1 Q2 Q3             = %10.6f %10.6f %10.6f\n", 
      Q1, Q2, Q3);
    sprintf(str, "   N1 N2 N3             = %10ld %10ld %10ld\n", 
      N1, N2, N3);
    os << str;
    sprintf(str, "   u1 u2 u3             = %10.6f %10.6f %10.6f\n", 
      u1, u2, u3);
    os << str;
    sprintf(str, "   P1 P2 P3             = %10.6f %10.6f %10.6f\n", 
      P1, P2, P3);
    os << str;
    sprintf(str, "   eps_V1 eps_V1 eps_V1 = %.4le %.4le %.4le\n", 
      eps_V1, eps_V2, eps_V3);
    os << str;
    sprintf(str, "   reliablePred         = %d\n", (int) reliablePred);
    os << str;
    sprintf(str, "   eps_V                = %.4le\n", eps_V);
    os << str;
    os.flush();
  }


  return true;
}


bool NPME_EstimateParmSingleBox_HelmholtzModel::GetParm (
  double& uMin, double& uMax, 
  double& wMin, double& wMax, 
  double a[25], 
  const long int BsplineOrder, const int Nder) const
{
  for (size_t n = 0; n < _parm.size(); n++)
    if (  (BsplineOrder  == _parm[n].BsplineOrder) &&
          (Nder          == _parm[n].Nder) )
    {
      uMin = _parm[n].uMin;
      uMax = _parm[n].uMax;
      wMin = _parm[n].wMin;
      wMax = _parm[n].wMax;
      for (int p = 0; p < 25; p++)
        a[p] = _parm[n].a[p];

      return true;
    }

  return false;
}

bool NPME_EstimateParmSingleBox_HelmholtzModel::GetParmIndex (size_t& parmIndex,
  const long int BsplineOrder, const int Nder) const
//finds parmIndex which haa the correct BsplineOrder
{
  using std::cout;

  for (size_t n = 0; n < _parm.size(); n++)
    if (BsplineOrder == _parm[n].BsplineOrder)
      if (Nder == _parm[n].Nder)
      {
        parmIndex = n;
        return true;
      }

  char str[2000];
  cout << "NPME_EstimateParmSingleBox_HelmholtzModel::GetParmIndex\n";
  sprintf(str, "parameters not defined for BsplineOrder = %ld Nder = %d\n",
    BsplineOrder, Nder);
  cout << str;
    
  return false;
}

bool NPME_EstimateParmSingleBox_HelmholtzModel::GetParmIndexList (
  std::vector<size_t>& parmIndexList,
  const long int BsplineOrder) const
//finds list of parmIndexes which have the correct BsplineOrder
{
  parmIndexList.clear();
  for (size_t n = 0; n < _parm.size(); n++)
    if (BsplineOrder == _parm[n].BsplineOrder)
    {
      parmIndexList.push_back(n);
    }

  if (parmIndexList.size() == 0)
  {
    using std::cout;
    cout << "Error in NPME_EstimateParmSingleBox_HelmholtzModel::GetParmIndexList\n";
    cout << "  parmIndexList is empty.\n";
    char str[500];
    sprintf(str, "parameters not available for BsplineOrder = %ld\n",
      BsplineOrder);
    return false;
  }

  return true;
}


bool NPME_EstimateParmSingleBox_HelmholtzModel::SetParm ()
{
  if (_isSet)
    return true;

  _isSet = 1;
  _parm.clear();

  NPME_ThetaParmHelmholtz parmTmp;

  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          4;
  parmTmp.uMin         =   4.439029;
  parmTmp.uMax         =  41.679975;
  parmTmp.wMin         =   0.000000;
  parmTmp.wMax         =  16.000000;

  parmTmp.a[ 0]        =  -1.301090e+00;
  parmTmp.a[ 1]        =  -2.016201e-01;
  parmTmp.a[ 2]        =   3.510645e-02;
  parmTmp.a[ 3]        =   7.371786e-03;
  parmTmp.a[ 4]        =  -5.292980e-04;
  parmTmp.a[ 5]        =  -1.682789e+00;
  parmTmp.a[ 6]        =   4.575102e-02;
  parmTmp.a[ 7]        =   1.607907e-02;
  parmTmp.a[ 8]        =  -3.037071e-03;
  parmTmp.a[ 9]        =   1.328048e-04;
  parmTmp.a[10]        =   8.539835e-02;
  parmTmp.a[11]        =  -6.656809e-03;
  parmTmp.a[12]        =  -4.827716e-04;
  parmTmp.a[13]        =   1.548256e-04;
  parmTmp.a[14]        =  -7.348546e-06;
  parmTmp.a[15]        =  -2.368402e-03;
  parmTmp.a[16]        =   3.147860e-04;
  parmTmp.a[17]        =  -6.082482e-06;
  parmTmp.a[18]        =  -2.921030e-06;
  parmTmp.a[19]        =   1.655311e-07;
  parmTmp.a[20]        =   2.600540e-05;
  parmTmp.a[21]        =  -4.744124e-06;
  parmTmp.a[22]        =   2.933109e-07;
  parmTmp.a[23]        =   1.362877e-08;
  parmTmp.a[24]        =  -1.246089e-09;
  _parm.push_back(parmTmp);


  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          6;
  parmTmp.uMin         =   4.439029;
  parmTmp.uMax         =  41.679975;
  parmTmp.wMin         =   0.000000;
  parmTmp.wMax         =  16.000000;

  parmTmp.a[ 0]        =   3.861960e-01;
  parmTmp.a[ 1]        =   1.496281e-01;
  parmTmp.a[ 2]        =  -1.505594e-01;
  parmTmp.a[ 3]        =   3.044945e-02;
  parmTmp.a[ 4]        =  -1.306729e-03;
  parmTmp.a[ 5]        =  -2.295812e+00;
  parmTmp.a[ 6]        =  -7.644093e-03;
  parmTmp.a[ 7]        =   3.806550e-02;
  parmTmp.a[ 8]        =  -5.621879e-03;
  parmTmp.a[ 9]        =   2.224230e-04;
  parmTmp.a[10]        =   1.180305e-01;
  parmTmp.a[11]        =  -4.268769e-03;
  parmTmp.a[12]        =  -1.761987e-03;
  parmTmp.a[13]        =   3.088724e-04;
  parmTmp.a[14]        =  -1.280458e-05;
  parmTmp.a[15]        =  -3.321375e-03;
  parmTmp.a[16]        =   3.021835e-04;
  parmTmp.a[17]        =   2.146085e-05;
  parmTmp.a[18]        =  -6.578191e-06;
  parmTmp.a[19]        =   3.005544e-07;
  parmTmp.a[20]        =   3.697678e-05;
  parmTmp.a[21]        =  -5.571849e-06;
  parmTmp.a[22]        =   1.828515e-07;
  parmTmp.a[23]        =   3.758614e-08;
  parmTmp.a[24]        =  -2.260830e-09;
  _parm.push_back(parmTmp);


  parmTmp.BsplineOrder =          8;
  parmTmp.Nder         =          8;
  parmTmp.uMin         =   4.439029;
  parmTmp.uMax         =  41.679975;
  parmTmp.wMin         =   0.000000;
  parmTmp.wMax         =  16.000000;

  parmTmp.a[ 0]        =   3.022587e+00;
  parmTmp.a[ 1]        =   4.774617e-01;
  parmTmp.a[ 2]        =  -3.223788e-01;
  parmTmp.a[ 3]        =   4.835895e-02;
  parmTmp.a[ 4]        =  -1.905552e-03;
  parmTmp.a[ 5]        =  -2.942561e+00;
  parmTmp.a[ 6]        =  -4.569314e-02;
  parmTmp.a[ 7]        =   5.936709e-02;
  parmTmp.a[ 8]        =  -7.528582e-03;
  parmTmp.a[ 9]        =   2.828380e-04;
  parmTmp.a[10]        =   1.548208e-01;
  parmTmp.a[11]        =  -4.162194e-03;
  parmTmp.a[12]        =  -2.619793e-03;
  parmTmp.a[13]        =   3.882013e-04;
  parmTmp.a[14]        =  -1.539223e-05;
  parmTmp.a[15]        =  -4.419453e-03;
  parmTmp.a[16]        =   3.719907e-04;
  parmTmp.a[17]        =   3.437933e-05;
  parmTmp.a[18]        =  -8.063501e-06;
  parmTmp.a[19]        =   3.531830e-07;
  parmTmp.a[20]        =   4.979933e-05;
  parmTmp.a[21]        =  -7.275302e-06;
  parmTmp.a[22]        =   1.978724e-07;
  parmTmp.a[23]        =   4.343684e-08;
  parmTmp.a[24]        =  -2.565570e-09;
  _parm.push_back(parmTmp);


  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          4;
  parmTmp.uMin         =   4.518525;
  parmTmp.uMax         =  41.679975;
  parmTmp.wMin         =   0.000000;
  parmTmp.wMax         =  16.000000;

  parmTmp.a[ 0]        =  -1.295865e+00;
  parmTmp.a[ 1]        =  -1.010980e-01;
  parmTmp.a[ 2]        =   8.809873e-03;
  parmTmp.a[ 3]        =   7.817337e-03;
  parmTmp.a[ 4]        =  -4.831934e-04;
  parmTmp.a[ 5]        =  -1.694104e+00;
  parmTmp.a[ 6]        =   2.236751e-02;
  parmTmp.a[ 7]        =   2.345840e-02;
  parmTmp.a[ 8]        =  -3.424598e-03;
  parmTmp.a[ 9]        =   1.347881e-04;
  parmTmp.a[10]        =   8.668734e-02;
  parmTmp.a[11]        =  -5.002464e-03;
  parmTmp.a[12]        =  -1.087785e-03;
  parmTmp.a[13]        =   1.958418e-04;
  parmTmp.a[14]        =  -8.026536e-06;
  parmTmp.a[15]        =  -2.433874e-03;
  parmTmp.a[16]        =   2.732863e-04;
  parmTmp.a[17]        =   1.201382e-05;
  parmTmp.a[18]        =  -4.301531e-06;
  parmTmp.a[19]        =   1.938647e-07;
  parmTmp.a[20]        =   2.707948e-05;
  parmTmp.a[21]        =  -4.458861e-06;
  parmTmp.a[22]        =   1.202472e-07;
  parmTmp.a[23]        =   2.801530e-08;
  parmTmp.a[24]        =  -1.571718e-09;
  _parm.push_back(parmTmp);


  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =          8;
  parmTmp.uMin         =   4.518524;
  parmTmp.uMax         =  41.679975;
  parmTmp.wMin         =   0.000000;
  parmTmp.wMax         =  16.000000;

  parmTmp.a[ 0]        =   2.961592e+00;
  parmTmp.a[ 1]        =  -2.309011e-01;
  parmTmp.a[ 2]        =  -4.528241e-03;
  parmTmp.a[ 3]        =   7.400383e-03;
  parmTmp.a[ 4]        =  -3.344016e-04;
  parmTmp.a[ 5]        =  -3.061980e+00;
  parmTmp.a[ 6]        =   1.035726e-01;
  parmTmp.a[ 7]        =  -1.528259e-03;
  parmTmp.a[ 8]        =  -3.763365e-04;
  parmTmp.a[ 9]        =   2.084924e-05;
  parmTmp.a[10]        =   1.640155e-01;
  parmTmp.a[11]        =  -1.333918e-02;
  parmTmp.a[12]        =   8.412875e-04;
  parmTmp.a[13]        =  -1.475054e-05;
  parmTmp.a[14]        =  -5.689474e-07;
  parmTmp.a[15]        =  -4.777251e-03;
  parmTmp.a[16]        =   6.345356e-04;
  parmTmp.a[17]        =  -5.590537e-05;
  parmTmp.a[18]        =   2.197438e-06;
  parmTmp.a[19]        =  -2.087985e-08;
  parmTmp.a[20]        =   5.462676e-05;
  parmTmp.a[21]        =  -1.003073e-05;
  parmTmp.a[22]        =   1.054773e-06;
  parmTmp.a[23]        =  -5.102477e-08;
  parmTmp.a[24]        =   8.365401e-10;
  _parm.push_back(parmTmp);


  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         12;
  parmTmp.uMin         =   5.024395;
  parmTmp.uMax         =  41.679975;
  parmTmp.wMin         =   0.000000;
  parmTmp.wMax         =  16.000000;

  parmTmp.a[ 0]        =   1.047494e+01;
  parmTmp.a[ 1]        =  -1.750536e+00;
  parmTmp.a[ 2]        =   1.609530e-01;
  parmTmp.a[ 3]        =  -4.733715e-03;
  parmTmp.a[ 4]        =   2.591859e-05;
  parmTmp.a[ 5]        =  -4.797613e+00;
  parmTmp.a[ 6]        =   5.562272e-01;
  parmTmp.a[ 7]        =  -5.112157e-02;
  parmTmp.a[ 8]        =   2.277314e-03;
  parmTmp.a[ 9]        =  -3.344025e-05;
  parmTmp.a[10]        =   2.829486e-01;
  parmTmp.a[11]        =  -5.718754e-02;
  parmTmp.a[12]        =   5.519079e-03;
  parmTmp.a[13]        =  -2.529124e-04;
  parmTmp.a[14]        =   4.180291e-06;
  parmTmp.a[15]        =  -9.104205e-03;
  parmTmp.a[16]        =   2.378222e-03;
  parmTmp.a[17]        =  -2.441052e-04;
  parmTmp.a[18]        =   1.174781e-05;
  parmTmp.a[19]        =  -2.128420e-07;
  parmTmp.a[20]        =   1.157623e-04;
  parmTmp.a[21]        =  -3.390051e-05;
  parmTmp.a[22]        =   3.671145e-06;
  parmTmp.a[23]        =  -1.830644e-07;
  parmTmp.a[24]        =   3.454851e-09;
  _parm.push_back(parmTmp);


  parmTmp.BsplineOrder =         16;
  parmTmp.Nder         =         16;
  parmTmp.uMin         =   5.753095;
  parmTmp.uMax         =  41.679975;
  parmTmp.wMin         =   0.000000;
  parmTmp.wMax         =  16.000000;

  parmTmp.a[ 0]        =   1.165828e+01;
  parmTmp.a[ 1]        =   1.368643e+00;
  parmTmp.a[ 2]        =  -3.561532e-01;
  parmTmp.a[ 3]        =   2.601190e-02;
  parmTmp.a[ 4]        =  -3.929398e-04;
  parmTmp.a[ 5]        =  -3.801066e+00;
  parmTmp.a[ 6]        =  -4.991591e-01;
  parmTmp.a[ 7]        =   1.199324e-01;
  parmTmp.a[ 8]        =  -8.790342e-03;
  parmTmp.a[ 9]        =   1.772532e-04;
  parmTmp.a[10]        =   8.318394e-02;
  parmTmp.a[11]        =   5.765291e-02;
  parmTmp.a[12]        =  -1.245546e-02;
  parmTmp.a[13]        =   9.145157e-04;
  parmTmp.a[14]        =  -2.032327e-05;
  parmTmp.a[15]        =   1.213179e-03;
  parmTmp.a[16]        =  -2.388535e-03;
  parmTmp.a[17]        =   4.811327e-04;
  parmTmp.a[18]        =  -3.482707e-05;
  parmTmp.a[19]        =   7.996837e-07;
  parmTmp.a[20]        =  -4.121201e-05;
  parmTmp.a[21]        =   3.215374e-05;
  parmTmp.a[22]        =  -6.170257e-06;
  parmTmp.a[23]        =   4.402799e-07;
  parmTmp.a[24]        =  -1.025361e-08;
  _parm.push_back(parmTmp);





  return true;
}

}//end namespace NPME_Library



