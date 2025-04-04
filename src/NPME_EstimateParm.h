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

#ifndef NPME_ESTIMATE_PARM_H
#define NPME_ESTIMATE_PARM_H

#include <iostream> 
#include <fstream> 
#include <vector>


namespace NPME_Library
{
//  FFT grid sizes (Nfft) are obtained for static kernels f(r) = r^alpha
//  using a model similar to the one proposed by Kolafa and Perram (Mol. Sim. 
//  9, 351 (1991)).  
//I) Static Model f(r) = r^alpha
//  where 
//      alpha   = -7.0, -6.0, .. -1.0, 1.0
//      Bspline = 4, 6, 8, 12, 16
//      Nder    = Number of derivatives in the Derivative Matching 
//                Ewald Splitting method
//  For a cubic volume, the rec sum error model is defined by
//      eps_V = Q*pow(Rdir,alpha)*pow(Rdir/L, 1.5)*theta(u)
//  where
//      eps_V = absolute error in potential
//      Q     = sqrt(sum over q[i]^2)
//      Rdir  = direct space cutoff
//      L     = length in 1D
//      u     = Nfft*Rdir/L
//      theta = exp(P(u))
//      P(u)  = a[0] + a[1]*u + a[2]*u^2 + a[3]*u^3 + a[4]*u^4
//    an empirical function for P(u) is constructed for integer
//    alpha = -7, -6, .. -1, 1 and BsplineOrder = 8, 12, 16
//    Given a desired tolerance in eps_V, Nfft can be solved for numerically  
//    Caveats:
//      1) For anisotropic volumes, L1, L2, L3, separate Q factors are 
//         constructed as 
//              Q1 = Q*sqrt(L1^3/(L1*L2*L3))
//              Q2 = Q*sqrt(L2^3/(L1*L2*L3))
//              Q3 = Q*sqrt(L3^3/(L1*L2*L3))
//         Using the same desired precision eps_V with different
//         Q1, Q2, Q3 factors, separate variables u1, u2, u3 are obtained 
//         leading to separate FFT sizes in each direction
//      2) Given the physical volume (X0, Y0, Z0), the FFT size Nfft is first 
//         needed to find the PME corrected volume (X, Y, Z) which then leads to
//         to the FFT periods (L1, L2, L3).  Given, (L1, L2, L3), an improved
//         estimate of FFT sizes are made.  This process is performed 
//         iteratively with a minimal starting FFT size of 100.
//
//
//II) Helmholtz Model f(r) = exp(I*k0*r)/r for (Bspline = 4, 6, 8, 12, 16)
//  For a cubic volume, the rec sum error model is defined by
//      eps_V = Q/Rdir*pow(Rdir/L, 1.5)*theta(u,w)
//  where
//      eps_V   = absolute error in potential
//      Q       = sqrt(sum over |q[i]|^2)
//      Rdir    = direct space cutoff
//      L       = length in 1D
//      u       = Nfft*Rdir/L
//      w       = k0*Rdir = (dimensionless plane wave)
//      theta   = exp(P(u,w))
//      P(u,w)  = 4th degree polynomial of u,w
//              = a[0][0] + a[0][1]*w + a[1][1]*u*w +.. a[4][4]*u^4*w^4
//  P(u,w) is a polynomial with empirically determined coefficients
//  3 separate sets of coefficients are developed for Bspline = 8, 12, 16


//I) Static Model
struct NPME_ThetaParmStatic
{
  int Nder;
  double uMin, uMax;
  long int BsplineOrder;
  double alpha;
  double a[5];
};

class NPME_EstimateParmSingleBox_StaticModel
{
public:
  NPME_EstimateParmSingleBox_StaticModel ()
  {
    _isSet = 0;
    SetParm ();
  }


  bool Get_FFT_outputNder (
    long int& N1, long int& N2, long int& N3, 
    long int& n1, long int& n2, long int& n3,
    int& Nder,
    const size_t nCharge, const double *charge, const double *coord, 
    const long int BsplineOrder, const double alpha, 
    const double tol, const double Rdir, bool printLog, std::ostream& os) const;
  //input:  BsplineOrder, alpha
  //        charge, coord, tol, Rdir
  //output: Nder = number of derivatives in DM
  //        FFT parameters  (N1, N2, N3)
  //        FFT block sizes (n1, n2, n3)

  bool Get_FFT_inputNder (
    long int& N1, long int& N2, long int& N3, 
    long int& n1, long int& n2, long int& n3,
    const size_t nCharge, const double *charge, const double *coord, 
    const long int BsplineOrder, const double alpha, const int Nder, 
    const double tol, const double Rdir, bool printLog, std::ostream& os) const;
  //input:  BsplineOrder, alpha, Nder = number of derivatives in DM
  //        charge, coord, tol, Rdir
  //output: FFT parameters  (N1, N2, N3)
  //        FFT block sizes (n1, n2, n3)

  bool CalcPredError_V (
    double& eps_V, bool& reliablePred,
    const long int N1, const long int N2, const long int N3, 
    const size_t nCharge, const double *charge, const double *coord, 
    const long int BsplineOrder, const double alpha, const int Nder, 
    const double Rdir, bool printLog, std::ostream& os);
  //input:  BsplineOrder, alpha, Nder = number of derivatives in DM
  //        charge, coord, tol, Rdir
  //        FFT parameters  (N1, N2, N3)
  //output: 
  //  eps_V         = predicted model error 
  //  reliablePred  = true/false if the predicition is reliable

  const std::vector<NPME_ThetaParmStatic>& GetParmVector ()  const
  {
    return _parm;
  }
  

private:
  bool SetParm ();

  bool GetParm (double& uMin, double& uMax, 
    double a[5], const long int BsplineOrder, 
    const double alpha, const int Nder) const;

  bool GetParmIndex (size_t& parmIndex,
      const long int BsplineOrder, const double alpha, const int Nder) const;

  bool GetParmIndexList (std::vector<size_t>& parmIndexList,
      const long int BsplineOrder, const double alpha) const;


  bool Get_FFT_inputParmIndex (
    long int& N1, long int& N2, long int& N3, 
    long int& n1, long int& n2, long int& n3,
    const size_t nCharge, const double *charge, const double *coord, 
    const size_t parmIndex,
    const double tol, const double Rdir, bool printLog, std::ostream& os) const;


  bool _isSet;
  std::vector<NPME_ThetaParmStatic> _parm;
};

//II) Helmholtz Model
struct NPME_ThetaParmHelmholtz
{
  int Nder;
  double uMin, uMax;
  double wMin, wMax;
  long int BsplineOrder;
  double a[25];  //a[K+1][K+1]
};

class NPME_EstimateParmSingleBox_HelmholtzModel
{
public:
  NPME_EstimateParmSingleBox_HelmholtzModel ()
  {
    _isSet = 0;
    SetParm ();
  }



  bool Get_FFT_outputNder (
    long int& N1, long int& N2, long int& N3, 
    long int& n1, long int& n2, long int& n3,
    int& Nder,
    const size_t nCharge, const _Complex double *charge, const double *coord, 
    const long int BsplineOrder, const _Complex double k0, const double tol,
    const double Rdir, bool printLog, std::ostream& os) const;
  //input:  BsplineOrder, 
  //        charge, coord, k0, tol, Rdir
  //output: Nder = number of derivatives in DM
  //        FFT parameters  (N1, N2, N3)
  //        FFT block sizes (n1, n2, n3)

  bool Get_FFT_inputNder (
    long int& N1, long int& N2, long int& N3, 
    long int& n1, long int& n2, long int& n3,
    const size_t nCharge, const _Complex double *charge, const double *coord, 
    const long int BsplineOrder, const _Complex double k0, const int Nder, 
    const double tol, const double Rdir, bool printLog, std::ostream& os) const;
  //input:  BsplineOrder, Nder = number of derivatives in DM
  //        charge, coord, k0, tol, Rdir
  //output: FFT parameters  (N1, N2, N3)
  //        FFT block sizes (n1, n2, n3)

  bool CalcPredError_V (
    double& eps_V, bool& reliablePred,
    const long int N1, const long int N2, const long int N3, 
    const size_t nCharge, const _Complex double *charge, const double *coord, 
    const long int BsplineOrder, const _Complex double k0, const int Nder, 
    const double Rdir, bool printLog, std::ostream& os);
  //input:  BsplineOrder, k0, Nder = number of derivatives in DM
  //        charge, coord, tol, Rdir
  //        FFT parameters  (N1, N2, N3)
  //output: 
  //  eps_V         = predicted model error 
  //  reliablePred  = true/false if the predicition is reliable

  const std::vector<NPME_ThetaParmHelmholtz>& GetParmVector ()  const
  {
    return _parm;
  }

private:
  bool SetParm ();
  bool GetParm (double& uMin, double& uMax, 
        double& wMin, double& wMax, 
        double a[25], 
        const long int BsplineOrder, const int Nder) const;


  bool GetParmIndex (size_t& parmIndex,
      const long int BsplineOrder, const int Nder) const;

  bool GetParmIndexList (std::vector<size_t>& parmIndexList,
      const long int BsplineOrder) const;

  bool Get_FFT_inputParmIndex (
    long int& N1, long int& N2, long int& N3, 
    long int& n1, long int& n2, long int& n3,
    const size_t nCharge, const _Complex double *charge, const double *coord, 
    const size_t parmIndex,
    const _Complex double k0, const double tol,
    const double Rdir, bool printLog, std::ostream& os) const;

  bool _isSet;
  std::vector<NPME_ThetaParmHelmholtz> _parm;
};

double NPME_EstimateParmCalcQsum (const size_t nCharge, 
  const double *charge);
double NPME_EstimateParmCalcQsum (const size_t nCharge, 
  const _Complex double *charge);

}//end namespace NPME_Library


#endif // NPME_ESTIMATE_PARM_H



