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

#ifndef NPME_REC_SUM_V_H
#define NPME_REC_SUM_V_H




namespace NPME_Library
{
void NPME_RecSumV_CalcV1 (double *V1, _Complex double *theta, 
  const size_t nCharge, const double *coord, 
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int n1, const long int n2, const long int n3,
  const long int BnOrder, const int nProc,
  const size_t nRecBlockTot, const long int *RecBlock2BlockIndex,
  const size_t *nRecBlock2Charge, const size_t **RecBlock2Charge,
  const double R0[3], int outputFormat, bool PRINT, std::ostream& os);
void NPME_RecSumV_CalcV1 (_Complex double *V1, _Complex double *theta, 
  const size_t nCharge, const double *coord, 
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int n1, const long int n2, const long int n3,
  const long int BnOrder, const int nProc,
  const size_t nRecBlockTot, const long int *RecBlock2BlockIndex,
  const size_t *nRecBlock2Charge, const size_t **RecBlock2Charge,
  const double R0[3], int outputFormat, bool PRINT, std::ostream& os);
//V1[nCharge][4] is not initialized to zero
//input:  nCharge, coord[3*nCharge], BnOrder = Bspline order
//        N1, N2, N3 = DFT Sizes
//        L1, L2, L3 = DFT box size
//        n1, n2, n3 = block sizes
//        n1, n2, n3 >= BnOrder
//        theta[N1*N2*N3/8] = theta potential assumed already calculated
//        nBlock2Charge[nBlock], Block2Charge[nBlock][], Block2V[nBlock]
//        nBlock = M1*M2*M3 (M1 = N1/2/n1, M2 = N2/2/n2, M3 = N3/2/n3)
//        R0[3] = center of coord
//output: V1[nCharge][4]  (outputFormat = 1)
//        V1[4][nCharge]  (outputFormat = 2)
//for a given charge, there are 4 components for (V, dVdr[3])








void NPME_RecSumV_CalcV1_Basic (_Complex double *V1, const size_t nCharge, 
  const double *coord, const _Complex double *theta,
  const long int N1, const long int N2, const long int N3, 
  const double   L1, const double   L2,   const double L3,
  const long int BnOrder, const double R0[3]);
//input:  coord[3*nCharge]
//        theta[N1*N2*N3/8]
//output: V1[4*nCharge] = {V[0], dVdx[0], dVdy[0], dVdz[0], 
//                         V[1], dVdx[1], dVdy[1], dVdz[1],..}

void NPME_RSTP_CalcLamda (const long int N1, const long int BnOrder, 
  _Complex double *lamda1, double *lamda1_2);


void NPME_RSTP_CalcTheta (
  _Complex double *T, _Complex double *X, 
  const _Complex double *F, const _Complex double *Qr,
  const long int N1, const long int N2, const long int N3,
  const _Complex double *C1, 
  const _Complex double *C2, 
  const _Complex double *C3,
  const double *lamda1_2, 
  const double *lamda2_2, 
  const double *lamda3_2,
  bool useSphereSymmCompactF,
  int nProc, bool PRINT, std::ostream& os);
//input:  Qr[N1*N2*N3/8]  = real compact Q
//        if useSphereSymmCompactF == 1
//          F [sizeF]     = spherically symmetric compact F
//                          sizeF = (N1/2+1)*(N2/2+1)*(N3/2+1)
//        else
//          F [N1*N2*N3]  = full FFT of F
//
//        a1, a2, a3      = 0, 1 (0 for even, 1 for odd)
//        X [N1*N2*N3/8]  = compact temp array
//
//output: T [N1*N2*N3/8]  = compact FFT of even/odd contribution to theta



void NPME_RSTP_CalcTheta_FullF_ReverseSign (
  _Complex double *T, _Complex double *X, 
  const _Complex double *F, const _Complex double *Qr,
  const long int N1, const long int N2, const long int N3,
  const _Complex double *C1, 
  const _Complex double *C2, 
  const _Complex double *C3,
  const double *lamda1_2, 
  const double *lamda2_2, 
  const double *lamda3_2,
  bool reverseSignX, bool reverseSignY, bool reverseSignZ, 
  int nProc, bool PRINT, std::ostream& os);
//input:  Qr[N1*N2*N3/8]  = real compact Q
//          F [N1*N2*N3]  = full FFT of F for translated helm potential for 
//                          R0[3] = {X0, Y0, Z0}
//        reverseSignX    = 1 uses F corresponding to R0[3] = {-X0, Y0, Z0}
//        X [N1*N2*N3/8]  = compact temp array
//output: T [N1*N2*N3/8]  = compact FFT of even/odd contribution to theta



void NPME_RSTP_CalcT_FullF_Contribution (
  _Complex double *X, const _Complex double *F, 
  const long int N1,      const long int N2,      const long int N3,
  const long int a1,      const long int a2,      const long int a3,
  const _Complex double *C1, 
  const _Complex double *C2, 
  const _Complex double *C3,
  const double *lamda1_2, 
  const double *lamda2_2, 
  const double *lamda3_2,
  int nProc);
//input:  X [N1*N2*N3/8]  = initialized with compact even/odd FFT(Q) contrib.
//        F [N1*N2*N3]    = full FFT of F
//        a1, a2, a3      = 0, 1 (0 for even, 1 for odd)
//        C1[N1/2]        = cexp(-2*NPME_Pi*I/N1*m1)  for -N1/4 <= m1 <= N1/4 - 1
//        C2[N2/2]        = cexp(-2*NPME_Pi*I/N2*m2)  for -N2/4 <= m2 <= N2/4 - 1
//        C3[N3/2]        = cexp(-2*NPME_Pi*I/N3*m3)  for -N3/4 <= m3 <= N3/4 - 1
//        lamda1_2[N1]
//        lamda2_2[N2]
//        lamda3_2[N3]
//output: X [N1*N2*N3/8]  = compact even/odd contribution to theta


void NPME_RSTP_CalcT_CompactSphereSymmF_Contribution (
  _Complex double *X, const _Complex double *F, 
  const long int N1,      const long int N2,      const long int N3,
  const long int a1,      const long int a2,      const long int a3,
  const _Complex double *C1, 
  const _Complex double *C2, 
  const _Complex double *C3,
  const double *lamda1_2, 
  const double *lamda2_2, 
  const double *lamda3_2,
  int nProc);
//input:  X [N1*N2*N3/8]  = initialized with compact even/odd FFT(Q) contrib.
//        F [sizeF]       = full FFT of F, sizeF = (N1/2+1)*(N2/2+1)*(N3/2+1)
//        a1, a2, a3      = 0, 1 (0 for even, 1 for odd)
//        C1[N1/2]        = cexp(-2*NPME_Pi*I/N1*m1)  for -N1/4 <= m1 <= N1/4 - 1
//        C2[N2/2]        = cexp(-2*NPME_Pi*I/N2*m2)  for -N2/4 <= m2 <= N2/4 - 1
//        C3[N3/2]        = cexp(-2*NPME_Pi*I/N3*m3)  for -N3/4 <= m3 <= N3/4 - 1
//        lamda1_2[N1]
//        lamda2_2[N2]
//        lamda3_2[N3]
//output: X [N1*N2*N3/8]  = compact even/odd contribution to theta

}//end namespace NPME_Library


#endif // NPME_REC_SUM_V_H



