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

#ifndef NPME_B_SPLINE_H
#define NPME_B_SPLINE_H



namespace NPME_Library
{
const size_t NPME_Bspline_MaxOrder = 40;

//calculates B-splines of order n Bn(w) as an explicit piecewise polynomial of w
//and its first derivative
double NPME_Bspline_B (const int n, const double w);
//returns Bn(w)


double NPME_Bspline_dBdw (double& dBdw, const int n, const double w);
//returns Bn(w) and dBdw(n)


}//end namespace NPME_Library



#endif // NPME_B_SPLINE_H



