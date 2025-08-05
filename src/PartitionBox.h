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

#ifndef NPME_PARTITION_BOX_H
#define NPME_PARTITION_BOX_H


namespace NPME_Library
{
//******************************************************************************
//******************************************************************************
//*******************************NPME_CellInterface*****************************
//******************************************************************************
//******************************************************************************

struct NPME_CellPairInteract
//includes self-interactions: cellIndex1 == cellIndex2
//and      pair-interactions: cellIndex1 != cellIndex2
{
  size_t cellIndex1, nPointPerCell1, startPointIndex1;
  size_t cellIndex2, nPointPerCell2, startPointIndex2; 
};
bool NPME_CellInterface (
  std::vector<NPME_Library::NPME_CellPairInteract>& interactList,
  std::vector<size_t>& P, std::vector<double>& coordPermute, 
  const size_t nPoint, const double *coord0, const size_t nNeigh, 
  const double Rdir, const char interactOpt, const int nProc);
//input:  nPoint, coord0[3*nPoint], nNeigh, Rdir
//        nPoint        = total number of points of the system
//        coord0        = input coordinates
//        Rdir          = direct space cutoff
//        nNeigh        = number of adjacent interacting neighboring cells where 
//                        cellSize = Rdir/nNeigh  (e.g. nNeigh = 2)
//        interactOpt   = interactionList option = 'A', 'B', or 'C'
//                        'A' = least pruned cell-cell list
//                        'B' = medium cell-cell pruning
//                        'C' = aggressive cell-cell pruning (longer SetUp)
//output: NPME_ClusterPair[nCluster], coordPermute[3*nPoint], P[nPoint]
//        coordPermute      = permuted set of coordinates
//        P                 = permutation operator to permute the coordinates
//                            which fit neatly into the arranged cells
//        interactList      = contains nPointPerCell and pointStartIndex for 
//                            each cell-cell pair contained in the cluster pair





//******************************************************************************
//******************************************************************************
//******************************NPME_PointIntoCell******************************
//******************************************************************************
//******************************************************************************
struct NPME_CellData
{
  size_t pointStartIndex;   
  size_t nPointPerCell;
  double cellCoord[3];
  size_t totCellIndex;
};

bool NPME_GetBoxSize (double& L1, double& L2, double& L3,
  double trans[3], const size_t nPoint, const double *coord);
//input:  nPoint, coord[3*nPoint]
//output: L1, L2, L3 specifying a rectangular volume containing the points
//coord'[] = coord[] + trans[]  where coord'[] = x, y, z and
//0 <= x <= L1
//0 <= y <= L2
//0 <= z <= L3

bool NPME_PointIntoCell (
  std::vector<NPME_CellData>& cell,
  size_t& nCell1D_1, size_t& nCell1D_2, size_t& nCell1D_3, size_t *P,
  const double cellSize, const double L1, const double L2, const double L3,
  const double trans[3], const size_t nPoint, const double *coord);
//input:  nPoint, coord[3*nPoint], L1, L2, L3 specifying a rectangular volume 
//        containing the points s.t.
//        coord'[] = coord[] + trans[]  where coord'[] = x, y, z and
//        0 <= x <= L1
//        0 <= y <= L2
//        0 <= z <= L3
//output: cell[nOccupyCell] = contains number of nPointPerCell, pointStartIndex,
//                            cellCoord, and totCellIndex for each 
//                            non-empty cell
//        P[nPoint]         = permutation matrix which puts the coordinates 
//                            neatly into each cell in an order
//        nTotCell          = nCell1D_1*nCell1D_2*nCell1D_3 
//                            nTotCell includes empty cells

bool NPME_CheckCoordInCell (const size_t nPoint, const double *coordPermute,
  const std::vector<NPME_Library::NPME_CellData>& cell, const double cellSize,
  double& maxDiff);
//input: coordPermute[nPoint]
//checks permuted coordinates are inside the assigned cell


//******************************************************************************
//******************************************************************************
//*************************NPME_FindInteractionList*****************************
//******************************************************************************
//******************************************************************************

bool NPME_FindInteractionList (const char option,
  std::vector<NPME_Library::NPME_CellPairInteract>& interactList,
  const size_t nNeigh, std::vector<NPME_Library::NPME_CellData>& cell,
  const double cellSize, const double *coordPerm, 
  const size_t nCell1D_X, const size_t nCell1D_Y, const size_t nCell1D_Z,
  int nProc);
//input:  cell[nOccupCell] = number of non-empty occupied cells
//        number of total cells = nCell1D_X*nCell1D_Y*nCell1D_Z
//        coordPerm[nPoint*3]   = permuted coordinates of points
//        option                = 'A', 'B', or 'C'
//            'A' = Simplest Least Pruning Algorithm
//            'B' = Medium Pruning Algorithm
//            'C' = Most Agressive Pruning Algorithm (Slow SetUp but 
//                                                    faster Direct Sum)
//output: interactList[nInteract]




//******************************************************************************
//******************************************************************************
//********************Test NPME_FindInteractionList*****************************
//******************************************************************************
//******************************************************************************

bool NPME_PointIntoCell_CheckCoordinates (const size_t i,
  const double x,  const double y,  const double z,
  const double L1, const double L2, const double L3);
bool NPME_PointIntoCell_CheckCellIndex (const size_t i,
  const size_t n1, const size_t n2, const size_t n3,
  const size_t N1, const size_t N2, const size_t N3);
void NPME_TotCellIndex2CellCoord (double cellCoord[3], 
  const size_t totCellIndex, const double cellSize, const double trans[3],
  const size_t nCell1D_2, const size_t nCell1D_3);

double NPME_GaussTestFunctionExact (const size_t nPoint, const double *coord,
  const double *charge, const double alpha, const int nProc);
double NPME_GaussTestFunctionModel (
  const double *coordPerm, const double *chargePerm, const double alpha, 
  const std::vector<NPME_Library::NPME_CellPairInteract>& interactList,
  const int nProc);
//input:  coordPerm[nPoint*3] = permuted coordinates
//        chargePerm[nPoint]  = permuted charges
//        alpha
//        interactList[nInteract]
//output: Gaussian charge energy

size_t NPME_CountNumDirectInteract (
  const std::vector<NPME_Library::NPME_CellPairInteract>& interactList);



}//end namespace NPME_Library


#endif // NPME_PARTITION_BOX_H


