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

#ifndef NPME_PARTITION_EMBEDDED_BOX_H
#define NPME_PARTITION_EMBEDDED_BOX_H


#include "NPME_PartitionBox.h"

//the main interface function in NPME_PartitionBox.h can be used to partition
//space into cells of size Rdir/nNeigh and get cell-cell interaction lists
//however, the number of points in a given cell may be small.  this will
//cause computational inefficiency while updating the total direct sum potential

//the main interface function in NPME_PartitionEmbeddedBox.h can be used
//to group nearby cells into clusters and output cluster-cluster interaction
//lists containing pairs of interacting cells.  each cluster-cluster update
//is larger, but there are fewer of them leading to better computational 
//efficieny when updating the total direct sum potential


namespace NPME_Library
{
struct NPME_ClusterPair
{
  size_t maxNumPointPerCell;
  size_t cellIndexA1, pointStartA1, nPointPerCluster1;
  size_t cellIndexA2, pointStartA2, nPointPerCluster2;
  //nPointPerCluster1, nPointPerCluster2 reflect the sum of nPointPerCell
  //in pairB and not the values in NPME_CellData for cellA
  //similarly, pointStartA1, pointStartA2 reflect the smallest start index
  //in pairB and not the values in NPME_CellData for cellA

  std::vector<NPME_Library::NPME_CellPairInteract> pairB;
};


//******************************************************************************
//******************************************************************************
//**************************NPME_ClusterInterface*******************************
//******************************************************************************
//******************************************************************************
bool NPME_ClusterInterface (
  std::vector<NPME_Library::NPME_ClusterPair>& cluster,
  std::vector<size_t>& P, std::vector<double>& coordPermute, 
  size_t& nAvgChgPerCell,     size_t& maxChgPerCell,
  size_t& nAvgChgPerCluster,  size_t& maxChgPerCluster,
  size_t& nAvgCellPerCluster, size_t& maxCellPerCluster,
  const size_t nPoint, const double *coord0, const size_t nNeigh, 
  const double Rdir, const size_t nCellClust1D, const char interactOpt,
  const int nProc);
bool NPME_ClusterInterface (
  std::vector<NPME_Library::NPME_ClusterPair>& cluster,
  std::vector<size_t>& P, std::vector<double>& coordPermute, 
  const size_t nPoint, const double *coord0, const size_t nNeigh, 
  const double Rdir, const size_t nCellClust1D, const char interactOpt,
  const int nProc);
//input:  nPoint, coord0[3*nPoint], nNeigh, Rdir, nCellClust1D
//        nPoint        = total number of points of the system
//        coord0        = input coordinates
//        Rdir          = direct space cutoff
//        nNeigh        = number of adjacent interacting neighboring cells where 
//                        cellSize = Rdir/nNeigh  (e.g. nNeigh = 2)
//        nCellClust1D  = number of cells per cluster along 1 dimension
//                        (e.g. nCellClust = 2 - 4)
//        interactOpt   = interactionList option = 'A', 'B', or 'C'
//                        'A' = least pruned cell-cell list
//                        'B' = medium cell-cell pruning
//                        'C' = aggressive cell-cell pruning (longer SetUp)
//output: NPME_ClusterPair[nCluster], coordPermute[3*nPoint], P[nPoint]
//        coordPermute      = permuted set of coordinates
//        P                 = permutation operator to permute the coordinates
//                            which fit neatly into the arranged cells
//        NPME_ClusterPair  = contains nPointPerCluster, pointStartIndex
//                            for each cluster pair.  also contains 
//                            nPointPerCell and pointStartIndex for each 
//                            cell-cell pair contained in the cluster pair





//******************************************************************************
//******************************************************************************
//*******************Lower Level Cell and Cluster Functions*********************
//******************************************************************************
//******************************************************************************


bool NPME_PointIntoEmbeddedCell (
  std::vector<NPME_Library::NPME_CellData>& cellA,
  std::vector<NPME_Library::NPME_CellData>& cellB,
  std::vector<size_t>& cellBToCellAindex,
  size_t& nCellA_1D_X, size_t& nCellA_1D_Y, size_t& nCellA_1D_Z, 
  size_t& nCellB_1D_X, size_t& nCellB_1D_Y, size_t& nCellB_1D_Z, 
  size_t *P, const double cellSizeB, const size_t nPartition, 
  const double L1, const double L2, const double L3,
  const double trans[3], const size_t nPoint, const double *coord);
//input:  nPoint, coord[3*nPoint], L1, L2, L3 specifying a rectangular volume 
//        containing the points s.t.
//        coord'[] = coord[] + trans[]  where coord'[] = x, y, z and
//        0 <= x <= L1
//        0 <= y <= L2
//        0 <= z <= L3
//        cellSizeB = size of smaller cell
//        cellSizeA = size of larger cell = cellSizeB*nPartition
//        cellA[] contains cellB[]
//output: P[nPoint] = permutation matrix which puts the coordinates 
//        into each smaller cell in an order
//        cellA[nOccupCellA]
//        cellB[nOccupCellB]
//        nOccupCellA = number of non-empty occupied type 1 cells
//        nOccupCellB = number of non-empty occupied type 2 cells
//        nTotCellA   = nCellA_1D_X*nCellA_1D_Y*nCellA_1D_Z
//        nTotCellB   = nCellB_1D_X*nCellB_1D_Y*nCellB_1D_Z
//        nOccupCellB >= nOccupCellA
//        each cellA[] contains multiple cellB[] and 
//        cellBToCellAindex[nOccupCellB] gives cell A indexes for each cell B


size_t NPME_EmbeddedTotCellIndexToTotCellIndex (
  const size_t embeddedTotCellIndexB, const size_t nPartition, 
  const size_t nCellA_1D_Y, const size_t nCellA_1D_Z);
size_t NPME_TotCellIndexToEmbeddedTotCellIndex (const size_t totCellIndexB,
  const size_t nPartition, const size_t nCellA_1D_Y, const size_t nCellA_1D_Z);



//******************************************************************************
//******************************************************************************
//***************************NPME_Cluster (Cell A pairs)************************
//******************************************************************************
//******************************************************************************
bool NPME_ConstructClusterPair (const size_t nPoint,
  const std::vector<size_t>& cellBToCellAindex,
  const std::vector<NPME_Library::NPME_CellPairInteract>& interactPairCellB,
  std::vector<NPME_Library::NPME_ClusterPair>& cluster);
//input:  cellBToCellAindex[nOccupyB] = map from cell B to cell A containining B
//        interactPairCellB[]         = cell B - cell B pairs of cells
//output: cluster[]                   = the cell B - cell B pairs are matched to
//                                      the cell A - cell A pairs


double NPME_GaussTestFunctionModel (
  const double *coordPerm, const double *chargePerm, const double alpha, 
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster,
  const int nProc);
//input:  coordPerm[nPoint*3] = permuted coordinates
//        chargePerm[nPoint]  = permuted charges
//        alpha
//        interactList[nInteract]
//output: Gaussian charge energy

size_t NPME_CountNumDirectInteract (
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster);

}//end namespace NPME_Library


#endif // NPME_PARTITION_EMBEDDED_BOX_H


