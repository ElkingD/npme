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
#include <algorithm>


#include "Constant.h"
#include "PartitionBox.h"
#include "PartitionEmbeddedBox.h"
#include "SupportFunctions.h"
#include "PermuteArray.h"

#define NPME_PARTITION_EMBEDDED_BOX_DEBUG       1
#define NPME_PARTITION_EMBEDDED_BOX_DEBUG_PRINT 0

namespace NPME_Library
{
//from NPME_PartitionBox.cpp
size_t NPME_ConvertCoord2CellIndex (const double x, const double cellSize, 
  const size_t nCell1D);


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
  const int nProc)
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
{
  using std::vector;
  using std::cout;

  const double cellSizeB  = Rdir/nNeigh;
  const size_t nPartition = nCellClust1D;
  const double cellSizeA  = nPartition*cellSizeB;

  //1) total box size
  double L1, L2, L3;
  double trans[3];
  if (!NPME_GetBoxSize (L1, L2, L3, trans, nPoint, coord0))
  {
    cout << "Error in NPME_ClusterInterface\n";
    cout << "  NPME_GetBoxSize failed\n";
    return false;
  }

  //2) point into embedded cells
  vector<NPME_CellData> cellA;  //larger cells (clusters)
  vector<NPME_CellData> cellB;  //small  cells
  vector<size_t> cellBToCellAindex;
  size_t nCellA_1D_X, nCellA_1D_Y, nCellA_1D_Z;
  size_t nCellB_1D_X, nCellB_1D_Y, nCellB_1D_Z;

  P.resize(nPoint);
  coordPermute.resize(3*nPoint);

  if (!NPME_PointIntoEmbeddedCell (cellA, cellB, cellBToCellAindex,
        nCellA_1D_X, nCellA_1D_Y, nCellA_1D_Z,
        nCellB_1D_X, nCellB_1D_Y, nCellB_1D_Z,
        &P[0], cellSizeB, nPartition, 
        L1, L2, L3, trans, nPoint, coord0))
  {
    cout << "Error in NPME_ClusterInterface\n";
    cout << "  NPME_PointIntoEmbeddedCell failed\n";
    return false;
  }

  //calc statistics  (Cell = CellB and Cluster = CellA)
  nAvgChgPerCell    = 0;
  nAvgChgPerCluster = 0;
  maxChgPerCell     = 0;
  maxChgPerCluster  = 0;
  for (size_t i = 0; i < cellA.size(); i++)
  {
    nAvgChgPerCluster += cellA[i].nPointPerCell;
    if (maxChgPerCluster < cellA[i].nPointPerCell)
      maxChgPerCluster = cellA[i].nPointPerCell;
  }
  nAvgChgPerCluster /= cellA.size();

  for (size_t i = 0; i < cellB.size(); i++)
  {
    nAvgChgPerCell += cellB[i].nPointPerCell;
    if (maxChgPerCell < cellB[i].nPointPerCell)
      maxChgPerCell = cellB[i].nPointPerCell;
  }
  nAvgChgPerCell /= cellB.size();


  {
    vector<size_t> nCellPerCluster(cellA.size());
    for (size_t i = 0; i < cellA.size(); i++)
      nCellPerCluster[i] = 0;
    
    for (size_t i = 0; i < cellB.size(); i++)
      nCellPerCluster[cellBToCellAindex[i]]++;
  
    nAvgCellPerCluster = 0;
    maxCellPerCluster  = 0;
    for (size_t i = 0; i < cellA.size(); i++)
    {
      nAvgCellPerCluster += nCellPerCluster[i];
      if (maxCellPerCluster < nCellPerCluster[i])
        maxCellPerCluster = nCellPerCluster[i];
    }
    nAvgCellPerCluster /= cellA.size();
  }

  //3) permute coordinates to fit neatly into cellB
  NPME_PermuteArrayInverse_MN (3, nPoint, &P[0], &coordPermute[0],  &coord0[0]);

  #if NPME_PARTITION_EMBEDDED_BOX_DEBUG
  {
    double maxDiff;
    if (!NPME_CheckCoordInCell (nPoint, &coordPermute[0],
      cellA, cellSizeA, maxDiff))
    {
      cout << "Error in NPME_ClusterInterface\n";
      cout << "  NPME_CheckCoordInCell failed for cellA\n";
      return false;
    }
    if (!NPME_CheckCoordInCell (nPoint, &coordPermute[0],
      cellB, cellSizeB, maxDiff))
    {
      cout << "Error in NPME_ClusterInterface\n";
      cout << "  NPME_CheckCoordInCell failed for cellB\n";
      return false;
    }
  }
  #endif

  //4) find cellB - cellB interaction lists
  vector<NPME_CellPairInteract> interactList;
  if (!NPME_FindInteractionList (interactOpt, interactList,
        nNeigh, cellB, cellSizeB, &coordPermute[0], 
        nCellB_1D_X, nCellB_1D_Y, nCellB_1D_Z, nProc))
  {
    cout << "Error in NPME_ClusterInterface\n";
    cout << "  NPME_FindInteractionList failed\n";
    return false;
  }


  //5) find cluster - cluster interaction lists
  if (!NPME_ConstructClusterPair (nPoint, cellBToCellAindex,
        interactList, cluster))
  {
    cout << "Error in NPME_ClusterInterface\n";
    cout << "  NPME_ConstructClusterPair failed\n";
    return false;
  }



  return true;
}



bool NPME_ClusterInterface (
  std::vector<NPME_Library::NPME_ClusterPair>& cluster,
  std::vector<size_t>& P, std::vector<double>& coordPermute, 
  const size_t nPoint, const double *coord0, const size_t nNeigh, 
  const double Rdir, const size_t nCellClust1D, const char interactOpt,
  const int nProc)
{
  size_t nAvgChgPerCell,     maxChgPerCell;
  size_t nAvgChgPerCluster,  maxChgPerCluster;
  size_t nAvgCellPerCluster, maxCellPerCluster;

  return NPME_ClusterInterface (cluster, P, coordPermute, 
              nAvgChgPerCell,     maxChgPerCell, 
              nAvgChgPerCluster,  maxChgPerCluster,
              nAvgCellPerCluster, maxCellPerCluster,
              nPoint, coord0, nNeigh, Rdir, nCellClust1D, interactOpt, nProc);
}


//******************************************************************************
//******************************************************************************
//**************************NPME_ClusterInterface*******************************
//******************************************************************************
//******************************************************************************

//from NPME_PartitionBox.cpp
bool NPME_PointIntoCell_CheckCoordinates (const size_t i,
  const double x,  const double y,  const double z,
  const double L1, const double L2, const double L3);
bool NPME_PointIntoCell_CheckCellIndex (const size_t i,
  const size_t n1, const size_t n2, const size_t n3,
  const size_t N1, const size_t N2, const size_t N3);
void NPME_TotCellIndex2CellCoord (double cellCoord[3], 
  const size_t totCellIndex, const double cellSize, const double trans[3],
  const size_t nCell1D_Y, const size_t nCell1D_Z);
bool NPME_FindInteractionList_C_CloseContact (const double Rdir2,
  const size_t nPoint1, const double *crdTmp1,
  const size_t nPoint2, const double *crdTmp2);


struct NPME_SortedEmbeddedCell
{
  size_t pointIndex;
  size_t totCellIndexA;
  size_t totCellIndexB;
};

bool NPME_PointEmbeddedCellPair_SortCriteria
  (const NPME_SortedEmbeddedCell& r1, 
   const NPME_SortedEmbeddedCell& r2)
{
  if (r1.totCellIndexB < r2.totCellIndexB)
    return true;
  else 
    return false;
}



bool NPME_PointIntoEmbeddedCell (
  std::vector<NPME_Library::NPME_CellData>& cellA,
  std::vector<NPME_Library::NPME_CellData>& cellB,
  std::vector<size_t>& cellBToCellAindex,
  size_t& nCellA_1D_X, size_t& nCellA_1D_Y, size_t& nCellA_1D_Z, 
  size_t& nCellB_1D_X, size_t& nCellB_1D_Y, size_t& nCellB_1D_Z, 
  size_t *P, const double cellSizeB, const size_t nPartition, 
  const double L1, const double L2, const double L3,
  const double trans[3], const size_t nPoint, const double *coord)
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
{
  using std::cout;
  using std::sort;
  using std::vector;

  const double cellSizeA  = cellSizeB*nPartition;
  const size_t Nc         = nPartition*nPartition*nPartition;
  //Nc = total number of cellB in cellA


  nCellA_1D_X = (size_t) (L1/cellSizeA) + 1;
  nCellA_1D_Y = (size_t) (L2/cellSizeA) + 1;
  nCellA_1D_Z = (size_t) (L3/cellSizeA) + 1;

  nCellB_1D_X = nPartition*nCellA_1D_X;
  nCellB_1D_Y = nPartition*nCellA_1D_Y;
  nCellB_1D_Z = nPartition*nCellA_1D_Z;

  vector<NPME_SortedEmbeddedCell> sortCell(nPoint);
  
  #if NPME_PARTITION_EMBEDDED_BOX_DEBUG_PRINT
  {
    char str[2000];
    cout << "\n\nNPME_PointIntoEmbeddedCell\n";
    sprintf(str, "nCellA_1D_X = %4lu nCellA_1D_Y = %4lu nCellA_1D_Z = %4lu\n",
      nCellA_1D_X, nCellA_1D_Y, nCellA_1D_Z);
    cout << str;
    sprintf(str, "nCellB_1D_X = %4lu nCellB_1D_Y = %4lu nCellB_1D_Z = %4lu\n",
      nCellB_1D_X, nCellB_1D_Y, nCellB_1D_Z);
    cout << str;

    sprintf(str, "nTotCellA = %lu nTotCellB = %lu\n",
      nCellA_1D_X*nCellA_1D_Y*nCellA_1D_Z,
      nCellB_1D_X*nCellB_1D_Y*nCellB_1D_Z);
    cout << str;
  }
  #endif

  #if NPME_PARTITION_EMBEDDED_BOX_DEBUG
  {
    //calculate error in extended box dimensions
    double error =  fabs(nCellA_1D_X*cellSizeA - nCellB_1D_X*cellSizeB) + 
                    fabs(nCellA_1D_Y*cellSizeA - nCellB_1D_Y*cellSizeB) + 
                    fabs(nCellA_1D_Z*cellSizeA - nCellB_1D_Z*cellSizeB);
    if (error > 1.0E-12)
    {
      char str[2000];
      cout << "Error in NPME_PointIntoEmbeddedCell\n";
      sprintf(str, "error in extended box = %.2le\n", error);
      cout << str;
      exit(0);
    }
  }
  #endif

  for (size_t i = 0; i < nPoint; i++)
  {
    //translate coord[] s.t. 
    //        0 <= x <= L1
    //        0 <= y <= L2
    //        0 <= z <= L3
    const double *r  = &coord[3*i];
    const double x1  = r[0] + trans[0];
    const double y1  = r[1] + trans[1];
    const double z1  = r[2] + trans[2];

    //integers defining cellA location
    const size_t n1 = NPME_ConvertCoord2CellIndex (x1, cellSizeA, nCellA_1D_X);
    const size_t n2 = NPME_ConvertCoord2CellIndex (y1, cellSizeA, nCellA_1D_Y);
    const size_t n3 = NPME_ConvertCoord2CellIndex (z1, cellSizeA, nCellA_1D_Z);

    const size_t totCellIndexA  = NPME_ind3D (n1, n2, n3, 
                                    nCellA_1D_Y, nCellA_1D_Z);
    //totCellIndexA = ordinary total cell index

    //coordinates relative to cellA vertex closest to origin
    const double x2             = x1 - n1*cellSizeA;
    const double y2             = y1 - n2*cellSizeA;
    const double z2             = z1 - n3*cellSizeA;

    //integers defining cellB location
    const size_t m1 = NPME_ConvertCoord2CellIndex (x2, cellSizeB, nPartition);
    const size_t m2 = NPME_ConvertCoord2CellIndex (y2, cellSizeB, nPartition);
    const size_t m3 = NPME_ConvertCoord2CellIndex (z2, cellSizeB, nPartition);

    const size_t locCellIndex2  = NPME_ind3D (m1, m2, m3, 
                                      nPartition, nPartition);
    const size_t totCellIndexB  = totCellIndexA*Nc + locCellIndex2;
    //totCellIndexB = embedded (inside A) total cell index of B

    //integers defining cellB location
    #if NPME_PARTITION_EMBEDDED_BOX_DEBUG
    {
      bool error = 0;
      char str[2000];
      if (locCellIndex2 >= Nc)
      {
        char str[2000];
        cout << "Error in NPME_PointIntoEmbeddedCell.\n";
        sprintf(str, "locCellIndex2 = %lu >= %lu = Nc\n", locCellIndex2, Nc);
        cout << str;
        error = 1;
      }
      if (!NPME_PointIntoCell_CheckCoordinates (i, x1, y1, z1, L1, L2, L3))
      {
        cout << "Error in NPME_PointIntoEmbeddedCell.\n";
        cout << "NPME_PointIntoCell_CheckCoordinates failed for cell 1\n";
        error = 1;
      }

      if (!NPME_PointIntoCell_CheckCoordinates (i, x2, y2, z2, 
        cellSizeA, cellSizeA, cellSizeA))
      {
        cout << "Error in NPME_PointIntoEmbeddedCell.\n";
        cout << "NPME_PointIntoCell_CheckCoordinates failed for cell 2\n";
        error = 1;
      }
      if (!NPME_PointIntoCell_CheckCellIndex (i, n1, n2, n3, 
              nCellA_1D_X, nCellA_1D_Y, nCellA_1D_Z))
      {
        cout << "Error in NPME_PointIntoEmbeddedCell.\n";
        cout << "NPME_PointIntoCell_CheckCellIndex failed for cell 1\n";
        error = 1;
      }
      if (!NPME_PointIntoCell_CheckCellIndex (i, m1, m2, m3, 
              nPartition, nPartition, nPartition))
      {
        cout << "Error in NPME_PointIntoEmbeddedCell.\n";
        cout << "NPME_PointIntoCell_CheckCellIndex failed for cell 2\n";
        error = 1;
      }

      if (error)
      {
        sprintf(str, "cellSizeA = %f cellSizeB = %f\n", cellSizeA , cellSizeB);
        cout << str;
        sprintf(str, "x1 y1 z1 = %f %f %f\n", x1, y1, z1);  cout << str;
        sprintf(str, "n2 n2 n2 = %lu %lu %lu\n", n1, n2, n3);  cout << str;

        sprintf(str, "x2 y2 z2 = %f %f %f\n", x2, y2, z2);  cout << str;
        sprintf(str, "m2 m2 m2 = %lu %lu %lu\n", m1, m2, m3);  cout << str;
        sprintf(str, "nPartition = %lu\n", nPartition);         cout << str;
  
        return false;
      }
    }
    #endif

    sortCell[i].pointIndex     = i;
    sortCell[i].totCellIndexA  = totCellIndexA;
    sortCell[i].totCellIndexB  = totCellIndexB;
  }

  sort(sortCell.begin(), sortCell.end(), 
    NPME_PointEmbeddedCellPair_SortCriteria);

  #if NPME_PARTITION_EMBEDDED_BOX_DEBUG_PRINT
  {
    char str[2000];
    cout << "\nSorted Cells\n";
    bool PRINT_ALL = 1;

    for (size_t i = 0; i < nPoint; i++)
    {
      sprintf(str, "  point %4lu  totCellIndexA = %5lu totCellIndexB = %5lu\n",
        sortCell[i].pointIndex, sortCell[i].totCellIndexA,
        sortCell[i].totCellIndexB);
      if (PRINT_ALL)
        cout << str;

      //make sure both totCellIndexA are increasing
      if (i != nPoint - 1)
      {
        if (sortCell[i].totCellIndexA > sortCell[i+1].totCellIndexA)
        {
          cout << "Error in NPME_PointIntoEmbeddedCell.\n";
          cout << "  totCellIndexA is decreasing\n";
        }
        if (sortCell[i].totCellIndexB > sortCell[i+1].totCellIndexB)
        {
          cout << "Error in NPME_PointIntoEmbeddedCell.\n";
          cout << "  totCellIndexB is decreasing\n";
        }
      }        
    }
  }
  #endif

  #if NPME_PARTITION_EMBEDDED_BOX_DEBUG
  {
    for (size_t i = 0; i < nPoint; i++)
    {
      //make sure both totCellIndexA are increasing
      if (i != nPoint - 1)
      {
        if (sortCell[i].totCellIndexA > sortCell[i+1].totCellIndexA)
        {
          cout << "Error in NPME_PointIntoEmbeddedCell.\n";
          cout << "  totCellIndexA is decreasing\n";
        }
        if (sortCell[i].totCellIndexB > sortCell[i+1].totCellIndexB)
        {
          cout << "Error in NPME_PointIntoEmbeddedCell.\n";
          cout << "  totCellIndexB is decreasing\n";
        }
      }        
    }
  }
  #endif

  size_t nOccupCellA = 1;
  size_t nOccupCellB = 1;
  for (size_t i = 1; i < nPoint; i++)
  {
    if (sortCell[i].totCellIndexA != sortCell[i-1].totCellIndexA)
      nOccupCellA++;

    if (sortCell[i].totCellIndexB != sortCell[i-1].totCellIndexB)
      nOccupCellB++;
  }

  #if NPME_PARTITION_EMBEDDED_BOX_DEBUG_PRINT
  {
    char str[2000];
    sprintf(str, "nOccupCellA = %lu nOccupCellB = %lu\n", 
      nOccupCellA, nOccupCellB);
    cout << str;
  }
  #endif

  cellA.resize(nOccupCellA);
  cellB.resize(nOccupCellB);
  cellBToCellAindex.resize(nOccupCellB);


  size_t occupCellIndexA                        = 0;
  size_t occupCellIndexB                        = 0;

  for (size_t i = 0; i < nPoint; i++)
  {
    double r1[3];   //center of cell 1
    double r2[3];   //center of cell 2

    const size_t totCellIndexA = sortCell[i].totCellIndexA;
    const size_t totCellIndexB = sortCell[i].totCellIndexB;

    #if NPME_PARTITION_EMBEDDED_BOX_DEBUG
    {
      char str[2000];
      if (totCellIndexB < totCellIndexA*Nc)
      {
        cout << "Error in NPME_PointIntoEmbeddedCell.\n";
        sprintf(str, "totCellIndexB = %lu < %lu = totCellIndexA*Nc\n", 
          totCellIndexB, totCellIndexA*Nc);
        cout << str;
        return false;
      }
    }
    #endif

    //calculate centers of cells 1 and 2
    {
      NPME_TotCellIndex2CellCoord (r1, totCellIndexA,
        cellSizeA, trans, nCellA_1D_Y, nCellA_1D_Z);

      size_t m1, m2, m3;
      const size_t locCellIndex2 = totCellIndexB - totCellIndexA*Nc;
      NPME_ind3D_2_n1_n2_n3 (locCellIndex2, nPartition, nPartition, 
        m1, m2, m3);

      r2[0] = r1[0] - cellSizeA/2.0 + (m1 + 0.5)*cellSizeB;
      r2[1] = r1[1] - cellSizeA/2.0 + (m2 + 0.5)*cellSizeB;
      r2[2] = r1[2] - cellSizeA/2.0 + (m3 + 0.5)*cellSizeB;
    }

    P[ sortCell[i].pointIndex ] = i;

    if (i == 0)
    {
      cellA[occupCellIndexA].pointStartIndex        = 0;
      cellB[occupCellIndexB].pointStartIndex        = 0;

      cellA[occupCellIndexA].nPointPerCell          = 1;
      cellB[occupCellIndexB].nPointPerCell          = 1;

      cellA[occupCellIndexA].totCellIndex = sortCell[i].totCellIndexA;
      cellB[occupCellIndexB].totCellIndex = sortCell[i].totCellIndexB;

      for (int p = 0; p < 3; p++)
        cellA[occupCellIndexA].cellCoord[p] = r1[p];

      for (int p = 0; p < 3; p++)
        cellB[occupCellIndexB].cellCoord[p] = r2[p];
    }
    else
    {
      if (sortCell[i].totCellIndexA != sortCell[i-1].totCellIndexA)
      {
        occupCellIndexA++;

        cellA[occupCellIndexA].pointStartIndex  = i;
        cellA[occupCellIndexA].nPointPerCell    = 1;
        cellA[occupCellIndexA].totCellIndex     = sortCell[i].totCellIndexA;
        for (int p = 0; p < 3; p++)
          cellA[occupCellIndexA].cellCoord[p] = r1[p];
      }
      else
        cellA[occupCellIndexA].nPointPerCell++;

      if (sortCell[i].totCellIndexB != sortCell[i-1].totCellIndexB)
      {
        occupCellIndexB++;

        cellB[occupCellIndexB].pointStartIndex  = i;
        cellB[occupCellIndexB].nPointPerCell    = 1;
        cellB[occupCellIndexB].totCellIndex     = sortCell[i].totCellIndexB;
        for (int p = 0; p < 3; p++)
          cellB[occupCellIndexB].cellCoord[p] = r2[p];
      }
      else
        cellB[occupCellIndexB].nPointPerCell++;

      cellBToCellAindex[occupCellIndexB] = occupCellIndexA;
    }
  }

  #if NPME_PARTITION_EMBEDDED_BOX_DEBUG_PRINT
  {
    char str[2000];
    sprintf(str, "nOccupCellA = %lu nOccupCellB = %lu\n", 
      nOccupCellA, nOccupCellB);
    cout << str;

    for (size_t n = 0; n < nOccupCellA; n++)
    {
      sprintf(str, "  cellA %4lu nPoint = %5lu pointStartInd = %5lu  %8.4f %8.4f %8.4f\n", 
        n, cellA[n].nPointPerCell, cellA[n].pointStartIndex, 
        cellA[n].cellCoord[0], 
        cellA[n].cellCoord[1], 
        cellA[n].cellCoord[2]);
      cout << str;
    }
    for (size_t n = 0; n < nOccupCellB; n++)
    {
      sprintf(str, "  cellB %4lu nPoint = %5lu pointStartInd = %5lu  %.4f %.4f %.4f\n", 
        n, cellB[n].nPointPerCell, cellB[n].pointStartIndex, 
        cellB[n].cellCoord[0], 
        cellB[n].cellCoord[1], 
        cellB[n].cellCoord[2]);
      cout << str;
    }
  }
  #endif


  //now convert cellB[].totCellIndex from embedded total index to 
  //ordinary total cell index
  for (size_t i = 0; i < nOccupCellB; i++)
  {
    const size_t embeddedTotCellIndexB  = cellB[i].totCellIndex;

    cellB[i].totCellIndex  = NPME_EmbeddedTotCellIndexToTotCellIndex (
                                    embeddedTotCellIndexB, nPartition, 
                                    nCellA_1D_Y, nCellA_1D_Z);
  }


  return true;
}





size_t NPME_EmbeddedTotCellIndexToTotCellIndex (
  const size_t embeddedTotCellIndexB, const size_t nPartition, 
  const size_t nCellA_1D_Y, const size_t nCellA_1D_Z)
{
  const size_t Nc             = nPartition*nPartition*nPartition;
  const size_t locCellIndexB  = embeddedTotCellIndexB%Nc;
  const size_t totCellIndexA  = embeddedTotCellIndexB/Nc;
  const size_t nCellB_1D_Y    = nPartition*nCellA_1D_Y;
  const size_t nCellB_1D_Z    = nPartition*nCellA_1D_Z;

  //cell x, y, z indexes for cell1
  size_t n1, n2, n3;
  NPME_ind3D_2_n1_n2_n3 (totCellIndexA, nCellA_1D_Y, nCellA_1D_Z, n1, n2, n3);

  //local cell x, y, z indexes for cell2
  size_t m1, m2, m3;
  NPME_ind3D_2_n1_n2_n3 (locCellIndexB, nPartition, nPartition, 
    m1, m2, m3);

  //total cell x, y, z indexes for cell2
  const size_t p1 = n1*nPartition + m1;
  const size_t p2 = n2*nPartition + m2;
  const size_t p3 = n3*nPartition + m3;


  const size_t totCellIndexB  = NPME_ind3D (p1, p2, p3, 
                                  nCellB_1D_Y, nCellB_1D_Z);

  return totCellIndexB;
}


size_t NPME_TotCellIndexToEmbeddedTotCellIndex (const size_t totCellIndexB,
  const size_t nPartition, const size_t nCellA_1D_Y, const size_t nCellA_1D_Z)
{
  const size_t Nc             = nPartition*nPartition*nPartition;
  const size_t nCellB_1D_Y    = nPartition*nCellA_1D_Y;
  const size_t nCellB_1D_Z    = nPartition*nCellA_1D_Z;

  //total cell x, y, z indexes for cellB
  size_t p1, p2, p3;
  NPME_ind3D_2_n1_n2_n3 (totCellIndexB, nCellB_1D_Y, nCellB_1D_Z, 
    p1, p2, p3);

  //local cell x, y, z indexes for cellB
  const size_t m1             = p1%nPartition;
  const size_t m2             = p2%nPartition;
  const size_t m3             = p3%nPartition;
  const size_t locCellIndexB  = NPME_ind3D (m1, m2, m3, nPartition, nPartition);

  //cell x, y, z indexes for cellA
  const size_t n1             = p1/nPartition;
  const size_t n2             = p2/nPartition;
  const size_t n3             = p3/nPartition;
  const size_t totCellIndexA  = NPME_ind3D (n1, n2, n3, 
                                        nCellA_1D_Y, nCellA_1D_Z);

  const size_t embeddedTotCellIndexB  = totCellIndexA*Nc + locCellIndexB;

  return embeddedTotCellIndexB;
}




//******************************************************************************
//******************************************************************************
//***************************NPME_Cluster (Cell A pairs)************************
//******************************************************************************
//******************************************************************************

struct NPME_ClusterPairTmp
{
  size_t cellIndexA1;
  size_t cellIndexA2;

  NPME_CellPairInteract pairB;
};



bool NPME_ClusterPairTmp_SortCriteria
  (const NPME_ClusterPairTmp& r1, 
   const NPME_ClusterPairTmp& r2)
{
  if (r1.cellIndexA1 < r2.cellIndexA1)
    return true;
  else if (r1.cellIndexA1 > r2.cellIndexA1)
    return false;
  else
  {
    if (r1.cellIndexA2 < r2.cellIndexA2)
      return true;
    else if (r1.cellIndexA2 > r2.cellIndexA2)
      return false;
    else
    {
      if (r1.pairB.cellIndex1 < r2.pairB.cellIndex1)
        return true;
      else if (r1.pairB.cellIndex1 > r2.pairB.cellIndex1)
        return false;
      else
      {
        if (r1.pairB.cellIndex2 < r2.pairB.cellIndex2)
          return true;
        else if (r1.pairB.cellIndex2 > r2.pairB.cellIndex2)
          return false;
        else
        {
          std::cout << "Error in NPME_ClusterPairTmp_SortCriteria\n";
          std::cout << "  found repeated element\n";
          exit(0);
        }
      }
    }
  }

  std::cout << "Error in NPME_ClusterPairTmp_SortCriteria\n";
  exit(0);
}


bool NPME_ConstructClusterPair (const size_t nPoint,
  const std::vector<size_t>& cellBToCellAindex,
  const std::vector<NPME_Library::NPME_CellPairInteract>& interactPairCellB,
  std::vector<NPME_Library::NPME_ClusterPair>& cluster)
//input:  cellBToCellAindex[nOccupyB] = map from cell B to cell A containining B
//        interactPairCellB[]         = cell B - cell B pairs of cells
//output: cluster[]                   = the cell B - cell B pairs are matched to
//                                      the cell A - cell A pairs
{
  using std::cout;
  using std::vector;

  const size_t nInteract = interactPairCellB.size();


  //construct NPME_ClusterPairTmp[nInteract] and sort
  vector<NPME_ClusterPairTmp> clusterTmp(nInteract);
  for (size_t i = 0; i < nInteract; i++)
  {
    size_t indB1 = interactPairCellB[i].cellIndex1;
    size_t indB2 = interactPairCellB[i].cellIndex2;

    clusterTmp[i].cellIndexA1 = cellBToCellAindex[indB1];
    clusterTmp[i].cellIndexA2 = cellBToCellAindex[indB2];
    clusterTmp[i].pairB       = interactPairCellB[i];
  }
  sort(clusterTmp.begin(), clusterTmp.end(), NPME_ClusterPairTmp_SortCriteria);
    


  //find number of clusters
  size_t nCluster       = 1;
  size_t prevCellIndA1  = clusterTmp[0].cellIndexA1;
  size_t prevCellIndA2  = clusterTmp[0].cellIndexA2;
  for (size_t i = 1; i < nInteract; i++)
  {
    size_t cellIndA1 = clusterTmp[i].cellIndexA1;
    size_t cellIndA2 = clusterTmp[i].cellIndexA2;

    if (  (cellIndA1 != prevCellIndA1) || 
          (cellIndA2 != prevCellIndA2) )
      nCluster++;

    prevCellIndA1 = cellIndA1;
    prevCellIndA2 = cellIndA2;
  }
  cluster.clear();
  cluster.resize(nCluster);


  size_t clusterInd = 0;
  {
    size_t cellIndA1                    = clusterTmp[0].cellIndexA1;
    size_t cellIndA2                    = clusterTmp[0].cellIndexA2;
    cluster[clusterInd].cellIndexA1     = cellIndA1;
    cluster[clusterInd].cellIndexA2     = cellIndA2;
    cluster[clusterInd].pairB.push_back(clusterTmp[0].pairB);
  }


  prevCellIndA1 = clusterTmp[0].cellIndexA1;
  prevCellIndA2 = clusterTmp[0].cellIndexA2;
  for (size_t i = 1; i < nInteract; i++)
  {
    size_t cellIndA1  = clusterTmp[i].cellIndexA1;
    size_t cellIndA2  = clusterTmp[i].cellIndexA2;

    if (  (cellIndA1 != prevCellIndA1) ||
          (cellIndA2 != prevCellIndA2)  )
    //start new cluster
    {
      clusterInd++;
      cluster[clusterInd].cellIndexA1     = cellIndA1;
      cluster[clusterInd].cellIndexA2     = cellIndA2;
    }

    cluster[clusterInd].pairB.push_back(clusterTmp[i].pairB);

    prevCellIndA1 = cellIndA1;
    prevCellIndA2 = cellIndA2;
  }


  for (size_t i = 0; i < cluster.size(); i++)
  {
    size_t minIndex1  = nPoint;
    size_t minIndex2  = nPoint;
    size_t maxIndex1  = 0;
    size_t maxIndex2  = 0;

    for (size_t j = 0; j < cluster[i].pairB.size(); j++)
    {
      size_t nPoint1 = cluster[i].pairB[j].nPointPerCell1;
      size_t nPoint2 = cluster[i].pairB[j].nPointPerCell2;
      size_t startPointIndex1 = cluster[i].pairB[j].startPointIndex1;
      size_t startPointIndex2 = cluster[i].pairB[j].startPointIndex2;

      if (minIndex1 > startPointIndex1)   minIndex1 = startPointIndex1;
      if (minIndex2 > startPointIndex2)   minIndex2 = startPointIndex2;

      if (maxIndex1 < startPointIndex1+nPoint1)   maxIndex1 = startPointIndex1+nPoint1;
      if (maxIndex2 < startPointIndex2+nPoint2)   maxIndex2 = startPointIndex2+nPoint2;
    }

    cluster[i].pointStartA1       = minIndex1;
    cluster[i].pointStartA2       = minIndex2;
    cluster[i].nPointPerCluster1  = maxIndex1 - minIndex1;
    cluster[i].nPointPerCluster2  = maxIndex2 - minIndex2;

    if ( (maxIndex1 > nPoint) || (maxIndex2 > nPoint) )
    {
      char str[2000];
      cout << "Error in NPME_ConstructClusterPair\n";
      sprintf(str, "maxIndex1 = %lu > nPoint = %lu\n", maxIndex1, nPoint);
      cout << str;
      sprintf(str, "maxIndex2 = %lu > nPoint = %lu\n", maxIndex2, nPoint);
      cout << str;
      exit(0);
    }
  }


  for (size_t i = 0; i < cluster.size(); i++)
  {
    size_t maxNumPoint = 0;
    for (size_t j = 0; j < cluster[i].pairB.size(); j++)
    {
      if (maxNumPoint < cluster[i].pairB[j].nPointPerCell1)
        maxNumPoint = cluster[i].pairB[j].nPointPerCell1;
      if (maxNumPoint < cluster[i].pairB[j].nPointPerCell2)
        maxNumPoint = cluster[i].pairB[j].nPointPerCell2;
    }
    cluster[i].maxNumPointPerCell = maxNumPoint;
  }


  if (clusterInd+1 != nCluster)
  {
    cout << "Error in NPME_ConstructClusterPair\n";
    char str[2000];
    sprintf(str, "clusterInd+1 = %lu != %lu = nCluster\n", 
      clusterInd+1, nCluster);
    cout << str;
    exit(0);  
  }

  return true;
}

double NPME_GaussTestFunctionModel (
  const double *coordPerm, const double *chargePerm, const double alpha, 
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster,
  const int nProc)
//input:  coordPerm[nPoint*3] = permuted coordinates
//        chargePerm[nPoint]  = permuted charges
//        alpha
//        interactList[nInteract]
//output: Gaussian charge energy
{
  //self cube contributions
  double E = 0.0;

  
  size_t n;
  const size_t nCluster = cluster.size();
  #pragma omp parallel for schedule(dynamic) shared(E, coordPerm, chargePerm, cluster) private(n) num_threads(nProc) 
  for (n = 0; n < nCluster; n++)
  {
    for (size_t i = 0; i < cluster[n].pairB.size(); i++)
    {
      if (cluster[n].pairB[i].cellIndex1 == cluster[n].pairB[i].cellIndex2)
      {
        const size_t nPoint1    = cluster[n].pairB[i].nPointPerCell1;
        const size_t startInd1  = cluster[n].pairB[i].startPointIndex1;

        const double *r = &coordPerm[3*startInd1];
        const double *q = &chargePerm[ startInd1];

        double Eloc = 0;
        for (size_t i = 0; i < nPoint1; i++)
        {
          double V = 0;
          for (size_t j = i; j < nPoint1; j++)
          {
            const double R2 = NPME_CalcDistance2 (&r[3*i], &r[3*j]);
            V += q[j]*exp(-alpha*R2);
          }
          Eloc += q[i]*V;
        }
        #pragma omp critical (update_NPME_GaussTestFunctionModel)
        {
          E += Eloc;
        }
      }
      else
      {
        const size_t nPoint1    = cluster[n].pairB[i].nPointPerCell1;
        const size_t nPoint2    = cluster[n].pairB[i].nPointPerCell2;
        const size_t startInd1  = cluster[n].pairB[i].startPointIndex1;
        const size_t startInd2  = cluster[n].pairB[i].startPointIndex2;

        const double *r1 = &coordPerm[3*startInd1];
        const double *r2 = &coordPerm[3*startInd2];
        const double *q1 = &chargePerm[ startInd1];
        const double *q2 = &chargePerm[ startInd2];

        double Eloc = 0;  
        for (size_t i = 0; i < nPoint1; i++)
        {
          double V = 0;
          for (size_t j = 0; j < nPoint2; j++)
          {
            const double R2 = NPME_CalcDistance2 (&r1[3*i], &r2[3*j]);
            V += q2[j]*exp(-alpha*R2);
          }
          Eloc += q1[i]*V;
        }
        #pragma omp critical (update_NPME_GaussTestFunctionModel)
        {
          E += Eloc;
        }
      }
    }
  }

  return E;
}

size_t NPME_CountNumDirectInteract (
  const std::vector<NPME_Library::NPME_ClusterPair>& cluster)
{
  size_t totNumInteract = 0;
  for (size_t n = 0; n < cluster.size(); n++)
  {
    for (size_t i = 0; i < cluster[n].pairB.size(); i++)
    {
      if (cluster[n].pairB[i].cellIndex1 == cluster[n].pairB[i].cellIndex2)
      {
        size_t nPoint1   = cluster[n].pairB[i].nPointPerCell1;
        totNumInteract  += (nPoint1*(nPoint1+1))/2;
      }
      else
      {
        const size_t nPoint1 = cluster[n].pairB[i].nPointPerCell1;
        const size_t nPoint2 = cluster[n].pairB[i].nPointPerCell2;
        totNumInteract      += nPoint1*nPoint2;
      }
    }
  }

  return totNumInteract;
}

}//end namespace NPME_Library



