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


#include "NPME_Constant.h"
#include "NPME_PartitionBox.h"
#include "NPME_SupportFunctions.h"
#include "NPME_PermuteArray.h"

#define NPME_PARTITION_BOX_DEBUG 0

namespace NPME_Library
{
//******************************************************************************
//******************************************************************************
//*******************************NPME_CellInterface*****************************
//******************************************************************************
//******************************************************************************


bool NPME_CellInterface (
  std::vector<NPME_Library::NPME_CellPairInteract>& interactList,
  std::vector<size_t>& P, std::vector<double>& coordPermute, 
  const size_t nPoint, const double *coord0, const size_t nNeigh, 
  const double Rdir, const char interactOpt, const int nProc)
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
{
  using std::vector;
  using std::cout;

  const double cellSize = Rdir/nNeigh;


  //1) total box size
  double L1, L2, L3;
  double trans[3];
  if (!NPME_GetBoxSize (L1, L2, L3, trans, nPoint, coord0))
  {
    cout << "Error in NPME_CellInterface\n";
    cout << "  NPME_GetBoxSize failed\n";
    return false;
  }

  //2) point into embedded cells
  vector<NPME_CellData> cell;
  size_t nCell_1D_X, nCell_1D_Y, nCell_1D_Z;


  P.resize(nPoint);
  coordPermute.resize(3*nPoint);

  if (!NPME_PointIntoCell (cell, 
        nCell_1D_X, nCell_1D_Y, nCell_1D_Z,
        &P[0], cellSize, L1, L2, L3, trans, nPoint, coord0))
  {
    cout << "Error in NPME_CellInterface\n";
    cout << "  NPME_PointIntoEmbeddedCell failed\n";
    return false;
  }


  //3) permute coordinates to fit neatly into cellB
  NPME_PermuteArrayInverse_MN (3, nPoint, &P[0], &coordPermute[0],  &coord0[0]);

  #if NPME_PARTITION_BOX_DEBUG
  {
    double maxDiff;
    if (!NPME_CheckCoordInCell (nPoint, &coordPermute[0],
      cellA, cellSizeA, maxDiff))
    {
      cout << "Error in NPME_CellInterface\n";
      cout << "  NPME_CheckCoordInCell failed for cellA\n";
      return false;
    }
    if (!NPME_CheckCoordInCell (nPoint, &coordPermute[0],
      cellB, cellSizeB, maxDiff))
    {
      cout << "Error in NPME_CellInterface\n";
      cout << "  NPME_CheckCoordInCell failed for cellB\n";
      return false;
    }
  }
  #endif

  //4) find cellB - cellB interaction lists
  if (!NPME_FindInteractionList (interactOpt, interactList,
        nNeigh, cell, cellSize, &coordPermute[0], 
        nCell_1D_X, nCell_1D_Y, nCell_1D_Z, nProc))
  {
    cout << "Error in NPME_CellInterface\n";
    cout << "  NPME_FindInteractionList failed\n";
    return false;
  }






  return true;
}




//******************************************************************************
//******************************************************************************
//******************************NPME_PointIntoCell******************************
//******************************************************************************
//******************************************************************************
bool NPME_GetBoxSize (double& L1, double& L2, double& L3,
  double trans[3], const size_t nPoint, const double *coord)
//input:  nPoint, coord[3*nPoint]
//output: L1, L2, L3 specifying a rectangular volume containing the points
//coord'[] = coord[] + trans[]  where coord'[] = x, y, z and
//0 <= x <= L1
//0 <= y <= L2
//0 <= z <= L3
{
  if (nPoint < 1)
  {
    std::cout << "Error in NPME_GetBoxSize.\n";
    std::cout << "nPoint = " << nPoint << " < 1\n";
    return false;
  }

  double min1 = coord[0];
  double min2 = coord[1];
  double min3 = coord[2];

  double max1 = coord[0];
  double max2 = coord[1];
  double max3 = coord[2];

  for (size_t i = 1; i < nPoint; i++)
  {
    const double x = coord[3*i  ];
    const double y = coord[3*i+1];
    const double z = coord[3*i+2];

    if (min1 > x) min1 = x;
    if (min2 > y) min2 = y;
    if (min3 > z) min3 = z;

    if (max1 < x) max1 = x;
    if (max2 < y) max2 = y;
    if (max3 < z) max3 = z;
  }

  L1 = max1 - min1;
  L2 = max2 - min2;
  L3 = max3 - min3;

  trans[0] = -min1;
  trans[1] = -min2;
  trans[2] = -min3;

  return true;
}

struct NPME_PointCellPair
{
  size_t pointIndex;
  size_t totCellIndex;
};
bool NPME_PointCellPair_SortCriteria 
  (const NPME_PointCellPair& r1, 
   const NPME_PointCellPair& r2)
{
  if (r1.totCellIndex < r2.totCellIndex)
    return true;
  else 
    return false;
}

bool NPME_PointIntoCell_CheckCoordinates (const size_t i,
  const double x,  const double y,  const double z,
  const double L1, const double L2, const double L3)
{
  const double tol = 1.0E-14;
  using std::cout;
  using std::endl;

  const char *strError = "Error in NPME_PointIntoCell_CheckCoordinates for ";
  if ( (x < -tol) || (x > L1+tol) )
  {
    cout << strError << "point " << i << endl;
    cout << " x  = " << x  << endl;
    cout << " L1 = " << L1 << endl;
    cout << "x is outsize [0, L1]\n";
    return false;
  }
  if ( (y < -tol) || (y > L2+tol) )
  {
    cout << strError << "point " << i << endl;
    cout << " y  = " << y  << endl;
    cout << " L2 = " << L2 << endl;
    cout << "y is outsize [0, L2]\n";
    return false;
  }

  if ( (z < -tol) || (z > L3+tol) )
  {
    cout << strError << "point " << i << endl;
    cout << " z  = " << z  << endl;
    cout << " L3 = " << L3 << endl;
    cout << "z is outsize [0, L3]\n";
    return false;
  }


  return true;
}

bool NPME_PointIntoCell_CheckCellIndex (const size_t i,
  const size_t n1, const size_t n2, const size_t n3,
  const size_t N1, const size_t N2, const size_t N3)
{
  using std::cout;
  using std::endl;

  const char *strError = "Error in NPME_PointIntoCell_CheckCellIndex for ";

  if (n1 >= N1)
  {
    cout << strError << "point " << i << endl;
    cout << " n1 = " << n1 << endl;
    cout << " N1 = " << N1 << endl;
    cout << "n1 >= N1\n";
    return false;
  }
  if (n2 >= N2)
  {
    cout << strError << "point " << i << endl;
    cout << " n2 = " << n2 << endl;
    cout << " N2 = " << N2 << endl;
    cout << "n2 >= N2\n";
    return false;
  }
  if (n3 >= N3)
  {
    cout << strError << "point " << i << endl;
    cout << " n3 = " << n3 << endl;
    cout << " N3 = " << N3 << endl;
    cout << "n3 >= N3\n";
    return false;
  }

  return true;
}

void NPME_CalcCellCoord (double cellCoord[3], 
  const size_t n1, const size_t n2, const size_t n3, const double cellSize,
  const double trans[3])
//input:  cell index (n1, n2, n3)
//        cellSize
//        trans[3]
//output: cellCoord[3]
{
  const double cs_2 = cellSize/2.0;
  cellCoord[0] = n1*cellSize - trans[0] + cs_2;
  cellCoord[1] = n2*cellSize - trans[1] + cs_2;
  cellCoord[2] = n3*cellSize - trans[2] + cs_2;
}

void NPME_TotCellIndex2CellCoord (double cellCoord[3], 
  const size_t totCellIndex, const double cellSize, const double trans[3],
  const size_t nCell1D_2, const size_t nCell1D_3)
{
  size_t n1, n2, n3;
  NPME_ind3D_2_n1_n2_n3 (totCellIndex, nCell1D_2, nCell1D_3, n1, n2, n3);
  NPME_CalcCellCoord (cellCoord, n1, n2, n3, cellSize, trans);
}


size_t NPME_ConvertCoord2CellIndex (const double x, const double cellSize, 
  const size_t nCell1D)
{
  using std::cout;

  const double tol = 1.0E-14;
  char str[2000];

  if (x < -tol)
  {
    cout << "Error in NPME_ConvertCoord2CellIndex\n";
    sprintf(str, "x = %.3le < 0.0\n", x);
    cout << str;
    exit(0);
  }

  size_t m = (size_t) (x/cellSize);
  if (m > nCell1D)
  {
    cout << "Error in NPME_ConvertCoord2CellIndex\n";
    sprintf(str, "x = %.3le < 0.0\n", x);
    cout << str;
    exit(0);
  }
  else if (m == nCell1D)
  {

    if (fabs(x - m*cellSize) < tol)
      m--;
    else
    {
      cout << "Error in NPME_ConvertCoord2CellIndex\n";
      sprintf(str, "x = %le cellSize = %le m = %lu nCell1D = %lu\n", 
        x, cellSize, m, nCell1D);
      cout << str;
      exit(0);
    }
  }

  return m;
}

bool NPME_PointIntoCell (
  std::vector<NPME_CellData>& cell,
  size_t& nCell1D_1, size_t& nCell1D_2, size_t& nCell1D_3, size_t *P,
  const double cellSize, const double L1, const double L2, const double L3,
  const double trans[3], const size_t nPoint, const double *coord)
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
{
  using std::vector;
  using std::cout;

  nCell1D_1 = (size_t) (L1/cellSize) + 1;
  nCell1D_2 = (size_t) (L2/cellSize) + 1;
  nCell1D_3 = (size_t) (L3/cellSize) + 1;

  std::vector<NPME_PointCellPair> pairTmp(nPoint);
  
  #if NPME_PARTITION_BOX_DEBUG
    cout << "nCell1D = " << nCell1D_1 << " ";
    cout << nCell1D_2 << " " << nCell1D_3 << "\n";
  #endif

  for (size_t i = 0; i < nPoint; i++)
  {
    //translate coord[] s.t. 
    //        0 <= x <= L1
    //        0 <= y <= L2
    //        0 <= z <= L3
    const double *r = &coord[3*i];
    const double x  = r[0] + trans[0];
    const double y  = r[1] + trans[1];
    const double z  = r[2] + trans[2];

    //integers defining cell location
    const size_t n1 = NPME_ConvertCoord2CellIndex (x, cellSize, nCell1D_1);
    const size_t n2 = NPME_ConvertCoord2CellIndex (y, cellSize, nCell1D_2);
    const size_t n3 = NPME_ConvertCoord2CellIndex (z, cellSize, nCell1D_3);

    #if NPME_PARTITION_BOX_DEBUG
    if (!NPME_PointIntoCell_CheckCoordinates (i, x, y, z, L1, L2, L3))
    {
      cout << "Error in NPME_PointIntoCell.\n";
      cout << "NPME_PointIntoCell_CheckCoordinates failed\n";
      return false;
    }
    if (!NPME_PointIntoCell_CheckCellIndex (i, n1, n2, n3, 
            nCell1D_1, nCell1D_2, nCell1D_3))
    {
      cout << "Error in NPME_PointIntoCell.\n";
      cout << "NPME_PointIntoCell_CheckCellIndex failed\n";
      return false;
    }
    #endif

    pairTmp[i].pointIndex = i;
    pairTmp[i].totCellIndex  = NPME_ind3D (n1, n2, n3, nCell1D_2, nCell1D_3);
  }

  std::sort(pairTmp.begin(), pairTmp.end(), NPME_PointCellPair_SortCriteria);

  #if NPME_PARTITION_BOX_DEBUG
  {
    for (size_t i = 0; i < nPoint; i++)
    {
      cout << pairTmp[i].pointIndex << " ";
      cout << pairTmp[i].totCellIndex << "\n";
    }
  }
  #endif

  size_t nOccupCell = 1;
  for (size_t i = 1; i < nPoint; i++)
    if (pairTmp[i].totCellIndex != pairTmp[i-1].totCellIndex)
      nOccupCell++;

  #if NPME_PARTITION_BOX_DEBUG
    cout << "nOccupCell = " << nOccupCell << "\n";
  #endif

  cell.resize(nOccupCell);

  size_t occupCellIndex                 = 0;
  cell[occupCellIndex].pointStartIndex  = 0;
  cell[occupCellIndex].nPointPerCell    = 1;
  cell[occupCellIndex].totCellIndex     = pairTmp[0].totCellIndex;

  NPME_TotCellIndex2CellCoord (cell[occupCellIndex].cellCoord, 
    pairTmp[0].totCellIndex,
    cellSize, trans, nCell1D_2, nCell1D_3);
  P[ pairTmp[0].pointIndex ] = 0;

  for (size_t i = 1; i < nPoint; i++)
  {
    if (pairTmp[i].totCellIndex != pairTmp[i-1].totCellIndex)
    {
      occupCellIndex++;
      NPME_TotCellIndex2CellCoord (cell[occupCellIndex].cellCoord, 
        pairTmp[i].totCellIndex, cellSize, trans, nCell1D_2, nCell1D_3);
      cell[occupCellIndex].pointStartIndex  = i;
      cell[occupCellIndex].nPointPerCell    = 1;
      cell[occupCellIndex].totCellIndex     = pairTmp[i].totCellIndex;
    }
    else
      cell[occupCellIndex].nPointPerCell++;

    P[ pairTmp[i].pointIndex ] = i;
  }

  #if NPME_PARTITION_BOX_DEBUG
  {
    cout << "nOccupCell = " << nOccupCell << "\n";
    for (size_t n = 0; n < nOccupCell; n++)
    {
      char str[2000];
      sprintf(str, "  cell %4lu nPoint = %5lu pointStartInd = %5lu  %.4f %.4f %.4f\n", 
        n, cell[n].nPointPerCell, cell[n].pointStartIndex, 
        cell[occupCellIndex].cellCoord[0], cell[occupCellIndex].cellCoord[1],
        cell[occupCellIndex].cellCoord[2]);
      cout << str;
    }
  }
  #endif

  return true;
}



bool NPME_CheckCoordInCell (const size_t nPoint, const double *coordPermute,
  const std::vector<NPME_Library::NPME_CellData>& cell, const double cellSize,
  double& maxDiff)
//input: coordPermute[nPoint]
//checks permuted coordinates are inside the assigned cell
{
  using std::cout;

  maxDiff = 0.0;
  const size_t nOccupCell = cell.size();

  for (size_t n = 0; n < nOccupCell; n++)
  {
    for (size_t i = 0; i < cell[n].nPointPerCell; i++)
    {
      const size_t index  = cell[n].pointStartIndex + i;
      const double *r1    = &coordPermute[3*index];
      const double *r2    = cell[n].cellCoord;

      const double del_x  = fabs(r1[0] - r2[0]);
      const double del_y  = fabs(r1[1] - r2[1]);
      const double del_z  = fabs(r1[2] - r2[2]);

      if (maxDiff < del_x)  maxDiff = del_x;
      if (maxDiff < del_y)  maxDiff = del_y;
      if (maxDiff < del_z)  maxDiff = del_z;

      if (del_x > cellSize/2.0*1.001)
      {
        cout << "Error in NPME_CheckCoordInCell.\n";
        cout << "  cell     = " << n << "\n";
        cout << "  point    = " << index << "\n";
        cout << "  del_x    = " << del_x << "\n";
        cout << "  cellSize = " << cellSize << "\n";
        cout << "del_x > cellSize/2";
        return false;
      }
      if (del_y > cellSize/2.0*1.001)
      {
        cout << "Error in NPME_CheckCoordInCell.\n";
        cout << "  cell     = " << n << "\n";
        cout << "  point    = " << index << "\n";
        cout << "  del_y    = " << del_y << "\n";
        cout << "  cellSize = " << cellSize << "\n";
        cout << "del_y > cellSize/2";
        return false;
      }
      if (del_z > cellSize/2.0*1.001)
      {
        cout << "Error in NPME_CheckCoordInCell.\n";
        cout << "  cell     = " << n << "\n";
        cout << "  point    = " << index << "\n";
        cout << "  del_z    = " << del_z << "\n";
        cout << "  cellSize = " << cellSize << "\n";
        cout << "del_z > cellSize/2";
        return false;
      }
    }
  }

  return true;
}



//******************************************************************************
//******************************************************************************
//************************NPME_FindInteractionList******************************
//******************************************************************************
//******************************************************************************

bool NPME_FindInteractionList_A (
  std::vector<NPME_Library::NPME_CellPairInteract>& interactList,
  const size_t nNeigh, std::vector<NPME_Library::NPME_CellData>& cell,
  const size_t nCell1D_X, const size_t nCell1D_Y, const size_t nCell1D_Z)
//1) Simplest Least Pruning Algorithm
//input:  cell[nOccupCell] = number of non-empty occupied cells
//        number of total cells = nCell1D_X*nCell1D_Y*nCell1D_Z
//output: interactList[nInteract]
{
  const size_t nTotCell   = nCell1D_X*nCell1D_Y*nCell1D_Z;
  bool *isTotCellOccupied = new bool [nTotCell];
  std::vector<size_t> totCell2OcCellIndex(nTotCell);

  memset(&isTotCellOccupied[0],   0, nTotCell*sizeof(bool));
  memset(&totCell2OcCellIndex[0], 0, nTotCell*sizeof(size_t));

  const size_t nOccupCell = cell.size();

  for (size_t i = 0; i < nOccupCell; i++)
  {
    size_t totCellInd = cell[i].totCellIndex;
    if (totCellInd >= nTotCell)
    {
      std::cout << "Error in NPME_FindInteractionList_A.\n";
      std::cout << "  totCellInd = " << totCellInd << "\n";
      std::cout << "  nTotCell   = " << nTotCell   << "\n";
      std::cout << "totCellInd >= nTotCell\n";
      return false;
    }
    isTotCellOccupied[totCellInd]   = 1;
    totCell2OcCellIndex[totCellInd] = i;
  }

  interactList.clear();
  for (size_t i = 0; i < nOccupCell; i++)
  {
    size_t occupCellInd1  = i;
    size_t totCellInd1    = cell[occupCellInd1].totCellIndex;
    size_t n1, n2, n3;
    NPME_ind3D_2_n1_n2_n3 (totCellInd1, nCell1D_Y, nCell1D_Z, n1, n2, n3);

    size_t m1_Min, m1_Max;
    size_t m2_Min, m2_Max;
    size_t m3_Min, m3_Max;

    if (n1 >= nNeigh) m1_Min = n1 - nNeigh;   else  m1_Min = 0;
    if (n2 >= nNeigh) m2_Min = n2 - nNeigh;   else  m2_Min = 0;
    if (n3 >= nNeigh) m3_Min = n3 - nNeigh;   else  m3_Min = 0;

    m1_Max = n1 + nNeigh;
    m2_Max = n2 + nNeigh;
    m3_Max = n3 + nNeigh;

    if (m1_Max >= nCell1D_X)  m1_Max = nCell1D_X - 1;
    if (m2_Max >= nCell1D_Y)  m2_Max = nCell1D_Y - 1;
    if (m3_Max >= nCell1D_Z)  m3_Max = nCell1D_Z - 1;

    //include self-interaction
    {
      struct NPME_CellPairInteract tmp;
      tmp.cellIndex1        = occupCellInd1;
      tmp.cellIndex2        = occupCellInd1;
      tmp.nPointPerCell1    = cell[occupCellInd1].nPointPerCell;
      tmp.nPointPerCell2    = cell[occupCellInd1].nPointPerCell;
      tmp.startPointIndex1  = cell[occupCellInd1].pointStartIndex;
      tmp.startPointIndex2  = cell[occupCellInd1].pointStartIndex;
      interactList.push_back(tmp);
    }
    for (size_t m1 = m1_Min; m1 <= m1_Max; m1++)
    for (size_t m2 = m2_Min; m2 <= m2_Max; m2++)
    for (size_t m3 = m3_Min; m3 <= m3_Max; m3++)
    {
      const size_t totCellInd2 = NPME_ind3D (m1, m2, m3, nCell1D_Y, nCell1D_Z);
      if (isTotCellOccupied[totCellInd2])
      {
        const size_t occupCellInd2 = totCell2OcCellIndex[totCellInd2];
        if (occupCellInd1 < occupCellInd2)
        {
          struct NPME_CellPairInteract tmp;
          tmp.cellIndex1        = occupCellInd1;
          tmp.cellIndex2        = occupCellInd2;
          tmp.nPointPerCell1    = cell[occupCellInd1].nPointPerCell;
          tmp.nPointPerCell2    = cell[occupCellInd2].nPointPerCell;
          tmp.startPointIndex1  = cell[occupCellInd1].pointStartIndex;
          tmp.startPointIndex2  = cell[occupCellInd2].pointStartIndex;
          interactList.push_back(tmp);
        }
      }
    }
  }


  delete [] isTotCellOccupied;

  return true;
}


bool NPME_FindInteractionList_B (
  std::vector<NPME_Library::NPME_CellPairInteract>& interactList,
  const size_t nNeigh, std::vector<NPME_Library::NPME_CellData>& cell,
  const double cellSize,
  const size_t nCell1D_X, const size_t nCell1D_Y, const size_t nCell1D_Z)
//2) Medium Pruning Algorithm
//input:  cell[nOccupCell] = number of non-empty occupied cells
//        number of total cells = nCell1D_X*nCell1D_Y*nCell1D_Z
//output: interactList[nInteract]
{
  const size_t nTotCell   = nCell1D_X*nCell1D_Y*nCell1D_Z;
  bool *isTotCellOccupied = new bool [nTotCell];
  std::vector<size_t> totCell2OcCellIndex(nTotCell);

  memset(&isTotCellOccupied[0],   0, nTotCell*sizeof(bool));
  memset(&totCell2OcCellIndex[0], 0, nTotCell*sizeof(size_t));

  const size_t nOccupCell = cell.size();

  for (size_t i = 0; i < nOccupCell; i++)
  {
    size_t totCellInd = cell[i].totCellIndex;
    if (totCellInd >= nTotCell)
    {
      std::cout << "Error in NPME_FindInteractionList_B.\n";
      std::cout << "  totCellInd = " << totCellInd << "\n";
      std::cout << "  nTotCell   = " << nTotCell   << "\n";
      std::cout << "totCellInd >= nTotCell\n";
      return false;
    }
    isTotCellOccupied[totCellInd]   = 1;
    totCell2OcCellIndex[totCellInd] = i;
  }

  const double Rdir = nNeigh*cellSize;

  interactList.clear();
  for (size_t i = 0; i < nOccupCell; i++)
  {
    size_t occupCellInd1  = i;
    size_t totCellInd1    = cell[occupCellInd1].totCellIndex;
    size_t n1, n2, n3;
    NPME_ind3D_2_n1_n2_n3 (totCellInd1, nCell1D_Y, nCell1D_Z, n1, n2, n3);

    size_t m1_Min, m1_Max;
    size_t m2_Min, m2_Max;
    size_t m3_Min, m3_Max;

    if (n1 >= nNeigh) m1_Min = n1 - nNeigh;   else  m1_Min = 0;
    if (n2 >= nNeigh) m2_Min = n2 - nNeigh;   else  m2_Min = 0;
    if (n3 >= nNeigh) m3_Min = n3 - nNeigh;   else  m3_Min = 0;

    m1_Max = n1 + nNeigh;
    m2_Max = n2 + nNeigh;
    m3_Max = n3 + nNeigh;

    if (m1_Max >= nCell1D_X)  m1_Max = nCell1D_X - 1;
    if (m2_Max >= nCell1D_Y)  m2_Max = nCell1D_Y - 1;
    if (m3_Max >= nCell1D_Z)  m3_Max = nCell1D_Z - 1;

    const double *r1 = cell[occupCellInd1].cellCoord;


    //include self-interaction
    {
      struct NPME_CellPairInteract tmp;
      tmp.cellIndex1        = occupCellInd1;
      tmp.cellIndex2        = occupCellInd1;
      tmp.nPointPerCell1    = cell[occupCellInd1].nPointPerCell;
      tmp.nPointPerCell2    = cell[occupCellInd1].nPointPerCell;
      tmp.startPointIndex1  = cell[occupCellInd1].pointStartIndex;
      tmp.startPointIndex2  = cell[occupCellInd1].pointStartIndex;
      interactList.push_back(tmp);
    }
    for (size_t m1 = m1_Min; m1 <= m1_Max; m1++)
    for (size_t m2 = m2_Min; m2 <= m2_Max; m2++)
    for (size_t m3 = m3_Min; m3 <= m3_Max; m3++)
    {
      const size_t totCellInd2 = NPME_ind3D (m1, m2, m3, nCell1D_Y, nCell1D_Z);
      if (isTotCellOccupied[totCellInd2])
      {
        const size_t occupCellInd2 = totCell2OcCellIndex[totCellInd2];
        if (occupCellInd1 < occupCellInd2)
        {
          const double *r2 = cell[occupCellInd2].cellCoord;
          double minDist   = NPME_CalcMinDistanceRectVolumes (
                                r1, cellSize, cellSize, cellSize,
                                r2, cellSize, cellSize, cellSize);

          if (minDist <= Rdir*1.0001)
          {
            struct NPME_CellPairInteract tmp;
            tmp.cellIndex1        = occupCellInd1;
            tmp.cellIndex2        = occupCellInd2;
            tmp.nPointPerCell1    = cell[occupCellInd1].nPointPerCell;
            tmp.nPointPerCell2    = cell[occupCellInd2].nPointPerCell;
            tmp.startPointIndex1  = cell[occupCellInd1].pointStartIndex;
            tmp.startPointIndex2  = cell[occupCellInd2].pointStartIndex;
            interactList.push_back(tmp);
          }
        }
      }
    }
  }


  delete [] isTotCellOccupied;

  return true;
}

bool NPME_FindInteractionList_C_CloseContact (const double Rdir2,
  const size_t nPoint1, const double *crdTmp1,
  const size_t nPoint2, const double *crdTmp2)
{
  bool closeContact = 0;
  for (size_t n1 = 0; n1 < nPoint1; n1++)
  for (size_t n2 = 0; n2 < nPoint2; n2++)
  {
    const double *rPt1 = &crdTmp1[3*n1];
    const double *rPt2 = &crdTmp2[3*n2];
    const double dist  = NPME_CalcDistance2 (rPt1, rPt2);
    if (dist < Rdir2*1.0001)
      return true;
  }
  return false;
}

bool NPME_FindInteractionList_C (
  std::vector<NPME_Library::NPME_CellPairInteract>& interactList,
  const size_t nNeigh, std::vector<NPME_Library::NPME_CellData>& cell,
  const double cellSize, const double *coordPerm, 
  const size_t nCell1D_X, const size_t nCell1D_Y, const size_t nCell1D_Z,
  int nProc)
//3) Most Agressive Pruning Algorithm (Slow SetUp but faster Direct Sum)
//input:  cell[nOccupCell] = number of non-empty occupied cells
//        number of total cells = nCell1D_X*nCell1D_Y*nCell1D_Z
//        coordPerm[nPoint*3]   = permuted coordinates of points
//output: interactList[nInteract]
{
  const size_t nTotCell   = nCell1D_X*nCell1D_Y*nCell1D_Z;
  bool *isTotCellOccupied = new bool [nTotCell];
  std::vector<size_t> totCell2OcCellIndex(nTotCell);

  memset(&isTotCellOccupied[0],   0, nTotCell*sizeof(bool));
  memset(&totCell2OcCellIndex[0], 0, nTotCell*sizeof(size_t));

  const size_t nOccupCell = cell.size();

  for (size_t i = 0; i < nOccupCell; i++)
  {
    size_t totCellInd = cell[i].totCellIndex;
    if (totCellInd >= nTotCell)
    {
      std::cout << "Error in NPME_FindInteractionList_C.\n";
      std::cout << "  totCellInd = " << totCellInd << "\n";
      std::cout << "  nTotCell   = " << nTotCell   << "\n";
      std::cout << "totCellInd >= nTotCell\n";
      return false;
    }
    isTotCellOccupied[totCellInd]   = 1;
    totCell2OcCellIndex[totCellInd] = i;
  }

  const double Rdir  = nNeigh*cellSize;
  const double Rdir2 = Rdir*Rdir;

  interactList.clear();

  size_t i;
  #pragma omp parallel for schedule(dynamic) shared(totCell2OcCellIndex, cell, coordPerm) private(i) num_threads(nProc)
  for (i = 0; i < nOccupCell; i++)
  {
    size_t occupCellInd1  = i;
    size_t totCellInd1    = cell[occupCellInd1].totCellIndex;
    size_t n1, n2, n3;
    NPME_ind3D_2_n1_n2_n3 (totCellInd1, nCell1D_Y, nCell1D_Z, n1, n2, n3);

    size_t m1_Min, m1_Max;
    size_t m2_Min, m2_Max;
    size_t m3_Min, m3_Max;

    if (n1 >= nNeigh) m1_Min = n1 - nNeigh;   else  m1_Min = 0;
    if (n2 >= nNeigh) m2_Min = n2 - nNeigh;   else  m2_Min = 0;
    if (n3 >= nNeigh) m3_Min = n3 - nNeigh;   else  m3_Min = 0;

    m1_Max = n1 + nNeigh;
    m2_Max = n2 + nNeigh;
    m3_Max = n3 + nNeigh;

    if (m1_Max >= nCell1D_X)  m1_Max = nCell1D_X - 1;
    if (m2_Max >= nCell1D_Y)  m2_Max = nCell1D_Y - 1;
    if (m3_Max >= nCell1D_Z)  m3_Max = nCell1D_Z - 1;

    const double *r1 = cell[occupCellInd1].cellCoord;

    std::vector<NPME_CellPairInteract> interactListTmp;
    //include self-interaction
    {
      struct NPME_CellPairInteract tmp;
      tmp.cellIndex1        = occupCellInd1;
      tmp.cellIndex2        = occupCellInd1;
      tmp.nPointPerCell1    = cell[occupCellInd1].nPointPerCell;
      tmp.nPointPerCell2    = cell[occupCellInd1].nPointPerCell;
      tmp.startPointIndex1  = cell[occupCellInd1].pointStartIndex;
      tmp.startPointIndex2  = cell[occupCellInd1].pointStartIndex;
      interactListTmp.push_back(tmp);
    }
    for (size_t m1 = m1_Min; m1 <= m1_Max; m1++)
    for (size_t m2 = m2_Min; m2 <= m2_Max; m2++)
    for (size_t m3 = m3_Min; m3 <= m3_Max; m3++)
    {
      const size_t totCellInd2 = NPME_ind3D (m1, m2, m3, nCell1D_Y, nCell1D_Z);
      if (isTotCellOccupied[totCellInd2])
      {
        const size_t occupCellInd2 = totCell2OcCellIndex[totCellInd2];
        if (occupCellInd1 < occupCellInd2)
        {
          const double *r2 = cell[occupCellInd2].cellCoord;
          double minDist   = NPME_CalcMinDistanceRectVolumes (
                                r1, cellSize, cellSize, cellSize,
                                r2, cellSize, cellSize, cellSize);

          if (minDist < Rdir*1.0001)
          {
            const size_t nPoint1  = cell[occupCellInd1].nPointPerCell;
            const size_t nPoint2  = cell[occupCellInd2].nPointPerCell;
            const size_t start1   = cell[occupCellInd1].pointStartIndex;
            const size_t start2   = cell[occupCellInd2].pointStartIndex;
            const double *crdTmp1 = &coordPerm[3*start1];
            const double *crdTmp2 = &coordPerm[3*start2];

            bool closeContact = NPME_FindInteractionList_C_CloseContact (Rdir2,
                                  nPoint1, crdTmp1,
                                  nPoint2, crdTmp2);
            if (closeContact)
            {
              struct NPME_CellPairInteract tmp;
              tmp.cellIndex1        = occupCellInd1;
              tmp.cellIndex2        = occupCellInd2;
              tmp.nPointPerCell1    = cell[occupCellInd1].nPointPerCell;
              tmp.nPointPerCell2    = cell[occupCellInd2].nPointPerCell;
              tmp.startPointIndex1  = cell[occupCellInd1].pointStartIndex;
              tmp.startPointIndex2  = cell[occupCellInd2].pointStartIndex;
              interactListTmp.push_back(tmp);
            }
          }
        }
      }
    }

    #pragma omp critical (update_NPME_FindInteractionList_C)
    {
      for (size_t n = 0; n < interactListTmp.size(); n++)
        interactList.push_back(interactListTmp[n]);
    }
  }

  delete [] isTotCellOccupied;

  return true;
}


bool NPME_FindInteractionList (const char option,
  std::vector<NPME_Library::NPME_CellPairInteract>& interactList,
  const size_t nNeigh, std::vector<NPME_Library::NPME_CellData>& cell,
  const double cellSize, const double *coordPerm, 
  const size_t nCell1D_X, const size_t nCell1D_Y, const size_t nCell1D_Z,
  int nProc)
//option = 'A', 'B', or 'C'
{
  if (option == 'A')
    return NPME_FindInteractionList_A (interactList, nNeigh, cell,
      nCell1D_X, nCell1D_Y, nCell1D_Z);
  else if (option == 'B')
    return NPME_FindInteractionList_B (interactList, nNeigh, cell, cellSize,
      nCell1D_X, nCell1D_Y, nCell1D_Z);
  else if (option == 'C')
    return NPME_FindInteractionList_C (interactList, nNeigh, cell, cellSize,
      coordPerm, nCell1D_X, nCell1D_Y, nCell1D_Z, nProc);
  else
  {
    std::cout << "Error in NPME_FindInteractionList\n";
    std::cout << "option = " << option << " is undefined\n";
    std::cout << "option must be 'A', 'B', 'C'\n";
    return false;
  }
}


//******************************************************************************
//******************************************************************************
//********************Test NPME_FindInteractionList*****************************
//******************************************************************************
//******************************************************************************

double NPME_GaussTestFunctionExact (const size_t nPoint, const double *coord,
  const double *charge, const double alpha, const int nProc)
{
  double E = 0.0;
  size_t i;
  #pragma omp parallel for schedule(dynamic) shared(coord, charge) private(i) num_threads(nProc)
  for (i = 0; i < nPoint; i++)
  {
    double V = 0;
    for (size_t j = i; j < nPoint; j++)
    {
      const double R2 = NPME_CalcDistance2 (&coord[3*i], &coord[3*j]);
      V += charge[j]*exp(-alpha*R2);
    }

    #pragma omp critical (update_NPME_GaussTestFunctionExact)
    {
      E += charge[i]*V;
    }
  }

  return E;
}




double NPME_GaussTestFunctionModel (
  const double *coordPerm, const double *chargePerm, const double alpha, 
  const std::vector<NPME_Library::NPME_CellPairInteract>& interactList,
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
  const size_t nInteractList = interactList.size();
  #pragma omp parallel for schedule(dynamic) shared(E, coordPerm, chargePerm, interactList) private(n) num_threads(nProc) 
  for (n = 0; n < nInteractList; n++)
  {
    if (interactList[n].cellIndex1 == interactList[n].cellIndex2)
    {
      const size_t nPoint1    = interactList[n].nPointPerCell1;
      const size_t startInd1  = interactList[n].startPointIndex1;

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
      #pragma omp critical (update_NPME_GaussTestFunctionModel2)
      {
        E += Eloc;
      }
    }
    else
    {
      const size_t nPoint1    = interactList[n].nPointPerCell1;
      const size_t nPoint2    = interactList[n].nPointPerCell2;
      const size_t startInd1  = interactList[n].startPointIndex1;
      const size_t startInd2  = interactList[n].startPointIndex2;

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
      #pragma omp critical (update_NPME_GaussTestFunctionModel2)
      {
        E += Eloc;
      }
    }
  }

  return E;
}

size_t NPME_CountNumDirectInteract (
  const std::vector<NPME_Library::NPME_CellPairInteract>& interactList)
{
  size_t totNumInteract = 0;
  for (size_t n = 0; n < interactList.size(); n++)
  {
    if (interactList[n].cellIndex1 == interactList[n].cellIndex2)
    {
      size_t nPoint1   = interactList[n].nPointPerCell1;
      totNumInteract  += (nPoint1*(nPoint1+1))/2;
    }
    else
    {
      const size_t nPoint1 = interactList[n].nPointPerCell1;
      const size_t nPoint2 = interactList[n].nPointPerCell2;
      totNumInteract      += nPoint1*nPoint2;
    }
  }

  return totNumInteract;
}






}//end namespace NPME_Library



