#create a file of nCharge = 100,000 random charges and coordinates
../../exe/makeRandomBox input_makeRandomBoxReal.txt
../../exe/makeRandomBox input_makeRandomBoxComplex.txt

#run npme on the charge/coord file (crdChg.txt) and npme instruction files with
#various direct space cutoffs Rdir = 4.0, 5.0, 6.0, .. 11.0
#note that calcType = pme_exact in laplaceDM_04.txt which instructs npme
#to calculate both the NPME and exact potential/potential gradient


echo " "
../../exe/laplaceDM crdChgReal.txt laplaceDM.txt
echo " "
../../exe/RalphaDM crdChgReal.txt RalphaDM.txt
echo " "
../../exe/helmholtzDM crdChgComplex.txt helmholtzDM.txt




echo " "
echo "Errors for laplaceDM.txt"
../../exe/compareV crdChgReal_laplaceDM_V_exact.output crdChgReal_laplaceDM_V_pme.output

echo " "
echo "Errors for RalphaDM.txt"
../../exe/compareV crdChgReal_RalphaDM_V_exact.output crdChgReal_RalphaDM_V_pme.output

echo " "
echo "Errors for helmholtzDM.txt"
../../exe/compareV crdChgComplex_helmholtzDM_V_exact.output crdChgComplex_helmholtzDM_V_pme.output



echo "---------------------------------------------------------------------"
echo " "
echo "Exact Time for laplaceDM:"
grep time_exact crdChgReal_laplaceDM.log
echo " "
echo "Exact Time for RalphaDM:"
grep time_exact crdChgReal_RalphaDM.log
echo " "
echo "Exact Time for helmholtzDM:"
grep time_exact crdChgComplex_helmholtzDM.log
echo "---------------------------------------------------------------------"

echo " "
echo "NPME Rec. Sum Times:"
grep 'rec sum' *.log

echo " "
echo "NPME Direct Sum Times:"
grep 'direct sum' *.log

echo " "
echo "NPME Total Times:"
grep 'total V1' *.log


