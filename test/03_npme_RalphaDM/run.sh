#create a file of nCharge = 100,000 random charges and coordinates
../../exe/makeRandomBox input_makeRandomBoxReal.txt

#run npme on the charge/coord file (crdChg.txt) and npme instruction files with
#various direct space cutoffs Rdir = 4.0, 5.0, 6.0, .. 11.0
#note that calcType = pme_exact in RalphaDM_04.txt which instructs npme
#to calculate both the NPME and exact potential/potential gradient

echo " "
../../exe/npme crdChg.txt RalphaDM_04.txt
echo " "
../../exe/npme crdChg.txt RalphaDM_05.txt
echo " "
../../exe/npme crdChg.txt RalphaDM_06.txt
echo " "
../../exe/npme crdChg.txt RalphaDM_07.txt
echo " "
../../exe/npme crdChg.txt RalphaDM_08.txt
echo " "
../../exe/npme crdChg.txt RalphaDM_09.txt
echo " "
../../exe/npme crdChg.txt RalphaDM_10.txt
echo " "
../../exe/npme crdChg.txt RalphaDM_11.txt
echo " "


echo " "
echo "Errors for RalphaDM_04.txt"
../../exe/compareV crdChg_RalphaDM_04_V_exact.output crdChg_RalphaDM_04_V_pme.output

echo " "
echo "Errors for RalphaDM_05.txt"
../../exe/compareV crdChg_RalphaDM_04_V_exact.output crdChg_RalphaDM_05_V_pme.output

echo " "
echo "Errors for RalphaDM_06.txt"
../../exe/compareV crdChg_RalphaDM_04_V_exact.output crdChg_RalphaDM_06_V_pme.output

echo " "
echo "Errors for RalphaDM_07.txt"
../../exe/compareV crdChg_RalphaDM_04_V_exact.output crdChg_RalphaDM_07_V_pme.output

echo " "
echo "Errors for RalphaDM_08.txt"
../../exe/compareV crdChg_RalphaDM_04_V_exact.output crdChg_RalphaDM_08_V_pme.output

echo " "
echo "Errors for RalphaDM_09.txt"
../../exe/compareV crdChg_RalphaDM_04_V_exact.output crdChg_RalphaDM_09_V_pme.output

echo " "
echo "Errors for RalphaDM_10.txt"
../../exe/compareV crdChg_RalphaDM_04_V_exact.output crdChg_RalphaDM_10_V_pme.output

echo " "
echo "Errors for RalphaDM_11.txt"
../../exe/compareV crdChg_RalphaDM_04_V_exact.output crdChg_RalphaDM_11_V_pme.output



echo "---------------------------------------------------------------------"
echo " "
echo "Exact Time:"
grep time_exact crdChg_RalphaDM_04.log
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


