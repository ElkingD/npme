#create a file of nCharge = 100,000 random charges and coordinates
../../exe/makeRandomBox input_makeRandomBoxReal.txt

#run npme on the charge/coord file (crdChg.txt) and npme instruction files with
#various direct space cutoffs Rdir = 4.0, 5.0, 6.0, .. 11.0
#note that calcType = pme_exact in laplaceOrig_04.txt which instructs npme
#to calculate both the NPME and exact potential/potential gradient

echo " "
../../exe/npme crdChg.txt laplaceOrig_04.txt
echo " "
../../exe/npme crdChg.txt laplaceOrig_06.txt
echo " "
../../exe/npme crdChg.txt laplaceOrig_08.txt
echo " "
../../exe/npme crdChg.txt laplaceOrig_10.txt
echo " "
../../exe/npme crdChg.txt laplaceOrig_12.txt
echo " "


echo " "
echo "Errors for laplaceOrig_04.txt"
../../exe/compareV crdChg_laplaceOrig_04_V_exact.output crdChg_laplaceOrig_04_V_pme.output

echo " "
echo "Errors for laplaceOrig_06.txt"
../../exe/compareV crdChg_laplaceOrig_04_V_exact.output crdChg_laplaceOrig_06_V_pme.output

echo " "
echo "Errors for laplaceOrig_08.txt"
../../exe/compareV crdChg_laplaceOrig_04_V_exact.output crdChg_laplaceOrig_08_V_pme.output

echo " "
echo "Errors for laplaceOrig_10.txt"
../../exe/compareV crdChg_laplaceOrig_04_V_exact.output crdChg_laplaceOrig_10_V_pme.output

echo " "
echo "Errors for laplaceOrig_12.txt"
../../exe/compareV crdChg_laplaceOrig_04_V_exact.output crdChg_laplaceOrig_12_V_pme.output


echo "---------------------------------------------------------------------"
echo " "
echo "Exact Time:"
grep time_exact crdChg_laplaceOrig_04.log
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


