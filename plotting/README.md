Tested on lpc el8. Work in CMSSW_12_3_5 in order to fetch python version with the few required modules.

To submit jobs to condor: `python3 makeCondorArgs.py`, if there are jobs to run, it will output `condor_submit jdl.txt` which is what one should manually execute then.
If all H5 files have been converted, it merged them in the `merged_output` directory. 

Conversion of merged .csv files to histograms (templates): `csv_to_histroot.py` 



