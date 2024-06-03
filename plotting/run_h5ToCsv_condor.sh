#!/bin/bash
echo "Starting h5 to csv conversion"
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc10
scramv1 project CMSSW CMSSW_12_3_5


cd CMSSW_12_3_5
eval `scramv1 runtime -sh`
cd ..

pwd
tar -xf tarball.tgz; rm -f tarball.tgz
mkdir output

ls
echo python3 h5ToCsv.py $*
python3 h5ToCsv.py $*
