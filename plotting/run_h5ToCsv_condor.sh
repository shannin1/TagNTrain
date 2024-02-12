#!/bin/bash
echo "Starting h5 to csv conversion"
source /cvmfs/cms.cern.ch/cmsset_default.sh
xrdcp root://cmseos.fnal.gov//store/user/roguljic/H5_env.tgz ./
export SCRAM_ARCH=slc7_amd64_gcc900
scramv1 project CMSSW CMSSW_11_3_4
tar -xzf H5_env.tgz
rm -f H5_env.tgz


cd CMSSW_11_3_4
eval `scramv1 runtime -sh`
cd ..

pwd
tar -xf tarball.tgz; rm -f tarball.tgz
mkdir output

ls
echo python3 h5ToCsv_condor.py $*
python3 h5ToCsv_condor.py $*
