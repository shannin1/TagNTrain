import ROOT

# open files
process = 'MX1600_MY150'
QCD = ROOT.TFile.Open('{}_JMEs_CR.root'.format(process),'READ')
#QCD = ROOT.TFile.Open('correctHistos/2dhist_TightCRMX3000_MY300.root','READ')

# get bkg distributions 
QCDpass = QCD.Get('PmjjvsPmY_CR_pass_JES_down')
QCDfail = QCD.Get('FmjjvsFmY_CR_fail_JES_down')
QCDloose = QCD.Get('LmjjvsLmY_CR_loose_JES_down')

# TH2::SetDirectory(0) if you don't want a histogram to be added to any directory, so that
# you can open and close root files without garbage collecting the histogram
QCDpass.SetDirectory(0)
QCDloose.SetDirectory(0)
QCDfail.SetDirectory(0)


# close the files so they won't get written to accidentally
QCD.Close()

#out = ROOT.TFile.Open('2dhist_TightCRMX3000_MY300_rescaled.root','RECREATE')
out = ROOT.TFile.Open('{}_CR.root'.format(process),'UPDATE')

# create new Pass and Fail histos
newFail = QCDfail.Clone()	# clone so that we get same binning and title


newLoose = QCDloose.Clone()	# clone so that we get same binning and title


newPass = QCDpass.Clone()

#newPass.Reset()
#newPass.SetDirectory(0)
#newPass.Add(QCDpass)

# cd to output file to write histos
out.cd()
newPass.Write()
newFail.Write()
newLoose.Write()

out.Close()
