import ROOT

# open files
processes = ['MX2400_MY100','MX2400_MY250','MX2400_MY350','MX1600_MY150','MX2000_MY250','MX3000_MY190','MX3000_MY300','MX3000_MY400','MX2800_MY100','MX2800_MY190','MX2600_MY300','TTToHadronic']
for process in processes:
    infile = ROOT.TFile.Open('../analysis_note_datasets_JME_SR/2dhistos_scaled/{}_JMEs_SR.root'.format(process),'READ')
    #QCD = ROOT.TFile.Open('correctHistos/2dhist_TightSRMX3000_MY300.root','READ')

    # get bkg distributions 
    pass_JERdown = infile.Get('PmjjvsPmY_SR_pass_JER_down')
    pass_JERup = infile.Get('PmjjvsPmY_SR_pass_JER_up')
    pass_JESup = infile.Get('PmjjvsPmY_SR_pass_JES_up')
    pass_JESdown = infile.Get('PmjjvsPmY_SR_pass_JES_down')
    pass_JMRdown = infile.Get('PmjjvsPmY_SR_pass_JMR_down')
    pass_JMRup = infile.Get('PmjjvsPmY_SR_pass_JMR_up')
    pass_JMSup = infile.Get('PmjjvsPmY_SR_pass_JMS_up')
    pass_JMSdown = infile.Get('PmjjvsPmY_SR_pass_JMS_down')

    loose_JERdown = infile.Get('LmjjvsLmY_SR_loose_JER_down')
    loose_JERup = infile.Get('LmjjvsLmY_SR_loose_JER_up')
    loose_JESup = infile.Get('LmjjvsLmY_SR_loose_JES_up')
    loose_JESdown = infile.Get('LmjjvsLmY_SR_loose_JES_down')
    loose_JMRdown = infile.Get('LmjjvsLmY_SR_loose_JMR_down')
    loose_JMRup = infile.Get('LmjjvsLmY_SR_loose_JMR_up')
    loose_JMSup = infile.Get('LmjjvsLmY_SR_loose_JMS_up')
    loose_JMSdown = infile.Get('LmjjvsLmY_SR_loose_JMS_down')

    fail_JERdown = infile.Get('FmjjvsFmY_SR_fail_JER_down')
    fail_JERup = infile.Get('FmjjvsFmY_SR_fail_JER_up')
    fail_JESup = infile.Get('FmjjvsFmY_SR_fail_JES_up')
    fail_JESdown = infile.Get('FmjjvsFmY_SR_fail_JES_down')
    fail_JMRdown = infile.Get('FmjjvsFmY_SR_fail_JMR_down')
    fail_JMRup = infile.Get('FmjjvsFmY_SR_fail_JMR_up')
    fail_JMSup = infile.Get('FmjjvsFmY_SR_fail_JMS_up')
    fail_JMSdown = infile.Get('FmjjvsFmY_SR_fail_JMS_down')

    # TH2::SetDirectory(0) if you don't want a histogram to be added to any directory, so that
    # you can open and close root files without garbage collecting the histogram
    pass_JERdown.SetDirectory(0)
    pass_JERup.SetDirectory(0)
    pass_JESup.SetDirectory(0) 
    pass_JESdown.SetDirectory(0)
    pass_JMRdown.SetDirectory(0) 
    pass_JMRup.SetDirectory(0)
    pass_JMSup.SetDirectory(0) 
    pass_JMSdown.SetDirectory(0)

    loose_JERdown.SetDirectory(0)
    loose_JERup.SetDirectory(0)
    loose_JESup.SetDirectory(0) 
    loose_JESdown.SetDirectory(0)
    loose_JMRdown.SetDirectory(0) 
    loose_JMRup.SetDirectory(0)
    loose_JMSup.SetDirectory(0) 
    loose_JMSdown.SetDirectory(0)

    fail_JERdown.SetDirectory(0)
    fail_JERup.SetDirectory(0)
    fail_JESup.SetDirectory(0) 
    fail_JESdown.SetDirectory(0)
    fail_JMRdown.SetDirectory(0) 
    fail_JMRup.SetDirectory(0)
    fail_JMSup.SetDirectory(0) 
    fail_JMSdown.SetDirectory(0)  

    # close the files so they won't get written to accidentally
    infile.Close()

    #out = ROOT.TFile.Open('2dhist_TightSRMX3000_MY300_rescaled.root','RESREATE')
    out = ROOT.TFile.Open('2dhistos_scaled/{}_SR.root'.format(process),'UPDATE')

    # SReate new Pass and Fail histos
    pass_JERdown = pass_JERdown.Clone('PmjjvsPmY_SR_pass_JER_down')
    pass_JERup = pass_JERup.Clone('PmjjvsPmY_SR_pass_JER_up')
    pass_JESup = pass_JESup.Clone('PmjjvsPmY_SR_pass_JES_up') 
    pass_JESdown = pass_JESdown.Clone('PmjjvsPmY_SR_pass_JES_down')
    pass_JMRdown = pass_JMRdown.Clone('PmjjvsPmY_SR_pass_JMR_down')
    pass_JMRup = pass_JMRup.Clone('PmjjvsPmY_SR_pass_JMR_up')
    pass_JMSup = pass_JMSup.Clone('PmjjvsPmY_SR_pass_JMS_up') 
    pass_JMSdown = pass_JMSdown.Clone('PmjjvsPmY_SR_pass_JMS_down')

    loose_JERdown = loose_JERdown.Clone('LmjjvsLmY_SR_loose_JER_down')
    loose_JERup = loose_JERup.Clone('LmjjvsLmY_SR_loose_JER_up')
    loose_JESup = loose_JESup.Clone('LmjjvsLmY_SR_loose_JES_up') 
    loose_JESdown = loose_JESdown.Clone('LmjjvsLmY_SR_loose_JES_down')
    loose_JMRdown = loose_JMRdown.Clone('LmjjvsLmY_SR_loose_JMR_down')
    loose_JMRup = loose_JMRup.Clone('LmjjvsLmY_SR_loose_JMR_up')
    loose_JMSup = loose_JMSup.Clone('LmjjvsLmY_SR_loose_JMS_up') 
    loose_JMSdown = loose_JMSdown.Clone('LmjjvsLmY_SR_loose_JMS_down')

    fail_JERdown = fail_JERdown.Clone('FmjjvsFmY_SR_fail_JER_down')
    fail_JERup = fail_JERup.Clone('FmjjvsFmY_SR_fail_JER_up')
    fail_JESup = fail_JESup.Clone('FmjjvsFmY_SR_fail_JES_up') 
    fail_JESdown = fail_JESdown.Clone('FmjjvsFmY_SR_fail_JES_down')
    fail_JMRdown = fail_JMRdown.Clone('FmjjvsFmY_SR_fail_JMR_down') 
    fail_JMRup = fail_JMRup.Clone('FmjjvsFmY_SR_fail_JMR_up')
    fail_JMSup = fail_JMSup.Clone('FmjjvsFmY_SR_fail_JMS_up') 
    fail_JMSdown = fail_JMSdown.Clone('FmjjvsFmY_SR_fail_JMS_down') 

    #newPass.Reset()
    #newPass.SetDirectory(0)
    #newPass.Add(QCDpass)

    # cd to output file to write histos
    out.cd()
    pass_JERdown.Write()
    pass_JERup.Write()
    pass_JESup.Write() 
    pass_JESdown.Write()
    pass_JMRdown.Write() 
    pass_JMRup.Write()
    pass_JMSup.Write() 
    pass_JMSdown.Write()

    loose_JERdown.Write()
    loose_JERup.Write()
    loose_JESup.Write() 
    loose_JESdown.Write()
    loose_JMRdown.Write() 
    loose_JMRup.Write()
    loose_JMSup.Write() 
    loose_JMSdown.Write()

    fail_JERdown.Write()
    fail_JERup.Write()
    fail_JESup.Write() 
    fail_JESdown.Write()
    fail_JMRdown.Write()
    fail_JMRup.Write()
    fail_JMSup.Write()
    fail_JMSdown.Write() 

    out.Close()
