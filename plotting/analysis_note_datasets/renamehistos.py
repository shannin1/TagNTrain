import ROOT as r
processes = ['MX2400_MY100','MX2400_MY250','MX2400_MY350','MX1600_MY150','MX2000_MY250','MX3000_MY190','MX3000_MY300','MX3000_MY400','MX2800_MY100','MX2800_MY190','MX2600_MY300','TTToHadronic']
for process in processes: 
    f=r.TFile.Open('2dhistos_scaled/{}_CR.root'.format(process))
    histos = []
    histnames = []
    for key in f.GetListOfKeys():
        
        name = key.ReadObj()
        hName = name.GetName()
        h = f.Get(hName)
        print(hName)
        rightside = "mjjvsmY_" + hName.split('mY_')[-1]
        if "fail" in rightside:
            newname = rightside.replace('fail', 'Fail')
        if "loose" in rightside:
            newname = rightside.replace('loose', 'Loose')
        if "pass" in rightside:
            newname = rightside.replace('pass', 'Pass')
        h.SetName(newname)
        h.SetDirectory(0) # Not 100% sure if this is needed, but I think it is
        histos.append(h)
    g = r.TFile.Open("2dhistos_scaled_renamed/{}_CR.root".format(process),"RECREATE")
    g.cd()
    for h in histos:
        h.Write()
    g.Close()
    f.Close()