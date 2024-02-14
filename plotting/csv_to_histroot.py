# http://scikit-hep.org/root_numpy/reference/generated/root_numpy.array2tree.html

from root_numpy import array2tree
import numpy as np
import ROOT as r
import csv

datasets = {
    "2017":["MX2400_MY100","data_obs","TTToHadronic"]
}

def make_histos(csvreader,process,year,region,weights):
    histos = {}
    for weight in weights:
        name   = f"{process}_{year}_{region}_{weight}"
        h2     = r.TH2F(f"mjj_my_{name}","",40,1000,3000,20,0,1000)
        histos[weight] = h2
    for i,row in enumerate(csvreader):
       mjj = float(row[0])
       #mh  = row[1] #Not needed
       my  = float(row[2])
       for weight in weights:
            if process=="data_obs":
                w = 1.
            else:
                w = float(row[column_names[weight]])
            histos[weight].Fill(mjj,my,w)
    return histos

def convert_region(process,year,region):
    csvfile    = open(f"merged_output/{process}_{year}_{region}.csv")
    csvreader  = csv.reader(csvfile,delimiter=",")
    variations = ["nom"]
    for variation in ["pdf","prefire","pileup","PS_ISR","PS_FSR","F","R","RF","top_ptrw"]:
        if process=="data_obs":
            break
        variations.append(f"{variation}_up")
        variations.append(f"{variation}_down")
    histos  = make_histos(csvreader,process,year,region,variations)
    return histos

column_names = {
        'mjj': 0,
        'mh': 1,
        'my' : 2,
        'nom' : 3,
        'pdf_up' : 4,
        'pdf_down': 5,
        'prefire_up': 6,
        'prefire_down' : 7,
        'pileup_up' : 8 ,
        'pileup_down' : 9,
        'btag_up' : 10,#These are dummy
        'btag_down' : 11,
        'PS_ISR_up' : 12,
        'PS_ISR_down' : 13,
        'PS_FSR_up' : 14,
        'PS_FSR_down' : 15,
        'F_up' : 16,
        'F_down' : 17,
        'R_up' : 18,
        'R_down' : 19,
        'RF_up' : 20,
        'RF_down' : 21,
        'top_ptrw_up' : 22,#These are dummy for non-ttbar events
        'top_ptrw_down' : 23,
    }

histos = []
for year in datasets:
    print(year)
    for process in datasets[year]:
        print(process)
        for region in ["SR_Pass","SR_Loose","SR_Fail","CR_Pass","CR_Loose","CR_Fail"]:
            histos.extend(convert_region(process,year,region).values())
f = r.TFile.Open("histograms.root","RECREATE")
f.cd()
for histo in histos:
    histo.Write()
f.Close()