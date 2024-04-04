import numpy as np
import ROOT as r
import csv

datasets = {
    "2016APV":["data_obs","TTToHadronic"],
    "2016":["data_obs","TTToHadronic"],
    "2017":["MX2400_MY100","data_obs","TTToHadronic"],
    "2018":["data_obs","TTToHadronic"],
    "run2":["data_obs"]
}

def make_templates(csvreader,process,year,region,weights):
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

def vae_hist(csvreader,process,year,region):
    name = f"{process}_{year}_{region}"
    h = r.TH2F(f"vae_loss_{name}","",100,0,0.0001)
    for i,row in enumerate(csvreader):
        vae_loss = float(row[3])
        if process=="data_obs":
            w = 1.
        else:
            w = float(row[4])#nominal weight
            h.Fill(vae_loss,w)
    return h

def convert_region_nom(process,year,region):
    data_flag       = False
    if process=="data_obs" or "JetHT" in process:
        data_flag   = True
    
    if data_flag:
        csvfile    = open(f"merged_output/{process}_{year}_{region}.csv")
    else:
        csvfile    = open(f"merged_output/{process}_{year}_{region}_nom.csv")
    csvreader  = csv.reader(csvfile,delimiter=",")
    variations = ["nom"]
    for variation in ["pdf","prefire","pileup","PS_ISR","PS_FSR","F","R","RF","top_ptrw","pnet"]:
        if data_flag:
            break
        variations.append(f"{variation}_up")
        variations.append(f"{variation}_down")
    histos  = make_templates(csvreader,process,year,region,variations)
    return histos


def convert_region_jecs(process,year,region,jec):
    csvfile    = open(f"merged_output/{process}_{year}_{region}_{jec}.csv")
    csvreader  = csv.reader(csvfile,delimiter=",")
    histos     = make_templates(csvreader,process,year,region,[jec])
    return histos

column_names = {
    'mjj': 0,
    'mh': 1,
    'my' : 2,
    'vae_loss': 3,
    'nom' : 4,
    'pdf_up' : 5,
    'pdf_down': 6,
    'prefire_up': 7,
    'prefire_down' : 8,
    'pileup_up' : 9 ,
    'pileup_down' : 10,
    'btag_up' : 11,
    'btag_down' : 12,
    'PS_ISR_up' : 13,
    'PS_ISR_down' : 14,
    'PS_FSR_up' : 15,
    'PS_FSR_down' : 16,
    'F_up' : 17,
    'F_down' : 18,
    'R_up' : 19,
    'R_down' : 20,
    'RF_up' : 21,
    'RF_down' : 22,
    'top_ptrw_up' : 23,
    'top_ptrw_down' : 24,
    'pnet_up' : 25,
    'pnet_down' : 26,
    'jes_up' : 4,
    'jes_down' : 4,
    'jer_up' : 4,
    'jer_down' : 4,    
    'jms_up' : 4,
    'jms_down' : 4,
    'jmr_up' : 4,
    'jmr_down' : 4 
}

histos = []
jecs = ["jes_up","jes_down","jer_up","jer_down","jms_up","jms_down","jmr_up","jmr_down"]

#for year in datasets:
for year in ["2017"]:
    print(year)
    for process in datasets[year]:
        print(process)
        for region in ["SR_Pass","SR_Fail","CR_Pass","CR_Fail"]:
            histos.extend(convert_region_nom(process,year,region).values())
            if not ("TTToHadronic" in process or "MX" in process):
                continue
            for jec in jecs:
                histos.extend(convert_region_jecs(process,year,region,jec).values())
f = r.TFile.Open("histograms.root","RECREATE")
f.cd()
for histo in histos:
    histo.Write()
f.Close()