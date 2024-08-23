import ROOT as r
import os
import numpy as np
import math
import mplhep as hep
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import numpy as np

base_path='/uscms_data/d3/roguljic/el8_anomalous/el9_fitting/templates_v4/templates_'
MXs=['1200','1400','1600','1800','2000','2200','2400','2500','2600','2800','3000','3500','4000']
MYs=['90','125','190','250','300','400']
years=['2016','2016APV','2017','2018']

generated_events=5*138

def get_efficiency(MX,MY):
    SR_Pass_events=0
    for year in years:
        filepath=base_path+'MX'+MX+'_MY'+MY+'_'+year+'.root'

        file=r.TFile.Open(filepath)
        keyname='mjj_my_MX'+MX+'_MY'+MY+'_'+year+'_SR_Pass_nom'

        h=file.Get(keyname)

        SR_Pass_events+=h.Integral()


    efficiency=SR_Pass_events/generated_events

    return efficiency

def Plot(Masses,Efficiencies,MXorMY,ConstantVal):
    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(12,9))

    plt.plot(Masses,Efficiencies,'o',color='r')
    
    yTitle='Efficiency'
    xTitle=MXorMY

    plt.xlabel(xTitle, horizontalalignment='right', x=1.0)
    plt.ylabel(yTitle,horizontalalignment='right', y=1.0)
    plt.legend(loc='upper right',ncol=2)
    if MXorMY=='MX':
        plt.title("MX vs. Efficiency with MY= "+ConstantVal)
        plt.savefig('MX_vs_Eff_MY'+ConstantVal+'.png')
        print('Saving plot as MX_vs_Eff_MY'+ConstantVal+'.png')
    else:
        plt.title("MY vs. Efficiency with MX= "+ConstantVal)
        plt.savefig('MY_vs_Eff_MX'+ConstantVal+'.png')
        print('Saving plot as MY_vs_Eff_MX'+ConstantVal+'.png')

def MakePlots(MXorMY):
    if MXorMY=='MX':
        for MY in MYs:
            MXstoPlot=[]
            efficiencies=[]
            for MX in MXs:
                filepathcheck=base_path+'MX'+MX+'_MY'+MY+'_2016.root'
                if os.path.exists(filepathcheck):
                    MXstoPlot.append(MX)
                    print('Calculating Efficiency with MX= '+MX+' MY= '+MY)
                    efficiency=get_efficiency(MX,MY)
                    efficiencies.append(efficiency)
            print('Making plot as a function of MX with MY= ',MY)
            Plot(MXstoPlot,efficiencies,MXorMY,MY)
    else:
        for MX in MXs:
            MYstoPlot=[]
            efficiencies=[]
            for MY in MYs:
                filepathcheck=base_path+'MX'+MX+'_MY'+MY+'_2016.root'
                if os.path.exists(filepathcheck):
                    MYstoPlot.append(MY)
                    print('Calculating Efficiency with MX= '+MX+' MY= '+MY)
                    efficiency=get_efficiency(MX,MY)
                    efficiencies.append(efficiency)
            print('Making plot as a function of MY with MX= ',MX)
            Plot(MYstoPlot,efficiencies,MXorMY,MX)
    return
    

if __name__=='__main__':
    print("Making plots as a function of MX")
    MakePlots('MX')

    print("Making plots as a function of MY")
    MakePlots('MY')
