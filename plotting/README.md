All the csv and 2dhistos discussed here are actually saved into this github repo which is great.

For Monte Carlo Signal and TTBar I first run "topscoresplots3.py" which used the autoencoder located at 
"/uscms/home/mchitoto/nobackup/XtoHY/CMSSW_11_1_4/src//CASEUtils/jet_images/AEmodels/AEs/jrand_autoencoder_m2500.h5" 
to get an autoencoder score for each jet and does SR and CR selections for all the SIGNAL and BKG MC. These jetimages are stored in
"/uscms/home/mchitoto/nobackup/XtoHY/CMSSW_11_1_4/src/CASEUtils/jet_images/analysis_note_jets/"
The lines 67 and 68 are to be replaced by lines 64 and 65 based on whether its the CR or CR that we are working to get and the limes 175 to 190 are renamed apparopriately 
for CR or SR regions. 

For the data its similar but now  "topscoresplots6.py" is the one to run. These two scripts spit out an array of masses that have passed these selections.
The locations these scripts spit out the csv scripts containing the masses of can be seen in the scripts. 

Next we have to save the systematics as well but for the MC only. The script "topscoresplots8.py" does this for the event weight systematics. For each systematic does selections 
similar to those done but the original "topscoresplots3.py" script and so the events that survive are identical to the ones before but for each of these events it sinstead spits 
out the nom_weight, sys_up and sys_down. Inside "analysis_note_datasets"  for CR and "analysis_note_datasets" for SR we have the script readcsv_make2DHist.py which uses the info
above to make MJJvsMY 2dhisto templates for the nomweight as well as the event weight variation systematics these are all saved into the "2dhistos_scaled" in that folder. Next 
 after we create the JME variations templates discussed in the paragraph be low we run the script "rescale_2dhist.py". This script takes the JME variation templates discussed
 below from where the code described below saves them and saves combines them with the event weight variations described by this paragraph and saves the combined rootfiles into 
 "2dhistos_scaled" in that folder. We then run the script renamehistos.py to tweak the names of the 2histo templates to a form that will be very useful for when we actually 
 run 2dalphabet.  

To calculate the Jet Mass and Energy variations(JME_vars), we run the script "topscoresplots9.py". This however is different from the one before because it recalculates the 
Jet Invariant Dijet Masses based on the up and down jet pt and jet masses. It does regular selections including only keeping events with up or down pt>300 which is a
cut we imposed in preselections but may be violated by the up and down pt. This script spits out a new csv file with corrected mmasses of the selected events as well as their corresponding
event weight for each event. Inside of "analysis_note_datasets_JME_CR" we have a script "readcsv_make2DHist.py" that takes the info saved baout the masses and event weights for each 
JME_vars and spits out rootfiles with MJJvsMY 2dhistogram templates with the up and down variations by the JME_vars. 

For the data once we run "topscoresplots6.py" we get chunks of csv files that we have to merge, and for that we use 




