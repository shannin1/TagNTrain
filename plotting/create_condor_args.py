import os
import subprocess

datasets_n_jobs = {
    "2017":{
        "JetHT_Run2017B":60,
        "JetHT_Run2017C":60,
        "JetHT_Run2017D":60,
        "JetHT_Run2017E":60,
        "JetHT_Run2017F":60,
        "JetHT_Run2017F1":10,
        "TTToHadronic":1,
        "MX2400_MY100":1
        }
    }

def merge_csvs(dataset,year,n_jobs):
    regions     = ["SR","CR"]
    categories  =  ["Pass","Loose","Fail"]
    for region in regions:
        for category in categories:
            cat_cmd = f"cat {dataset}_{year}_{region}_{category}_*_{n_jobs}.csv > merged_output/{dataset}_{year}_{region}_{category}.csv"
            subprocess.call(cat_cmd,shell=True)

def check_other_regions(filename):
    #Assumes that we pass SR_Pass!
    replacers = ["SR_Loose","SR_Fail","CR_Pass","CR_Loose","CR_Fail"]
    for replacer in replacers:
        file_to_check = filename.replace("SR_Pass",replacer)
        if not os.path.exists(file_to_check):
            #This should not really happen
            print(f"Missing {file_to_check}, exiting")
            exit()

def move_individual_files(dataset,year):
    regions = ["SR_Pass","SR_Loose","SR_Fail","CR_Pass","CR_Loose","CR_Fail"]
    for region in regions:
        mv_cmd = f"mv {dataset}_{year}_{region}_*csv single_output/. 2>/dev/null"
        #2>/dev/null supresses mv errors if files have already been moved
        try:
            subprocess.call(mv_cmd,shell=True)
        except:
            continue

def write_arguments(args_to_write):
    if args_to_write:
        f = open("h5ToCsv_args.txt","w")
        for line in args_to_write:
            f.write(line)
        f.close()

        print("Creating tarball")
        subprocess.call("tar cf tarball.tgz h5ToCsv_condor.py jrand_autoencoder_m2500.h5 ../utils",shell=True)
        print("condor_submit jdl.txt")
    else:
        print("All processed")

def merge_data(year):
    #Assumes that we pass SR_Pass!
    dir_list = os.listdir("merged_output")
    if not any(("JetHT" in name) and (year in name) for name in dir_list):
        print(f"Did not find data for {year} in merged_output")
        return
    regions = ["SR_Pass","SR_Loose","SR_Fail","CR_Pass","CR_Loose","CR_Fail"]
    for region in regions:
        cat_cmd = f"cat merged_output/JetHT*{year}_{region}.csv > merged_output/data_obs_{year}_{region}.csv"
        subprocess.call(cat_cmd,shell=True)

args_to_write = []
for year in datasets_n_jobs:
    for dataset in datasets_n_jobs[year]:
        print("-----------")
        print(dataset, year)
        final_file = f"merged_output/{dataset}_{year}_SR_Pass.csv"
        
        #Check for merged file
        if os.path.exists(final_file):
            check_other_regions(final_file)
            print(f"{dataset} {year} processed and merged, continuing")
            move_individual_files(dataset,year)
            continue
        n_jobs = datasets_n_jobs[year][dataset]

        temp_args = []
        #Check for individual files
        for i in range(n_jobs):
            filename = f"{dataset}_{year}_SR_Pass_{i}_{n_jobs}.csv"
            if os.path.exists(filename):
                check_other_regions(filename)
            else:
                temp_args.append(f"{dataset} {year} {i} {n_jobs}\n")
        
        if temp_args:
            print("{} jobs to send".format(len(temp_args)))
            args_to_write.extend(temp_args)
        else:
            print("Merging csvs")
            merge_csvs(dataset,year,n_jobs)
    merge_data(year)
write_arguments(args_to_write)
