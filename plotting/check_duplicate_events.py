import pandas as pd

def check_duplicates_in_csv(filename):
    df = pd.read_csv(filename, header=None)

    duplicates = df.duplicated(subset=[0,1]) #based on fist column - evt number

    removed_duplicates = df[duplicates]
    print("Duplicates:")
    print(removed_duplicates)

    # Step 4: Calculate percentage of removed lines
    total_rows = len(df)
    removed_rows = len(removed_duplicates)
    frac = removed_rows / total_rows
    print(f"Fraction of duplicate lines: {frac:.3f}%")


# Example usage:
#check_duplicates_in_csv("merged_output/data_obs_run2_CR_Pass.csv")
check_duplicates_in_csv("merged_output/MX1800_MY90_2016APV_CR_Pass_nom.csv")
