import pandas as pd

def check_duplicates_in_csv(filename):
    df = pd.read_csv(filename, header=None)

    duplicates = df.duplicated(subset=[0]) #based on fist column - evt number

    removed_duplicates = df[duplicates]
    print("Removed duplicates:")
    print(removed_duplicates)

    # Step 4: Calculate percentage of removed lines
    total_rows = len(df)
    removed_rows = len(removed_duplicates)
    frac = removed_rows / total_rows
    print(f"Fraction of removed lines: {frac:.3f}%")

    # Step 5: Save the cleaned DataFrame back to a new CSV file
    df_cleaned = df[~duplicates]
    df_cleaned.to_csv('test.csv', index=False, header=False)

# Example usage:
#check_duplicates_in_csv("merged_output/data_obs_run2_CR_Pass.csv")
check_duplicates_in_csv("merged_output/TTToHadronic_2018_CR_Fail_nom.csv")
