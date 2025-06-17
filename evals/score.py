import pandas as pd
import os
from glob import glob

# Root results folder
root_folder = "results"

# Find all *_results.csv files
csv_files = glob(os.path.join(root_folder, "**/*_results.csv"), recursive=True)

# Output list
output_rows = []

for csv_file in csv_files:
    try:
        # Read CSV
        df = pd.read_csv(csv_file)

        # Sanity check
        if "custom_id" in df.columns and "BERTScore_F1" in df.columns and "AlignScore" in df.columns:
            # Extract custom_id_prefix (before first dash)
            df["custom_id_prefix"] = df["custom_id"].apply(lambda x: x.split("-")[0] + "-test")

            # Group by custom_id_prefix
            agg_df = df.groupby("custom_id_prefix").agg({
                "BERTScore_F1": "mean",
                "AlignScore": "mean"
            }).reset_index()

            # Add rows to output
            for _, row in agg_df.iterrows():
                output_rows.append({
                    "relative_path_of_file": csv_file,
                    "custom_id_prefix": row["custom_id_prefix"],
                    "avgBertScore": row["BERTScore_F1"],
                    "avgAlignScore": row["AlignScore"]
                })
        else:
            print(f"Skipping {csv_file} (missing required columns)")
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

# Convert to DataFrame
output_df = pd.DataFrame(output_rows)

# Save to CSV
output_filename = "results/aggregated_results_per_dataset.csv"
output_df.to_csv(output_filename, index=False)

print(f"\nDone! Aggregated results saved to {output_filename}")