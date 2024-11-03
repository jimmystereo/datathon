import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Import tqdm for progress bar

def extract(file_path):
    try:
        file_name = os.path.basename(file_path).replace('.csv', '')
        word = file_name.split('_')[0]
        languages = file_name.split('_')[1].split('-')
        prime = languages[0]
        target = languages[1]
        word_type = file_name.split('_')[-2]
        participant = int(file_name.split('_')[-1])

        # Load and augment the DataFrame
        df = pd.read_csv(file_path)
        df = df.reset_index()  # Reset index without 'name' argument
        df.rename(columns={'index': 'time'}, inplace=True)  # Rename the new index column to 'time'

        df['word'] = word
        df['prime'] = prime
        df['target'] = target
        df['participant'] = participant
        df['word_type'] = word_type

        return df
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

# Directory containing the files
directory = "/Users/jimmy/Documents/GitHub/datathon/codes/EEG_Measurements"
meta_data = pd.read_csv('codes/metadata.csv')

# Function to process all files concurrently
def process_files_in_directory(directory):
    output = pd.DataFrame()
    file_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".csv")]

    # Use ThreadPoolExecutor to process files in parallel and display progress with tqdm
    with ThreadPoolExecutor(max_workers=800) as executor:  # Adjust max_workers based on CPU capacity
        futures = {executor.submit(extract, file_path): file_path for file_path in file_paths}

        # Track the progress using tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            df = future.result()
            if df is not None:
                output = pd.concat([output, df], ignore_index=True)

    return output

# Run the parallel processing
output = process_files_in_directory(directory)

# Merge with metadata
output = output.merge(meta_data, on='participant', how='left')

# Example: Display or save the final output
print(output.head())
# output.to_csv('full_data.csv', encoding = 'utf-8-sig', index=False)  # Save to a CSV if desired

# Load the CSV and create labels
# df = pd.read_csv('spanish_short.csv')
features = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
            'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6',
            'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2']
for f in features:
    df_f = output.pivot(
        index=['word', 'participant', 'prime', 'target', 'word_type'],  # Rows
        columns='time',  # Use the unique index as columns
        values=f  # Values to aggregate
    ).reset_index()
    df_f.to_csv(f'transposed/{f}.csv', encoding = 'utf-8-sig', index=False)  # Save to a CSV if desired

