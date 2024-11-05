import pandas as pd
import os


def extract(file_path):
    file_name = file_path.split('/')[-1].replace('.csv', '')
    word = file_name.split('_')[0]
    languages = file_name.split('_')[1].split('-')
    prime = languages[0]
    target = languages[1]
    word_type = file_name.split('_')[-2]
    participant = int(file_name.split('_')[-1])
    df = pd.read_csv(file_path)
    df['word'] = word
    df['prime'] = prime
    df['target'] = target
    df['participant'] = participant
    df['word_type'] = word_type
    df = df.reset_index()  # Reset index without 'name' argument
    df.rename(columns={'index': 'time'}, inplace=True)  # Rename the new index column to 'time'

    return df

# Directory containing the files
directory = "/Users/jimmy/Documents/GitHub/datathon/codes/EEG_Measurements"
meta_data = pd.read_csv('codes/metadata.csv')

output = pd.DataFrame()
c = 0
# Loop through each file in the directory
for filename in os.listdir(directory):
    # Process only CSV files (adjust if needed)
    try:
        if filename.endswith(".csv"):
            if 'spanish' not in filename:
                continue
            print(filename)
            # Full path to the file
            file_path = os.path.join(directory, filename)
            df = extract(file_path)
            output = pd.concat([output, df])
            c+=1
    except Exception as e:
        print('Failed: ', filename)
    # if c == 10:
    #     break


# dft = pd.read_csv('data/spanish.csv')
output = output.merge(meta_data, on='participant', how='left')
output.to_csv('data/spanish2.csv', encoding = 'utf-8-sig', index = False)
features = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
            'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6',
            'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2']
for f in features:
    df_f = output.pivot(
        index=['word', 'participant', 'prime', 'target', 'word_type', "spanish","french","german","other"],  # Rows
        columns='time',  # Use the unique index as columns
        values=f  # Values to aggregate
    ).reset_index()
    df_f.to_csv(f'data/spanish/{f}.csv', encoding = 'utf-8-sig', index=False)  # Save to a CSV if desired


# df