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
    df['']
    df['word'] = word
    df['prime'] = prime
    df['target'] = target
    df['participant'] = participant
    df['word_type'] = word_type
    df = df.reset_index(name = 'time')
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
            print(filename)
            # Full path to the file
            file_path = os.path.join(directory, filename)
            df = extract(file_path)
            output = pd.concat([output, df])
            c+=1
    except Exception as e:
        print('Failed: ', filename)
    # if c == 3:
    #     break
output = output.merge(meta_data, on='participant', how='left')

# df