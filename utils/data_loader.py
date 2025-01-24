import pandas as pd
from glob import glob

def load_subtitles_dataset(dataset_path):
    subtitles_paths = glob(dataset_path + '/*.ass')

    scripts = []
    episode_num = []

    for path in subtitles_paths:
        with open(path, 'r', encoding='utf-8') as files:  # Added encoding for safety
            lines = files.readlines()
            lines = lines[27:]
            lines = [",".join(line.split(',')[9:]) for line in lines]

        lines = [line.replace('\\N', ' ') for line in lines]
        script = " ".join(lines)

        # Extract episode number, assumes format like "something-01.ass"
        try:
            episode = int(path.split('-')[-1].split('.')[0].strip())
        except ValueError:
            print(f"Skipping file with invalid format: {path}")
            continue

        scripts.append(script)
        episode_num.append(episode)
        

    df = pd.DataFrame.from_dict({"episode": episode_num, "script": scripts})
    return df
