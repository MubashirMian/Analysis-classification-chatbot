
from transformers import pipeline
import torch
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import os
import sys
import pathlib
import nltk
#from Naruto.utils import load_subtitles_dataset
from Naruto.utils.data_loader import load_subtitles_dataset
nltk.download('punkt')
nltk.download('punkt_tab')

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))


import os
import pandas as pd
import numpy as np
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import torch
from utils.data_loader import load_subtitles_dataset  # Adjust import as per your structure


class ThemeClassifier:
    def __init__(self, theme_list):
        self.model_name = "facebook/bart-large-mnli"

        # Check for device compatibility
        if torch.backends.mps.is_available():  # Check for MPS (Apple Silicon)
            self.device = torch.device("mps")
            print("Using MPS device")
        elif torch.cuda.is_available():  # Check for CUDA (less likely on macOS)
            self.device = torch.device("cuda")
            print("Using CUDA device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")

        print(f"Using device: {self.device}")

        self.theme_list = theme_list
        self.theme_classifier = self.load_model()

    def load_model(self):
        try:
            classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if str(self.device) != "cpu" else -1,  # Device index for GPU
                torch_dtype=torch.float16 if str(self.device) != "cpu" else None  # Use float16 for GPU
            )
            return classifier
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_themes_inference(self, script):
        if not self.theme_classifier:
            print("Model not loaded. Unable to perform inference.")
            return {}

        script_sentences = sent_tokenize(script)
        sentence_batch_size = 2

        # Batch sentences
        script_batches = []
        for index in range(0, len(script_sentences), sentence_batch_size):
            sent = " ".join(script_sentences[index:index + sentence_batch_size])
            script_batches.append(sent)

        # Run model
        try:
            theme_output = []
            for batch in script_batches:
                output = self.theme_classifier(batch, self.theme_list, multi_label=True)
                theme_output.append(output)
        except Exception as e:
            print(f"Error during inference: {e}")
            return {}

        # Wrangle results
        themes = {}
        for output in theme_output:
            for label, score in zip(output['labels'], output['scores']):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)

        # Average scores for each theme
        themes = {key: np.mean(value) for key, value in themes.items()}
        return themes

    def get_themes(self, dataset_path, save_path=None):
        # Load dataset
        df = load_subtitles_dataset(dataset_path)

        # Limit for debugging (optional, remove in production)
        df = df.head(2)

        # Run inference
        df['themes'] = df['script'].apply(self.get_themes_inference)
        print('the following data')
        print(df['themes'])
        # Save output
        if save_path is not None:
            try:
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                df.to_csv(save_path, index=False)
                print(f"Saved themes to {save_path}")
            except Exception as e:
                print(f"Error saving file: {e}")

        return df

# class ThemeClassifier:
#     def __init__(self, theme_list, sentence_batch_size=10):
#         self.model_name = "facebook/bart-large-mnli"
#         self.sentence_batch_size = sentence_batch_size

#         # Check for device compatibility
#         if torch.backends.mps.is_available():
#             self.device = torch.device("mps")
#         elif torch.cuda.is_available():
#             self.device = torch.device("cuda")
#         else:
#             self.device = torch.device("cpu")
#         print(f"Using device: {self.device}")

#         self.theme_list = theme_list
#         self.theme_classifier = self.load_model()

#     def load_model(self):
#         try:
#             classifier = pipeline(
#                 "zero-shot-classification",
#                 model=self.model_name,
#                 device=0 if self.device.type != "cpu" else -1
#             )
#             return classifier
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             return None

#     def get_themes_inference(self, script):
#         script_sentences = sent_tokenize(script)
#         try:
#             theme_output = self.theme_classifier(
#                 script_sentences, self.theme_list, multi_label=True
#             )
#         except Exception as e:
#             print(f"Error during inference: {e}")
#             return {}

#         # Aggregate results
#         from collections import defaultdict
#         themes = defaultdict(lambda: {"sum": 0, "count": 0})
#         for output in theme_output:
#             for label, score in zip(output['labels'], output['scores']):
#                 themes[label]["sum"] += score
#                 themes[label]["count"] += 1

#         return {key: value["sum"] / value["count"] for key, value in themes.items()}

#     def get_themes(self, dtaset_path, save_path=None):
#         # if save_path and os.path.exists(save_path):
#         #     return pd.read_csv(save_path)

#         try:
#             df = load_subtitles_dataset(dtaset_path)
#         except Exception as e:
#             print(f"Error loading dataset: {e}")
#             return None

#         output_themes = df['script'].apply(self.get_themes_inference)
#         themes_df = pd.DataFrame(output_themes.tolist())
#         df[themes_df.columns] = themes_df

#         if save_path:
#             df.to_csv(save_path, index=False)
#         return df



