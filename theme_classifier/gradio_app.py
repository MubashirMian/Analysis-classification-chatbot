import gradio as gr
#from theme_classifier import ThemeClassifier  # Fixed potential typo in the import
#from utils.data_loader import load_subtitles_datasetfrom 
#..utils import load_subtitles_dataset
#from'/Naruto/theme_classifier/theme_classifier' import ThemeClassifier
#from theme_classifier.theme_classifier import ThemeClassifier
from Naruto.theme_classifier.theme_classifier import ThemeClassifier
from Naruto.utils.data_loader import load_subtitles_dataset
from ..utils import load_subtitles_dataset
import streamlit as st
import pandas as pd

# def get_themes(theme_list_str, subtitles_path, save_path):
#     theme_list = theme_list_str.split(',')
#     theme_classifier = ThemeClassifier(theme_list)

#     output_df = theme_classifier.get_themes(subtitles_path, save_path)

#     # Remove "dialogue" from the theme list
#     theme_list = [theme for theme in theme_list if theme != 'dialogue']
#     output_df = output_df[theme_list]

#     # Sum the scores and reset index for the output DataFrame
#     output_df = output_df[theme_list].sum().reset_index()

#     output_df.columns = ['theme', 'score']
    
#     return output_df

# def main():
#     st.title("Theme Classification (Zero Shot Classifiers)")
    
#     # Input fields
#     theme_list_str = st.text_input("Themes", "comedy,drama,action")  # Default value or leave empty
#     subtitles_path = st.text_input("Subtitles or Script Path")
#     save_path = st.text_input("Save Path")
    
#     if st.button("Get Themes"):
#         # Get themes based on user input
#         output_df = get_themes(theme_list_str, subtitles_path, save_path)

#         # Display output dataframe as a table
#         st.write("Themes and Scores", output_df)

#         # Display the plot
#         st.bar_chart(output_df.set_index('theme')['score'])

# if __name__ == '__main__':
#     main()

# def get_themes(theme_list_str, subtitles_path, save_path):
#     theme_list = theme_list_str.split(',')
#     theme_classifier = ThemeClassifier(theme_list)

#     output_df = theme_classifier.get_themes(subtitles_path, save_path)

#     # Remove "dialogue" from the theme list
#     theme_list = [theme for theme in theme_list if theme != 'dialogue']
#     print(theme_list)
#     output_df = output_df[theme_list]

#     # Sum the scores and reset index for the output DataFrame
#     output_df = output_df[theme_list].sum().reset_index()
#     output_df = theme_classifier.get_themes(subtitles_path, save_path)
#     print(output_df.columns)  # Print the columns
#     print(output_df.head()) # Print the first few rows
    
#     output_df.columns = ['theme', 'score']

#     output_char = gr.BarPlot(
#           output_df,
#             x="theme",
#             y="score",
#             title="Series Theme",
#             tooltip=["theme", "score"],
#             vertical=False,
#             width=500,
#             height=260
#       )
#     return output_df
# def get_themes(theme_list_str, subtitles_path, save_path):
#     theme_list = theme_list_str.split(',')
#     theme_classifier = ThemeClassifier(theme_list)

#     output_df = theme_classifier.get_themes(subtitles_path, save_path)
#     print(output_df.columns)  # Print columns after first call

#     # Remove "dialogue" from the theme list and filter the DataFrame
#     theme_list = [theme for theme in theme_list if theme != 'dialogue']
#     print(theme_list)
#     print(output_df)
#     output_df = output_df['theme']

#     output_df = output_df[theme_list]
#     print(output_df)

#     # Sum the scores and reset index for the output DataFrame
#     output_df = output_df.sum(axis=0, numeric_only=True).reset_index()
#     output_df.columns = ['theme', 'score']

#     print(output_df.head())  # Print the first few rows for inspection

#     output_char = gr.BarPlot(
#         output_df,
#         x="theme",
#         y="score",
#         title="Series Theme",
#         tooltip=["theme", "score"],
#         vertical=False,
#         width=500,
#         height=260
#     )
#     return output_df

import pandas as pd
from random import *

def get_themes(theme_list_str, subtitles_path, save_path):
    import pandas as pd
    import gradio as gr

    # Convert theme list from string to list
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)

    # Get themes from the subtitles
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    # Print DataFrame columns for debugging
    print("Initial DataFrame Columns:", output_df.columns)

    # Extract 'themes' column (contains dictionaries)
    theme_scores = pd.DataFrame(output_df['themes'].tolist())

    # Keep only the themes in the theme_list
    theme_scores = theme_scores[theme_list]

    # Sum scores for each theme
    summed_scores = theme_scores.sum(axis=0).reset_index()
    summed_scores.columns = ['theme', 'score']

    # Remove "dialogue" if present in the theme_list
    summed_scores = summed_scores[summed_scores['theme'] != 'dialogue']

    # Debugging outputs
    print("Filtered Theme List:", theme_list)
    print("Summed Theme Scores:\n", summed_scores)

    # Ensure summed_scores has the required structure
    if not {'theme', 'score'}.issubset(summed_scores.columns):
        raise ValueError("The DataFrame does not contain required columns 'theme' and 'score'.")
   
    
   
    # Return the processed DataFrame
    return summed_scores

# def get_themes(theme_list_str, subtitles_path, save_path):
#     theme_list = theme_list_str.split(',')
#     theme_classifier = ThemeClassifier(theme_list)

#     output_df = theme_classifier.get_themes(subtitles_path, save_path)

#     # Remove "dialogue" from the theme list
#     theme_list = [theme for theme in theme_list if theme != 'dialogue']
    
#     # Filter out non-existing columns
#     theme_list = [theme for theme in theme_list if theme in output_df.columns]

#     output_df = output_df[theme_list]

#     # Sum the scores and reset index for the output DataFrame
#     output_df = output_df.sum().reset_index()

#     output_df.columns = ['theme', 'score']
#     output_df['theme'] = output_df['theme'].astype(str)  # Ensure 'theme' is a string
#     output_df['score'] = output_df['score'].astype(float) 

#     output_char = gr.BarPlot(
#           output_df,
#             x="Theme",
#             y="score",
#             title="Series Theme",
#             tooltip=["theme", "score"],
#             vertical=False,
#             width=500,
#             height=260
#       )
#     return output_df

def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot(
                                    value=None,  # Start with no data
                                    x="theme",
                                    y="score",
                                    title="Series Theme Scores",
                                    tooltip=["theme", "score"],
                                    vertical=False,
                                    width=500,
                                    height=260
                                            ) 
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(
                            get_themes,
                            inputs=[theme_list, subtitles_path, save_path],
                            outputs=plot
                        )
    iface.launch(share=True)

if __name__ == '__main__':
    main()



# import gradio as gr
# from theme_classifier import ThemeClassifier
# import os

# def get_themes(theme_list_str, subtitles_path, save_path):
#     if not theme_list_str:
#         return "Theme list cannot be empty."
#     if not subtitles_path or not os.path.exists(subtitles_path):
#         return "Subtitles path is invalid or does not exist."
#     print(theme_list_str, subtitles_path, save_path)
#     theme_list = theme_list_str.split(',')
#     theme_classifier = ThemeClassifier(theme_list)

#     try:
#         output_df = theme_classifier.get_themes(subtitles_path, save_path)
#     except Exception as e:
#         return f"Error during theme classification: {e}"
    
#     theme_list = [theme for theme in theme_list if theme != 'dialogue' and theme in output_df.columns]
#     if not theme_list:
#         return "No valid themes found in the output."
    
#     output_df = output_df[theme_list]
#     output_df = output_df.sum().reset_index()
#     output_df.columns = ['Theme', 'Score']
#     return output_df

# def get_themes(theme_list_str, subtitles_path, save_path):
#     if not theme_list_str:
#         return "Theme list cannot be empty."
#     if not subtitles_path or not os.path.exists(subtitles_path):
#         return "Subtitles path is invalid or does not exist."
    
#     theme_list = theme_list_str.split(',')
#     theme_classifier = ThemeClassifier(theme_list)

#     try:
#         output_df = theme_classifier.get_themes(subtitles_path, save_path)
#     except Exception as e:
#         return f"Error during theme classification: {e}"
    
#     theme_list = [theme for theme in theme_list if theme != 'dialogue' and theme in output_df.columns]
#     if not theme_list:
#         return "No valid themes found in the output."
    
#     # Select only the relevant themes and calculate their scores
#     output_df = output_df[theme_list]
#     output_df = output_df.sum().reset_index()
#     output_df.columns = ['Theme', 'Score']
    
#     # Return the data as a list of dictionaries for Gradio
#     return output_df.to_dict('records')


# def main():
#     with gr.Blocks() as iface:
#         with gr.Row():
#             with gr.Column():
#                 gr.HTML("<h1>Theme Classification (Zero Shot Classifiers)</h1>")
#                 with gr.Row():
#                     with gr.Column():
#                         plot = gr.BarPlot(
#                             x="Theme",
#                             y="Score",
#                             title="Series Theme",
#                             tooltip=["Theme", "Score"],
#                             vertical=False,
#                             width=500,
#                             height=260,
#                         )
#                     with gr.Column():
#                         theme_list = gr.Textbox(label="Themes", placeholder="Enter comma-separated themes, e.g., comedy,drama")
#                         subtitles_path = gr.Textbox(label="Subtitles Path", placeholder="Path to subtitles or script file")
#                         save_path = gr.Textbox(label="Save Path", placeholder="Path to save the output CSV")
#                         get_themes_button = gr.Button("Get Themes")
#                         get_themes_button.click(
#                             get_themes,
#                             inputs=[theme_list, subtitles_path, save_path],
#                             outputs=plot,
#                         )
#     iface.launch(share=True)

# if __name__ == '__main__':
#     main()

# import gradio as gr
# from theme_classifier import ThemeClassifier
# import os
# import pandas as pd

# # Function to classify themes and prepare data for the BarPlot
# def get_themes(theme_list_str, subtitles_path, save_path):
#     if not theme_list_str:
#         return "Theme list cannot be empty."
#     if not subtitles_path or not os.path.exists(subtitles_path):
#         return "Subtitles path is invalid or does not exist."
    
#     theme_list = theme_list_str.split(',')
#     theme_classifier = ThemeClassifier(theme_list)

#     try:
#         output_df = theme_classifier.get_themes(subtitles_path, save_path)
#     except Exception as e:
#         return f"Error during theme classification: {e}"
    
#     theme_list = [theme for theme in theme_list if theme != 'dialogue' and theme in output_df.columns]
#     if not theme_list:
#         return []
    
#     # Prepare DataFrame for plotting
#     output_df = output_df[theme_list]
#     output_df = output_df.sum().reset_index()
#     output_df.columns = ['Theme', 'Score']

#     # Convert DataFrame to list of dictionaries
#     return output_df

# # Main function to create the Gradio interface
# def main():
#     with gr.Blocks() as iface:
#         with gr.Row():
#             with gr.Column():
#                 gr.HTML("<h1>Theme Classification (Zero Shot Classifiers)</h1>")
#                 with gr.Row():
#                     with gr.Column():
#                         plot = gr.BarPlot()
#                     with gr.Column():
#                         theme_list = gr.Textbox(label="Themes", placeholder="Enter comma-separated themes, e.g., comedy,drama")
#                         subtitles_path = gr.Textbox(label="Subtitles Path", placeholder="Path to subtitles or script file")
#                         save_path = gr.Textbox(label="Save Path", placeholder="Path to save the output CSV")
#                         get_themes_button = gr.Button("Get Themes")
                        
#                         # Button action to trigger the theme classification
#                         get_themes_button.click(
#                             get_themes,
#                             inputs=[theme_list, subtitles_path, save_path],
#                             outputs=plot,
#                         )
#     iface.launch(share=True)

# # Run the Gradio interface
# if __name__ == '__main__':
#     main()

