import gradio as gr
from Naruto.theme_classifier.theme_classifier import ThemeClassifier
from Naruto.utils.data_loader import load_subtitles_dataset
from ..utils import load_subtitles_dataset
import streamlit as st
import pandas as pd


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




import gradio as gr  # Ensure this library is imported

def main():
    with gr.Blocks() as iface:
        # Theme Classification Section
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

        # Network Classification Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Classification (Ners and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        ners_path = gr.Textbox(label="Save Path")
                        get_network_graph_button = gr.Button("Get Network Graph")
                        # Uncomment and update the logic for the button click when needed
                        # get_network_graph_button.click(
                        #     get_network_graph,
                        #     inputs=[subtitles_path, ners_path],
                        #     outputs=network_html
                        # )
                        
    iface.launch(share=True)

if __name__ == '__main__':
    main()





