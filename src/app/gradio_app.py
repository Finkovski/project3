
import gradio as gr
from transformers import pipeline

clf = pipeline("sentiment-analysis", model="distilbert-base-uncased")

def classify(text):
    return clf(text)[0]['label']

demo = gr.Interface(fn=classify, inputs=gr.Textbox(label="Review"), outputs=gr.Label(label="Sentiment"))
if __name__ == "__main__":
    demo.launch()
