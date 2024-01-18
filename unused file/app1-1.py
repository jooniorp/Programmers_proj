import gradio as gr
import os
from dotenv import load_dotenv
import openai
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
import pc_input_text_preprocessing as pt
from PIL import Image
import matplotlib.pyplot as plt

load_dotenv()

API=os.getenv('API')
openai.api_key = API

# Code for Task 1
def task1(text):
    value = int(text)
    probability = np.random.rand(value)
    probability /= probability.sum()

    # 원형 그래프 생성
    labels = [f"Class {i+1}" for i in range(len(probability))]
    plt.pie(probability, labels=labels, autopct='%1.1f%%', startangle=90)

    # 이미지 저장
    delete_file('data_set\save.png')
    image_path = "data_set\save.png"
    plt.savefig(image_path)
    plt.close()  # 그래프를 화면에 표시하지 않고 바로 이미지로 저장

    return image_path

def delete_file(filename):
    file_path = filename
    if os.path.exists(file_path):
        os.remove(file_path)
        print("delete")
    else:
        print("no")
        
   
# Code for Task 2
def task2(text, mbti):
    # 여기에 MBTI 스타일로 변환하는 로직을 구현합니다.
    # 예: "Please rewrite this in [MBTI] style: [text]"
    prompt = f"'{text}'라는 문장을 {mbti} 스타일로 한글로 재작성해주세요."
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()


# interface one
iface1 = gr.Interface(
    fn=task1,
    inputs=[gr.Files(type="filepath", label="File Upload"), gr.Dropdown(["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"], label="MBTI")],
    outputs=["image","image"],
    title="Multi-Page Interface"
)
# interface two
iface2 = gr.Interface(
    fn=task2,
    inputs=[gr.Textbox(), gr.Dropdown(["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"], label="MBTI")],
    outputs=gr.Textbox(),
    title="Multi-Page Interface"
)

demo = gr.TabbedInterface([iface1, iface2], ["Model1", "Model2"])

# Run the interface
demo.launch(share=True)