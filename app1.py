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
import re

load_dotenv()

API=os.getenv('API')
openai.api_key = API

# Code for Task 1
'''
model_save_path_ie = 'kc_bert_{}_classifier.pth'.format('ie')
model_save_path_ns = 'kc_bert_{}_classifier.pth'.format('ns')
model_save_path_tf = 'kc_bert_{}_classifier.pth'.format('tf')
model_save_path_pj = 'kc_bert_{}_classifier.pth'.format('pj')
'''

model_save_path_ie = 'model1_server\kc_bert_ie_classifier_20230115.pth'
model_save_path_ns = 'model1_server\kc_bert_ns_classifier_20230115.pth'
model_save_path_tf = 'model1_server\kc_bert_tf_classifier_20230115.pth'
model_save_path_pj = 'model1_server\kc_bert_pj_classifier.pth'

# 모델 아키텍처 생성
loaded_model_ie = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-large", num_labels=2)
loaded_model_ns = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-large", num_labels=2)
loaded_model_tf = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-large", num_labels=2)
loaded_model_pj = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-large", num_labels=2)

# 저장된 가중치 불러오기
loaded_model_ie.load_state_dict(torch.load(model_save_path_ie, map_location=torch.device('cpu')),strict=False)
loaded_model_ns.load_state_dict(torch.load(model_save_path_ns, map_location=torch.device('cpu')),strict=False)
loaded_model_tf.load_state_dict(torch.load(model_save_path_tf, map_location=torch.device('cpu')),strict=False)
loaded_model_pj.load_state_dict(torch.load(model_save_path_pj, map_location=torch.device('cpu')),strict=False)


# 모델을 평가 모드로 설정
loaded_model_ie.eval()  
loaded_model_ns.eval()  
loaded_model_tf.eval()  
loaded_model_pj.eval()  

def valid_label(mode, label):
    if mode == 'ie':
        if label == 0:
            return 'i'
        elif label == 1:
            return 'e'
    if mode == 'ns':
        if label == 0:
            return 'n'
        elif label == 1:
            return 's'
    if mode == 'tf':
        if label == 0:
            return 't'
        elif label == 1:
            return 'f'
    if mode == 'pj':
        if label == 0:
            return 'p'
        elif label == 1:
            return 'j'
    
def model_score(mode, input_encodings):
    if mode == 'ie':
    # 모델에 입력 데이터 전달
        with torch.no_grad():
            output = loaded_model_ie(**input_encodings)
    elif mode == 'ns':
    # 모델에 입력 데이터 전달
        with torch.no_grad():
            output = loaded_model_ns(**input_encodings)
    elif mode == 'tf':
    # 모델에 입력 데이터 전달
        with torch.no_grad():
            output = loaded_model_tf(**input_encodings)
    elif mode == 'pj':
    # 모델에 입력 데이터 전달
        with torch.no_grad():
            output = loaded_model_pj(**input_encodings)
    logits = output.logits
    softmax_layer = nn.Softmax(-1)
    softmax_result = softmax_layer(logits)
    predicted_labels = logits.argmax(dim=1)

    return softmax_result[0].tolist(), predicted_labels[0]

def my_detech_text(my_chat):
    chats = ['']
    chat_cnt = 0
    chat_loc = 0
    for i in range(len(my_chat['user_chat'])):
        if chat_cnt <= 30:
            chats[chat_loc] += my_chat['user_chat'][i]
            chat_cnt += len(my_chat['user_chat'][i])
        else:
            chats.append(my_chat['user_chat'][i])
            chat_cnt = len(my_chat['user_chat'][i])

    return chats

def op_detech_text(op_chat):
    chats = ['']
    chat_cnt = 0
    chat_loc = 0
    for i in range(len(op_chat['opponent_chat'])):
        if chat_cnt <= 30:
            chats[chat_loc] += op_chat['opponent_chat'][i]
            chat_cnt += len(op_chat['opponent_chat'][i])
        else:
            chats.append(op_chat['opponent_chat'][i])
            chat_cnt = len(op_chat['opponent_chat'][i])

    return chats

def draw_circle(data):

    
    ratio_ie=[data[0][0],data[0][1]]
    ratio_ns=[data[1][0],data[1][1]]
    ratio_tf=[data[2][0],data[2][1]]
    ratio_pj=[data[3][0],data[3][1]]
       
    label_ie=['i','e']
    label_ns=['n','s']
    label_tf=['t','f']
    label_pj=['p','j']
    
    plt.rc('font', size=30)
    plt.pie(ratio_ie, labels=label_ie)
    img_path1 = 'data_set\save_ie.png'
    delete_file(img_path1)
    plt.savefig(img_path1)
    plt.close()
    plt.pie(ratio_ns, labels=label_ns)
    img_path2 = 'data_set\save_ns.png'
    delete_file(img_path2)
    plt.savefig(img_path2)
    plt.close()
    plt.pie(ratio_tf, labels=label_tf)
    img_path3 = 'data_set\save_tf.png'
    delete_file(img_path3)
    plt.savefig(img_path3)
    plt.close()
    plt.pie(ratio_pj, labels=label_pj)
    img_path4 = 'data_set\save_pj.png'
    delete_file(img_path4)
    plt.savefig(img_path4)
    plt.close()
   
    image1 = Image.open(img_path1)
    image2 = Image.open(img_path2)
    image3 = Image.open(img_path3)
    image4 = Image.open(img_path4)
    
    image1_size = image1.size
    new_image = Image.new('RGB',(4*image1_size[0], image1_size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    new_image.paste(image3,(2*image1_size[0],0))
    new_image.paste(image4,(3*image1_size[0],0))
    img_path = 'data_set\save.png'
    delete_file(img_path)
    new_image.save(img_path)

    
    # 이미지 저장
    #delete_file('data_set\save.png')
    #image_path = "data_set\save.png"
    #plt.savefig(image_path)
    #plt.close()  # 그래프를 화면에 표시하지 않고 바로 이미지로 저장

    return img_path

def delete_file(filename):
    file_path = filename
    if os.path.exists(file_path):
        os.remove(file_path)
        print("delete")
    else:
        print("no")
        
        

def task1(file_path,mbti):
    my_fin = ''
    op_fin=''
    mbti = [['i','e'],['n','s'],['t','f'],['p','j']]
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-large")
    my_label_score = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]
    op_label_score = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]
    my_cnt = 0
    op_cnt=0
    
    my_chat, op_chat = pt.read_file(file_path)
  
    #my_chat, 내 채팅으로 mbti출력
    chats = my_detech_text(my_chat)
    
    for chat in chats:
        input_encodings = tokenizer(chat, padding=True, truncation=True, return_tensors="pt")
        score, pred = model_score('ie', input_encodings)
        my_label_score[0][0] += score[0]
        my_label_score[0][1] += score[1]
        score, pred = model_score('ns', input_encodings)
        my_label_score[1][0] += score[0]
        my_label_score[1][1] += score[1]
        score, pred = model_score('tf', input_encodings)
        my_label_score[2][0] += score[0]
        my_label_score[2][1] += score[1]
        score, pred = model_score('pj', input_encodings)
        my_label_score[3][0] += score[0]
        my_label_score[3][1] += score[1]
        my_cnt += 1

    for i in range(4):
        my_label_score[i][0] /= my_cnt/100
        my_label_score[i][1] /= my_cnt/100
        if my_label_score[i][0] >= my_label_score[i][1]:
            my_fin += mbti[i][0]
        else:
            my_fin += mbti[i][1]
    
    #op_chat, 상대방 채팅으로 mbti출력        
    chats = op_detech_text(op_chat)

    for chat in chats:
        input_encodings = tokenizer(chat, padding=True, truncation=True, return_tensors="pt")
        score, pred = model_score('ie', input_encodings)
        op_label_score[0][0] += score[0]
        op_label_score[0][1] += score[1]
        score, pred = model_score('ns', input_encodings)
        op_label_score[1][0] += score[0]
        op_label_score[1][1] += score[1]
        score, pred = model_score('tf', input_encodings)
        op_label_score[2][0] += score[0]
        op_label_score[2][1] += score[1]
        score, pred = model_score('pj', input_encodings)
        op_label_score[3][0] += score[0]
        op_label_score[3][1] += score[1]
        op_cnt += 1

    for i in range(4):
        op_label_score[i][0] /= op_cnt/100
        op_label_score[i][1] /= op_cnt/100
        if op_label_score[i][0] >= op_label_score[i][1]:
            op_fin += mbti[i][0]
        else:
            op_fin += mbti[i][1]

    result = {
        'ie': {'i': round(op_label_score[0][0], 2), 'e': round(op_label_score[0][1], 2)},
        'ns': {'n': round(op_label_score[1][0], 2), 's': round(op_label_score[1][1], 2)},
        'tf': {'t': round(op_label_score[2][0], 2), 'f': round(op_label_score[2][1], 2)},
        'pj': {'p': round(op_label_score[3][0], 2), 'j': round(op_label_score[3][1], 2)},
        'mbti': op_fin
    }
    
    my_image_path = 'model1_server\img_mbti\img_{}.png'.format(my_fin)
    op_image_path = 'model1_server\img_mbti\img_{}.png'.format(op_fin)
    
    return my_image_path,op_image_path,draw_circle(my_label_score) 

    
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
'''
iface1 = gr.Interface(
    fn=task1,
    inputs=[gr.File(type="filepath", label="File Upload"), gr.Dropdown(["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"], label="MBTI")],
    outputs=[gr.Image(label="나의 mbti"),gr.Image(label="상대방의 mbti")],
    title="Multi-Page Interface"
)
'''


with gr.Blocks() as iface1:
    gr.Markdown("당신의 카톡속 mbti는!?")
    with gr.Row():
        inp1 = gr.File(type="filepath", label="File Upload")
        inp2 = gr.Dropdown(["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"], label="MBTI")
    
    with gr.Row():
        btn1=gr.ClearButton()
        btn2=gr.Button("실행")
    
    with gr.Row():
        out1 = gr.Image(label="나의 mbti")
        out2 = gr.Image(label="상대방의 mbti")
        
    with gr.Row():   
        out3=gr.Image(label="나의 mbti비율")    

    # 버튼에 이벤트 리스너를 추가한다. 
    # 버튼 클릭시 update함수를 호출하고, inp에 입력된 문자열을 파라미터로 보낸다. 함수의 반환값은 out에 출력한다.
    btn2.click(fn=task1, inputs=[inp1,inp2], outputs=[out1,out2, out3])
    
# interface two
iface2 = gr.Interface(
    fn=task2,
    inputs=[gr.Textbox(), gr.Dropdown(["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"], label="MBTI")],
    outputs=gr.Textbox(),
    title="mbti말투별로 바꿔드립니다"
)

demo = gr.TabbedInterface([iface1, iface2], ["Model1", "Model2"])

# Run the interface
demo.launch(share=True)