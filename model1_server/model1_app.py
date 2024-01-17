import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
import pc_input_text_preprocessing as pt
import gradio as gr

model_save_path_ie = 'kc_bert_{}_classifier.pth'.format('ie')
model_save_path_ns = 'kc_bert_{}_classifier.pth'.format('ns')
model_save_path_tf = 'kc_bert_{}_classifier.pth'.format('tf')
model_save_path_pj = 'kc_bert_{}_classifier.pth'.format('pj')

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

def detech_text(op_chat):
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

def main(file_path):
  fin = ''
  mbti = [['i','e'],['n','s'],['t','f'],['p','j']]
  tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-large")
  label_score = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]
  cnt = 0

  my_chat, op_chat = pt.read_file(file_path)
  
  chats = detech_text(op_chat)

  for chat in chats:
    input_encodings = tokenizer(chat, padding=True, truncation=True, return_tensors="pt")
    score, pred = model_score('ie', input_encodings)
    label_score[0][0] += score[0]
    label_score[0][1] += score[1]
    score, pred = model_score('ns', input_encodings)
    label_score[1][0] += score[0]
    label_score[1][1] += score[1]
    score, pred = model_score('tf', input_encodings)
    label_score[2][0] += score[0]
    label_score[2][1] += score[1]
    score, pred = model_score('pj', input_encodings)
    label_score[3][0] += score[0]
    label_score[3][1] += score[1]
    cnt += 1

  for i in range(4):
    label_score[i][0] /= cnt/100
    label_score[i][1] /= cnt/100
    if label_score[i][0] >= label_score[i][1]:
      fin += mbti[i][0]
    else:
      fin += mbti[i][1]

  result = {
        'ie': {'i': round(label_score[0][0], 2), 'e': round(label_score[0][1], 2)},
        'ns': {'n': round(label_score[1][0], 2), 's': round(label_score[1][1], 2)},
        'tf': {'t': round(label_score[2][0], 2), 'f': round(label_score[2][1], 2)},
        'pj': {'p': round(label_score[3][0], 2), 'j': round(label_score[3][1], 2)},
        'mbti': fin
    }
  
  return result


# Gradio 인터페이스 생성
iface = gr.Interface(
    fn=main,
    inputs=gr.File(type="filepath", label="File Upload"),
    outputs="json"
)


if __name__ == '__main__':
    # Gradio 웹 애플리케이션 실행
  iface.launch(share=False,server_name="0.0.0.0", server_port=3000)  # share=True로 설정하면 외부에서도 접근 가능한 링크 생성
