import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datetime import datetime
import os

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}

def cur_date():
  return datetime.today().strftime("%Y%m%d")
                            
def check_label(label):
  if label['label'] == 'intp':
    return 0
  elif label['label'] == 'intj':
    return 1
  elif label['label'] == 'infp':
    return 2
  elif label['label'] == 'infj':
    return 3
  elif label['label'] == 'istp':
    return 4
  elif label['label'] == 'istj':
    return 5
  elif label['label'] == 'isfp':
    return 6
  elif label['label'] == 'isfj':
    return 7
  elif label['label'] == 'entp':
    return 8
  elif label['label'] == 'entj':
    return 9
  elif label['label'] == 'enfp':
    return 10
  elif label['label'] == 'enfj':
    return 11
  elif label['label'] == 'estp':
    return 12
  elif label['label'] == 'estj':
    return 13
  elif label['label'] == 'esfp':
    return 14
  elif label['label'] == 'esfj':
    return 15
  

def check_label_ie(label):
  if label['label'][0] == 'i' or label['label'][0] == 'I':
    return 0
  else:
    return 1

def check_label_ns(label):
  if label['label'][1] == 'n' or label['label'][1] == 'N':
    return 0
  else:
    return 1

def check_label_tf(label):
  if label['label'][2] == 't' or label['label'][2] == 'T':
    return 0
  else:
    return 1

def check_label_pj(label):
  if label['label'][3] == 'p' or label['label'][3] == 'P':
    return 0
  else:
    return 1

def create_grad(mbti):
  print(mbti)
  rawdata = pd.read_csv('last_datset.csv') # 데이터 셋에 따라 수정. 데이터셋은 tsv 파일에 document와 label이 tab으로 구분되어 있어야 함.mb
  rawdata['label'] = rawdata['mbti']

  texts = []
  labels = []

  for idx, label in rawdata.iterrows():
    texts.append(label['text'])
    if mbti == 'ie':
      labels.append(check_label_ie(label))
    elif mbti == 'ns':
      labels.append(check_label_ns(label))
    elif mbti == 'tf' :
      labels.append(check_label_tf(label))
    else:
      labels.append(check_label_pj(label))

    #labels.append(label['label'])

  from sklearn.preprocessing import LabelEncoder

  encoder = LabelEncoder()
  labels = encoder.fit_transform(labels)


  model_name = "beomi/KcBERT-Large"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) # 라벨링 종류 수에 따라 수정

  max_length = 128

  labels = torch.tensor(labels, dtype=torch.long)
  dataset = CustomDataset(texts, labels, tokenizer, max_length)

  batch_size = 64

  from sklearn.model_selection import train_test_split
  train, test = train_test_split(dataset, test_size=0.1, random_state=43)

  train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
  valid_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)



  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if device.type == 'cuda':
      print("Current device:", torch.cuda.get_device_name(device))
  else:
      print("Current device: CPU")
  model = model.to(device)

  learning_rate = 5e-6
  epochs = 5

  optimizer = AdamW(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss()


  best_loss = float('inf')
  early_stop_counter = 0
  early_stopping_epochs = 2

  # ie에 대해 레이블링 후 돌림
  for epoch in range(epochs):
      model.train()
      total_loss = 0

      for cnt, batch in enumerate(train_dataloader):
          if cnt % 100 == 0:
            print('cur : {}'.format(cnt))
          input_ids = batch['input_ids']
          attention_mask = batch['attention_mask']
          labels = batch['label']

          input_ids = input_ids.to(device)
          attention_mask = attention_mask.to(device)
          labels = labels.to(device)

          # 그래디언트 초기화
          optimizer.zero_grad()
          # 모델에 입력을 주어 예측을 생성합니다.
          outputs = model(input_ids, attention_mask=attention_mask)
          # 모델 출력에서 로짓(분류에 대한 점수)을 얻습니다.
          logits = outputs.logits
          # 손실을 계산합니다.
          loss = criterion(logits, labels)
          # 역전파를 통해 그래디언트 계산
          loss.backward()
          # 옵티마이저를 사용해 가중치를 업데이트
          optimizer.step()
          # 에포크 전체 손실을 누적합니다.
          total_loss += loss.item()

      # 에포크 평균 손실 계산
      avg_loss = total_loss / len(train_dataloader)

      # 에포크별 손실 출력
      print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

      # 모델 평가
      model.eval()
      val_total_loss = 0
      correct = 0
      total = 0

      with torch.no_grad():
          for val_batch in valid_dataloader:
              # Validation 데이터 가져오기
              val_input_ids = val_batch['input_ids']
              val_attention_mask = val_batch['attention_mask']
              val_labels = val_batch['label']

              val_input_ids = val_input_ids.to(device)
              val_attention_mask = val_attention_mask.to(device)
              val_labels = val_labels.to(device)

              # 모델 예측
              val_outputs = model(val_input_ids, attention_mask=val_attention_mask)
              val_logits = val_outputs.logits

              # 손실 계산
              val_loss = criterion(val_logits, val_labels)
              val_total_loss += val_loss.item()

              # 정확도 계산
              val_preds = val_logits.argmax(dim=1)
              correct += (val_preds == val_labels).sum().item()
              total += val_labels.size(0)

      val_avg_loss = val_total_loss / len(valid_dataloader)
      val_accuracy = correct / total

      if val_avg_loss > best_loss:
        early_stop_counter += 1
      else:
        best_loss = val_avg_loss
        early_stop_counter = 0

      # 조기 종료 조건 확인
      if early_stop_counter >= early_stopping_epochs:
        print("Early Stopping!")
        break

      print(f"Validation Loss: {val_avg_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")

  model_save_path = "kc_bert_{}_classifier_{}.pth".format(mbti, cur_date())
  torch.save(model.state_dict(), model_save_path)

def main():
  if not os.path.exists("kc_bert_{}_classifier_{}.pth".format('ie', cur_date())):
    create_grad('ie')
  if not os.path.exists("kc_bert_{}_classifier_{}.pth".format('ns', cur_date())):
    create_grad('ns')
  if not os.path.exists("kc_bert_{}_classifier_{}.pth".format('tf', cur_date())):
    create_grad('tf')
  if not os.path.exists("kc_bert_{}_classifier_{}.pth".format('ie', cur_date())):
    create_grad('pj')


main()