import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import pymongo
from pymongo import MongoClient
import schedule
import time
from datetime import datetime, timedelta
import db_to_csv as dtoc
import read_last_dataset_db_to_csv as dtocog

update_cycle = 30 #업데이트 주기

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
        attention_mask = encoding['attention된 파라미터가 저장될 경로]
  model_save_path = "kc_bert_{}_classifier_{}.pth".format(mbti,date_string)

  # 모델 아키텍처 생성
  loaded_model = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-large", num_labels=2)
  tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-large")#이것도 이전에 학습한 모델 토크나이저로 가져와야됨

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
  loaded_model = loaded_model.to(device)

  learning_rate = 5e-6
  epochs = 5

  optimizer = AdamW(loaded_model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss()

  best_loss = float('inf')
  early_stop_counter = 0
  early_stopping_epochs = 2

  # ie에 대해 레이블링 후 돌림
  for epoch in range(epochs):
      loaded_model.train()
      total_loss = 0

      for cnt, batch in enumerate(train_dataloader):
          print(cnt)
          input_ids = batch['input_ids']
          attention_mask = batch['attention_mask']
          labels = batch['label']

          input_ids = input_ids.to(device)
          attention_mask = attention_mask.to(device)
          labels = labels.to(device)

          # 그래디언트 초기화
          optimizer.zero_grad()
          # 모델에 입력을 주어 예측을 생성합니다.
          outputs = loaded_model(input_ids, attention_mask=attention_mask)
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
      loaded_model.eval()
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
              val_outputs = loaded_model(val_input_ids, attention_mask=val_attention_mask)
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

  torch.save(loaded_model.state_dict(),model_save_path)

def main():

  #가중치 업데이트
  train_and_update_model('ie')
  train_and_update_model('ns')
  train_and_update_model('tf')
  train_and_update_model('pj')

  # 주기적으로 모델 업데이트를 스케줄링
  schedule.every(update_cycle).days.do(train_and_update_model, mbti='ie', df=df)
  schedule.every(update_cycle).days.do(train_and_update_model, mbti='ns', df=df)
  schedule.every(update_cycle).days.do(train_and_update_model, mbti='tf', df=df)
  schedule.every(update_cycle).days.do(train_and_update_model, mbti='pj', df=df)
  
  while True:
    schedule.run_pending()
    time.sleep(1)

if __name__ == '__main__':
    main()


main()
