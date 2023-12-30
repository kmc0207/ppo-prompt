import torch
from tqdm.auto import tqdm
import pandas as pd
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import FuyuProcessor, FuyuForCausalLM
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm.auto import tqdm
import argparse
from PIL import Image
import os
from peft import LoraConfig
import warnings
import numpy as np
import wandb
import copy
from collections import deque
from transformers import ViltProcessor, ViltForQuestionAnswering
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import random
import torch
import heapq

class TopAccuracyTextsNoDuplicates:
    def __init__(self,max_size= 5):
        self.heap = []
        self.text_map = {}  # 텍스트를 키로, 힙 내 위치를 값으로 하는 딕셔너리
        self.max_size = max_size

    def add(self, accuracy, text):
        if text in self.text_map:
            # 이미 존재하는 텍스트의 정확도 업데이트 (더 높은 정확도로)
            heap_index = self.text_map[text]
            if accuracy > self.heap[heap_index][0]:
                self.heap[heap_index] = (accuracy, len(text), text)
                heapq.heapify(self.heap)  # 힙 속성 유지를 위해 재구성
        else:
            # 새로운 텍스트 추가
            if len(self.heap) < self.max_size:
                heapq.heappush(self.heap, (accuracy, len(text), text))
                self.text_map[text] = len(self.heap) - 1
            elif accuracy > self.heap[0][0]:
                # 현재 힙의 최소 정확도보다 높은 경우에만 추가
                removed_text = heapq.heappop(self.heap)[2]
                if removed_text in self.text_map:
                    self.text_map.pop(removed_text)  # 제거된 텍스트를 딕셔너리에서 삭제
                heapq.heappush(self.heap, (accuracy, len(text), text))
                self.text_map[text] = len(self.heap) - 1
                #print('!',list(queue.get_top_texts()), '\n')
                #print("Something Changed!!!!\n")
                #print(sorted([item for item in self.heap], reverse=True))
                return True
        return False

    def get_top_texts(self):
        return sorted([item for item in self.heap], reverse=True)
    


#전체 테스트 셋에 대해서 테스트
def evaluation_full(prompts,imdb,model,tokenizer,device,verbalizer = ['Yes','No']):
    accs=  []
    for prompt in prompts:
        model.eval()
        subset_indices = random.sample(range(len(imdb["test"])), 100)

        # 서브셋 생성
        imdb_subset = Subset(imdb["test"], subset_indices)

        # DataLoader 설정 (서브셋 사용)
        dl = DataLoader(imdb["test"], batch_size=1, shuffle=True)


        tp = 0  # True Positive
        tn = 0  # True Negative
        fp = 0  # False Positive
        fn = 0  # False Negative
        # 배치 처리
        correct = 0
        total = 0

        yes_token_id = tokenizer.encode(verbalizer[0], add_special_tokens=False)[0]
        no_token_id = tokenizer.encode(verbalizer[1], add_special_tokens=False)[0]

        yes_answer_num = 0
        no_answer_num = 0
        yes_predictioon_num = 0
        no_prediction_num = 0

        for batch in tqdm(dl):
            # 텍스트 인코딩
            input_ids = tokenizer( batch['text'][0] + '\n' + prompt, return_tensors='pt',truncation=True).input_ids.to(device)
            
            # 모델 실행
            with torch.no_grad():
                outputs = model(input_ids)
            logits = outputs.logits

            # 'Yes'와 'No'의 첫 번째 토큰에 대한 로짓 비교
            yes_logits = logits[0, -1, yes_token_id]
            no_logits = logits[0, -1, no_token_id]

            prediction = 'Yes' if yes_logits > no_logits else 'No'
            correct_label = 'Yes' if batch['label'][0] == 1 else 'No'
            if correct_label == 'Yes':
                yes_answer_num += 1
            else:
                no_answer_num += 1
            if prediction == 'Yes':
                yes_predictioon_num += 1
            else:
                no_prediction_num += 1
            # 정답 레이블과 비교
            if prediction == 'Yes' and correct_label == 'Yes':
                tp += 1
            elif prediction == 'No' and correct_label == 'No':
                tn += 1
            elif prediction == 'Yes' and correct_label == 'No':
                fp += 1
            elif prediction == 'No' and correct_label == 'Yes':
                fn += 1

        # 성능 지표 계산
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = sensitivity  # 재현율은 민감도와 동일
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        accs.append(accuracy)
    return accs



def evaluation(prompts,
               dataset,
               model,
               tokenizer,
               device,
               verbalizer=['Yes', 'No', 'Maybe'],
               dataset_size=100,
               debug=False
               ):
    # 레이블 개수 확인
    accs = []
    for prompt in prompts:
        model.eval()
        subset_indices = random.sample(range(len(dataset)), dataset_size)

        # 서브셋 생성
        imdb_subset = Subset(dataset, subset_indices)
        #print(len(imdb_subset))
        if debug:
            print(len(imdb_subset))
        # DataLoader 설정 (서브셋 사용)
        dl = DataLoader(imdb_subset, batch_size=1, shuffle=True)

        # 레이블별 토큰 ID 매핑
        label_token_ids = {label: tokenizer.encode(label, add_special_tokens=False)[0] for label in verbalizer}

        # 배치 처리 및 평가
        correct = 0
        total = 0
        if debug:
            
            for batch in tqdm(dl):
                if 'text' in  batch.keys() :
                    
                # 텍스트 인코딩
                    input_ids = tokenizer(batch['text'][0] + '\n' + prompt, return_tensors='pt',truncation=True).input_ids.to(device)
                else : 
                    input_ids = tokenizer(batch['sentence'][0] + '\n' + prompt, return_tensors='pt',truncation=True).input_ids.to(device)
                
                # 모델 실행
                with torch.no_grad():
                    outputs = model(input_ids)
                logits = outputs.logits

                # 레이블별 로짓 계산 및 예측
                label_logits = {label: logits[0, -1, token_id] for label, token_id in label_token_ids.items()}
                prediction = max(label_logits, key=label_logits.get)
                #print(batch['label'])
                # 정답 레이블과 비교
                correct_label = verbalizer[batch['label'][0]]
                if prediction == correct_label:
                    correct += 1
                total += 1
        else:
            for batch in dl:
                if 'text' in  batch.keys() :
                    
                # 텍스트 인코딩
                    input_ids = tokenizer(batch['text'][0] + '\n' + prompt, return_tensors='pt',truncation=True).input_ids.to(device)
                else : 
                    input_ids = tokenizer(batch['sentence'][0] + '\n' + prompt, return_tensors='pt',truncation=True).input_ids.to(device)
                
                # 모델 실행
                with torch.no_grad():
                    outputs = model(input_ids)
                logits = outputs.logits

                # 레이블별 로짓 계산 및 예측
                label_logits = {label: logits[0, -1, token_id] for label, token_id in label_token_ids.items()}
                prediction = max(label_logits, key=label_logits.get)
                #print(batch['label'])
                # 정답 레이블과 비교
                correct_label = verbalizer[batch['label'][0]]
                if prediction == correct_label:
                    correct += 1
                total += 1            

        # 정확도 계산
        accuracy = correct / total if total != 0 else 0
        accs.append(accuracy)

    return accs


def evaluation_loss(prompts,
               dataset,
               model,
               tokenizer,
               device,
               verbalizer=['Yes', 'No', 'Maybe'],
               dataset_size=100):
    # 크로스 엔트로피 손실 함수 초기화
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    for prompt in prompts:
        model.eval()
        subset_indices = random.sample(range(len(dataset)), dataset_size)

        # 서브셋 생성
        imdb_subset = Subset(dataset, subset_indices)

        # DataLoader 설정 (서브셋 사용)
        dl = DataLoader(imdb_subset, batch_size=1, shuffle=True)

        # 레이블별 토큰 ID 매핑
        label_token_ids = {label: tokenizer.encode(label, add_special_tokens=False)[0] for label in verbalizer}

        # 배치 처리 및 평가
        total_loss = 0

        for batch in dl:
            if 'text' in batch.keys():
                input_ids = tokenizer(batch['text'][0] + '\n' + prompt, return_tensors='pt', truncation=True).input_ids.to(device)
            else:
                input_ids = tokenizer(batch['sentence'][0] + '\n' + prompt, return_tensors='pt', truncation=True).input_ids.to(device)

            # 실제 레이블
            true_label = batch['label'][0]

            # 모델 실행
            with torch.no_grad():
                outputs = model(input_ids)
            logits = outputs.logits

            # 크로스 엔트로피 손실 계산
            label_logits = torch.stack([logits[0, -1, token_id] for _, token_id in label_token_ids.items()])
            loss = 1-criterion(label_logits.unsqueeze(0), torch.tensor([true_label]).to(device))
            total_loss += loss.item()

        # 평균 손실 계산
        average_loss = total_loss / len(dl) if len(dl) != 0 else 0
        losses.append(average_loss)

    return losses

#이전 대화 항목을 제거하고 프롬프트로 사용
def extract_text_after_colon(text):
    # ':' 문자의 위치를 찾습니다.
    colon_index = text.find('AI:')

    # ':' 문자가 없으면, 원본 텍스트를 반환합니다.
    if colon_index == -1:
        return text

    # ':' 다음의 문자부터 문자열 끝까지 반환합니다.
    return text[colon_index + 4:]


import random
from dataset_utils import dataset_dicts
def got_example(dataset,dataset_dict,shot=5):
    examples =[]
    for i in range(shot):
        idx = random.randint(0,len(dataset))
        example = dataset[idx]
        if 'text' in example.keys():
            a = 'text : ' +example['text']+ 'label : '+ dataset_dict[example['label']]
            examples.append(a)
        else:
            a= 'sentence : ' +example['sentence']+ 'label : '+ dataset_dict[example['label']]
            examples.append(a)
    return examples