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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type= str,
        default= 'gpt2'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default= 5
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default= 1.41e-5
    )
    
    parser.add_argument(
        '--using_lora',
        type=bool,
        default= True
    )
    
    parser.add_argument(
        '--max_epochs',
        type=int,
        default= 1000
    )
    parser.add_argument(
        '--start_prompt',
        type=str,
        default= 'To determine whether she is Harin or not, note the following instruction : '
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default= 'ppo-prompt'
    )
    args = parser.parse_args()
    return args


def resize_image_keep_aspect(image_path, new_height):
    # 이미지를 불러옵니다
    with Image.open(image_path) as img:
        # 원본 이미지의 가로, 세로 크기를 가져옵니다
        original_width, original_height = img.size

        # 새로운 가로 크기를 계산합니다 (비율 유지)
        new_width = int(original_width * (new_height / original_height))

        # 이미지를 새로운 크기로 리사이즈합니다
        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

        return resized_img

def get_haerin_bench(version=2):
    images = []
    labels = []
    texts = []
    version = 'haerin_bench_v' + str(version)
    for img_path in os.listdir(version+'/haerin/'):
        if '.jpg' in img_path:
            img_path = version+'/haerin/'+img_path
            img = resize_image_keep_aspect(img_path,256)
            images.append(img)
            labels.append('Yes')
            texts.append('Is she haerin?')
    for img_path in os.listdir(version+'/non_haerin/'):
        if '.jpg' in img_path or '.jpeg' in img_path:
            img_path = version+'/non_haerin/'+img_path
            img = resize_image_keep_aspect(img_path,256)
            images.append(img)
            labels.append("No")
            texts.append('Is she haerin?')
    return images, labels, texts

def evaluation(prompt,images,labels,processor,eval_model):
    total_size = 0
    correct = 0
    prompt = prompt + 'Question : Is this girl haerin? please say yes or no. Answer :'
    with torch.no_grad():
        for i in range(len(images)):
            img = images[i]
            inputs = processor(text=prompt,images=img,return_tensors='pt').to('cuda:1',torch.float16)
            outputs = eval_model.generate(**inputs)
            #print(outputs)
            decoded_outputs = processor.batch_decode(outputs,skip_special_tokens=True)[0].strip()
            #print(decoded_outputs)
            answer = decoded_outputs.lower()
            label = labels[i].lower()
            if answer == label:
                correct+=1
            total_size +=1
    acc = correct/ total_size
    return acc
            

def batch_evaluation(prompt_batch,images,labels,processor,eval_model):
    accs = []
    for prompt in prompt_batch:
        acc = evaluation(prompt,images,labels,processor,eval_model)
        accs.append(torch.tensor(acc))
    return accs


def main():
    args = parse_args()
    warnings.filterwarnings(action='ignore')
    
    
    #데이터셋 불러오기
    images,labels,texts = get_haerin_bench()
    
    
    
    
    
    
    
    #PPO 모델 만들기
    config = PPOConfig(
        model_name = args.model_name,
        learning_rate = args.learning_rate,
        batch_size = args.batch_size
    )
    if args.using_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        train_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name,device_map ='cuda:0',torch_dtype=torch.bfloat16,peft_config=lora_config)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name,device_map='cuda:0',torch_dtype=torch.bfloat16,peft_config=lora_config)
    else:
        train_model = AutoModelForCausalLM.from_pretrained(config.model_name,device_map ='cuda:0',torch_dtype=torch.bfloat16)
        ref_model = AutoModelForCausalLM.from_pretrained(config.model_name,device_map='cuda:0',torch_dtype=torch.bfloat16)
    train_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_tokenizer.pad_token = train_tokenizer.eos_token
    
    ppo_trainer = PPOTrainer(config,train_model,ref_model,train_tokenizer)
    
    generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": train_tokenizer.eos_token_id,
    }
    
    
    
    
    #LVLM 모델 평가 만들기
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    eval_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",torch_dtype=torch.float16,device_map='cuda:1')
    eval_tokenizer = AutoTokenizer.from_pretrained('Salesforce/blip2-opt-2.7b')
    eval_tokenizer.pad_token = eval_tokenizer.eos_token
    
    
    
    #테스트
    test_acc = evaluation('Is she haerin?',images,labels,processor,eval_model)
    print('test_acc : ',test_acc)
    
    
    
    #메인 학습 부분
    wandb.login()
    run = wandb.init(project='ppo-prompt',name= args.run_name,config=args)
    for ep in tqdm(range(args.max_epochs)):
            text = args.start_prompt
            query_tensors = train_tokenizer.encode(text,return_tensors='pt').view(-1).to('cuda:0')
            response_tensors = ppo_trainer.generate(query_tensors,**generation_kwargs,num_return_sequences=5)
            output = [train_tokenizer.decode(r.squeeze()) for r in response_tensors]
            rewards = batch_evaluation(output,images,labels,processor,eval_model)
            reward_np = [reward.item() for reward in rewards]
            print(reward_np)
            print(output[np.argmax(np.array(reward_np))])
            wandb.log({'max_reward':np.max(np.array(reward_np)), 'max_reward_output':output[np.argmax(np.array(reward_np))],'mean_reward':np.mean(np.array(reward_np))})
            stats = ppo_trainer.step([query_tensors for i in range(5)],[response for response in response_tensors],rewards)
    
    
    
if __name__ == '__main__':
    main()