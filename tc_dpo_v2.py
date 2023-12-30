import torch
from tqdm.auto import tqdm
import pandas as pd
from transformers import pipeline, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, DPOTrainer
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
from utils import TopAccuracyTextsNoDuplicates, extract_text_after_colon, evaluation, evaluation_full,got_example,evaluation_loss
from dataset_utils import load_all_dataset
import sys
import datetime
import random
from dataset_utils import dataset_dicts

    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target_model_name',
        type= str,
        default= 'roberta-large'
    )
    parser.add_argument(
        '--agent_model_name',
        type= str,
        default= 'gpt2'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default= 'sst2'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default= 5
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default= 1.41e-6
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
    
    parser.add_argument(
        '--train_on_gpu',
        type=bool,
        default= True
    )
    parser.add_argument(
        '--language_feedback',
        type=bool,
        default= True
    )
    parser.add_argument(
        '--topk',
        type=int,
        default= 5
    )
    parser.add_argument(
        '--init_kl_coef',
        type=float,
        default= 0.4
    )

    parser.add_argument(
        '--verbalizer',
        type = str,
        nargs = '+',
        default = ['negative','positive']
    )
    parser.add_argument(
        '--dataset_size',
        type = int,
        default = -1
    )
    parser.add_argument(
        '--test_batch_size',
        type = int,
        default = 100
    )
    parser.add_argument(
        '--meta_question',
        type = str,
        default = "Please write instructions that will help others judge positive or negative by looking at the sentence. In one or two sentences.",
    )
    parser.add_argument(
        '--test_term',
        type = int,
        default = 20,
    )
    parser.add_argument(
        '--example',
        type = int,
        default = 0,
    )
    parser.add_argument(
        '--max_length',
        type = int,
        default = 100,
    )
    parser.add_argument(
        '--softmax_reward',
        type = bool,
        default = False,
    )
    parser.add_argument(
        '--balanced_set',
        type = bool,
        default = True,
    )
    args = parser.parse_args()
    return args




def main():
    training_args = TrainingArguments(
        #per_device_train_batch_size=script_args.per_device_train_batch_size,
        #max_steps=script_args.max_steps,
        remove_unused_columns=False,
        #gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        #learning_rate=script_args.learning_rate,
        
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10,  # match results in blog post
        eval_steps=500,
        output_dir="./test",
        optim="rmsprop",
        warmup_steps=150,
        report_to="none",
        bf16=True,
        #gradient_checkpointing=script_args.gradient_checkpointing,
        # TODO: uncomment that on the next transformers release
        # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
        #padding_value=0,
    )
    args = parse_args()
    wandb.init(project='tc_dpo_v2',name = args.agent_model_name + '_' + args.target_model_name + '_' + args.dataset_name)
    device = 'cuda:0'
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_name = args.agent_model_name
    if '/' in agent_name:
        agent_name = agent_name.replace('/','_')
    target_name = args.target_model_name
    if '/' in target_name:
        target_name = target_name.replace('/','_')
    #print(agent_name)
    
    filename = f"dpo_data/{agent_name}_{target_name}_{current_time}_{args.dataset_name}.txt"
    original_stdout = sys.stdout
    #dataset load
    dataset = load_all_dataset(args.dataset_name)
    train_dataset = dataset[0]
    if args.dataset_size != -1 and args.balanced_set == False:
        subset_indices = random.sample(range(len(train_dataset)), args.dataset_size)
        train_dataset = Subset(train_dataset, subset_indices)
    elif args.dataset_size != -1 and args.balanced_set == True:
        num_classes = len(dataset_dict['train']['label'].unique())
        subset_indices = []
        for i in range(num_classes):
            subset_indices += random.sample(dataset_dict['train'][dataset_dict['train']['label']==i].index.tolist(), args.dataset_size//num_classes)
        train_dataset = Subset(train_dataset, subset_indices)
    test_dataset = dataset[2]
    verbalizer = args.verbalizer
    print(train_dataset[0])
    
    with open(filename,'w') as f:
        sys.stdout = f
        #model_name = 'roberta-large'
        tokenizer = AutoTokenizer.from_pretrained(args.target_model_name,cache_dir = '../../mnt/minchan.kwon/')
        model = AutoModelForCausalLM.from_pretrained(
            args.target_model_name,  device_map = device, cache_dir = '../../mnt/minchan.kwon/'
        )
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        a = evaluation(
            ['Is this review positive?', 'Is this review positive? Please Answer Yes or No.'],
            test_dataset,
            model,
            tokenizer,
            device,
            verbalizer
            )
        print('agent : ',args.agent_model_name)
        print('target : ',args.target_model_name)
        print('dataset : ',args.dataset_name)
        print('test accuracy : ' ,a)
        print(args)
        
        #test

        bs = args.batch_size
        config = PPOConfig(
            model_name = args.agent_model_name,
            learning_rate = args.learning_rate,
            batch_size = bs,
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        
        train_model = AutoModelForCausalLM.from_pretrained(
            args.agent_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            #peft_config=lora_config,
            cache_dir = '../../mnt/minchan.kwon/')
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.agent_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            #peft_config=lora_config,
            cache_dir = '../../mnt/minchan.kwon/')
        ppo_tokenizer = AutoTokenizer.from_pretrained(args.agent_model_name,cache_dir='../../mnt/minchan.kwon')
        ppo_tokenizer.pad_token = ppo_tokenizer.eos_token
        #ppo_trainer = PPOTrainer(config, train_model, ref_model, tokenizer)

        generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_length":args.max_length,
        #"temperature": 0.9,
        "min_length": -1,
        }

        meta_text = [
            {"role": "system", "content": "You are ai helper, You should help solve the following classification problems."},
            {"role": "user", "content": args.meta_question},
            {"role": "AI:", "content" : ""}
            #{"role": "user", "content": "Who are you?"}
        ]

        dataset_dict = dataset_dicts(args.dataset_name)
        if args.dataset_size != -1:
            subset_indices = random.sample(range(len(train_dataset)), args.dataset_size)
            train_dataset = Subset(train_dataset, subset_indices)
        print('meta_text : ',meta_text)
        queue = TopAccuracyTextsNoDuplicates(max_size=5)
        real_queue = TopAccuracyTextsNoDuplicates(max_size=5)
        #meta_text = 'Human: Write a prompt that allows you to say yes if the sentence is positive and no if it is negative. AI:'
        
        
        for ep in tqdm(range(100)):
            #make query tensor
            """
            if args.example != 0:
                examples = got_example(train_dataset,dataset_dict,shot=args.example)
                example = ''
                for e in examples:
                    example += e + '\n'
                meta_text = [
                    {"role": "system", "content": "You are ai helper, You should help solve the following classification problems."},
                    {"role": "user", "content": args.meta_question},
                    {'role' : 'system', 'content' : 'Here is some example :' + example},
                    {"role": "AI:", "content" : ""}
                ]
            """
            query_tensors = ppo_tokenizer.apply_chat_template(meta_text, return_tensors='pt').to(device)
            print(query_tensors.shape)
            #generate prompt in token level
            response_tensors = [train_model.generate(query_tensors,**generation_kwargs) for i in range(bs)]
            
            #decode generate prompt and extract after 'AI:'
            output = [ppo_tokenizer.decode(r.squeeze(),skip_special_tokens = True) for r in response_tensors]
            used_prompt = [extract_text_after_colon(out) for out in output]
            used_prompt.append('Determine whether the following sentence is positive or negative.')
            used_prompt.append('Does the sentence express a positive or negative emotion?')
            #evalaute prompt
            if args.softmax_reward==False:
                rewards = evaluation(used_prompt,
                                    train_dataset,
                                    model,
                                    tokenizer,
                                    device,
                                    verbalizer,
                                    dataset_size = args.test_batch_size,
                                    )
            else:
                rewards = evaluation_loss(used_prompt,
                                    train_dataset,
                                    model,
                                    tokenizer,
                                    device,
                                    verbalizer,
                                    dataset_size = args.test_batch_size,
                                    )                
            rewards_tensor = [torch.Tensor([r]) for r in rewards]
            reward_np = np.array(rewards)
            wandb.log({'mean' : np.mean(reward_np), 'std' : np.std(reward_np), 'max' : np.max(reward_np)})
            
            
            
            text_score_pairs = sorted(zip(used_prompt, rewards), key=lambda x: x[1], reverse=True)

            # 모든 가능한 쌍을 생성하되, 첫 번째 텍스트의 점수가 더 높도록
            pairs = [(text1, text2) for i, (text1, score1) in enumerate(text_score_pairs)
                    for text2, score2 in text_score_pairs[i + 1:]]
            
            dpo_dataset = []
            prom = ''
            for m in meta_text:
                prom += m['content'] + ' '
            for text1, text2 in pairs:
                dic = {'prompt':prom,
                       'chosen':text1,
                       'rejected':text2,}
                dpo_dataset.append(dic)
            
            dpo_trainer = DPOTrainer(
                train_model,
                ref_model,
                args=  training_args,
                beta=0.1,
                train_dataset = dpo_dataset,
                max_length=100,
                tokenizer = ppo_tokenizer,
                
            )
            #print('\n !!!!: ',dpo_trainer.padding_value,dpo_trainer.tokenizer.pad_token_id)
            #add queue for logging
            for i in range(len(rewards_tensor)):
                #print('prompt : \n' , used_prompt[i], '\n reward : ',rewards[i])
                changed = queue.add(rewards[i],used_prompt[i])
                if changed == True:
                    li  = queue.get_top_texts()
                    print('queue updated\n')
                    print('epoch : ',str(ep),'\n')
                    num = 5
                    for l in li:
                        print('top ',num,'\n')
                        print('score : ',l[0], '\n text : ',l[2])
                        num -=1
                    print('\n\n')
                
            #step for ppo 
            dpo_trainer.train()
            
            #reset record to real score
            if ep % args.test_term == 0 and ep !=0:
                a = queue.get_top_texts()
                new_acc = evaluation([l[2] for l in a],
                                        test_dataset,
                                        model,
                                        tokenizer,
                                        device,
                                        verbalizer,
                                        dataset_size = len(test_dataset),
                                        )
                for i in range(len(a)):
                    real_queue.add(new_acc[i],a[i][2])
                    #print('score : ',new_acc[i],'\ntext : ',a[i][2])
                print('real queue updated!\n')
                li = real_queue.get_top_texts()
                for l in li:
                    print('score : ',l[0], 'text : ',l[2])
                li_new_np = np.array([l[0] for l in li])
                wandb.log({'real_mean' : np.mean(li_new_np), 'real_std' : np.std(li_new_np), 'real_max' : np.max(li_new_np)})
                print('\n\n')
        sys.stdout = original_stdout
if __name__ == '__main__':
    main()