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
        default= 'imdb'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default= 5
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default= 1e-6
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
        '--use_fewshot',
        type = bool,
        default = False,
    )
    
    parser.add_argument(
        '--debug_mode',
        type = bool,
        default = False,
    )
    
    parser.add_argument(
        '--warmup_step',
        type= bool,
        default= False
    )
    parser.add_argument(
        '--balanced_set',
        type= bool,
        default= True
    )
        
    args = parser.parse_args()
    return args




def main():
    args = parse_args()
    wandb.init(project='tc_ppo_v4',name = args.agent_model_name + '_' + args.target_model_name + '_' + args.dataset_name)
    
    device = 'cuda:0'
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_name = args.agent_model_name
    if '/' in agent_name:
        agent_name = agent_name.replace('/','_')
    target_name = args.target_model_name
    if '/' in target_name:
        target_name = target_name.replace('/','_')
    #print(agent_name)
    
    filename = f"data/{agent_name}_{target_name}_{current_time}_{args.dataset_name}.txt"
    original_stdout = sys.stdout
    #dataset load
    dataset = load_all_dataset(args.dataset_name)
    train_dataset = dataset[0]
    test_dataset = dataset[2]
    verbalizer = args.verbalizer
    print(train_dataset[0])
    with open(filename,'w') as f:
        if args.debug_mode == False:
            sys.stdout = f
        
        
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
        
        
        train_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.agent_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            peft_config=lora_config,
            cache_dir = '../../mnt/minchan.kwon/')
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.agent_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            peft_config=lora_config,
            cache_dir = '../../mnt/minchan.kwon/')
        ppo_tokenizer = AutoTokenizer.from_pretrained(args.agent_model_name,cache_dir='../../mnt/minchan.kwon')
        ppo_tokenizer.pad_token = ppo_tokenizer.eos_token
        ppo_trainer = PPOTrainer(config, train_model, ref_model, tokenizer)

        generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_length":args.max_length,
        "temperature": 0.9,
        "min_length": -1,
        }

        meta_text = [
            {"role": "system", "content": "You are ai helper, You should help solve the following classification problems."},
            {"role": "user", "content": args.meta_question},
            {"role": "AI:", "content" : ""}
            #{"role": "user", "content": "Who are you?"}
        ]
        meta_text = [
        #{"role": "system", "content": "You are Hermes 2"},
        {"role": "user", "content": "Please write instructions that will help others judge positive or negative by looking at the sentence. In one or two sentences."},
        {"role": "AI:", "content" : ""}
        #{"role": "user", "content": "Who are you?"}
        ]
        if args.agent_model_name == 'gpt2':
            meta_text = 'please write down instruction to judge positive or negative by looking at the sentence. in one or two sentences. AI:'
        
        dataset_dict = dataset_dicts(args.dataset_name)
        if args.dataset_size != -1 and args.balanced_set == False:
            subset_indices = random.sample(range(len(train_dataset)), args.dataset_size)
            train_dataset = Subset(train_dataset, subset_indices)
        elif args.dataset_size != -1 and args.balanced_set == True:
            num_classes = len(dataset_dict['train']['label'].unique())
            subset_indices = []
            for i in range(num_classes):
                subset_indices += random.sample(dataset_dict['train'][dataset_dict['train']['label']==i].index.tolist(), args.dataset_size//num_classes)
            train_dataset = Subset(train_dataset, subset_indices)
        print('meta_text : ',meta_text)
        queue = TopAccuracyTextsNoDuplicates(max_size=5)
        real_queue = TopAccuracyTextsNoDuplicates(max_size=5)
        #meta_text = 'Human: Write a prompt that allows you to say yes if the sentence is positive and no if it is negative. AI:'
        
        
        for ep in tqdm(range(100)):
            if args.warmup_step and ep == 0:
                query_tensors = ppo_tokenizer.apply_chat_template(meta_text, return_tensors='pt').view(-1).to(device)
                used_prompts = ['Please judge whether the following sentence is positive or negative.',
                                    'Is this review positive?',
                                    'Is this review positive? Please Answer Positive or Negative.',
                                    'Please judge whether the following sentence is positive or negative. Please Answer Positive or Negative.',
                                    ]
                rewards = evaluation(used_prompts,
                                    train_dataset,
                                    model,
                                    tokenizer,
                                    device,
                                    verbalizer,
                                    dataset_size = args.test_batch_size,
                                    )
                response_tensors = ppo_tokenizer.batch_encode_plus(used_prompts,return_tensors='pt',padding=True).to(device)
                #print(response_tensors)
                rewards_tensor = [torch.Tensor([r]) for r in rewards]
                #print(query_tensors)
                stats = ppo_trainer.step([query_tensors.view(-1) for i in range(bs)],[response for response in response_tensors['input_ids']],rewards_tensor)
            
            
            
            else:
                #make query tensor
                if args.use_fewshot:
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
                query_tensors = ppo_tokenizer.apply_chat_template(meta_text, return_tensors='pt').view(-1).to(device)
                
                #generate prompt in token level
                response_tensors = ppo_trainer.generate(query_tensors.view(-1),**generation_kwargs, num_return_sequences=bs)
                
                #decode generate prompt and extract after 'AI:'
                output = [ppo_tokenizer.decode(r.squeeze(),skip_special_tokens = True) for r in response_tensors]
                used_prompt = [extract_text_after_colon(out) for out in output]
                
                #evalaute prompt
                if args.softmax_reward:
                    rewards = evaluation_loss(used_prompt,
                        train_dataset,
                        model,
                        tokenizer,
                        device,
                        verbalizer,
                        dataset_size = args.test_batch_size,
                        )  

                else:
                    #print('!')
                    rewards = evaluation(used_prompt,
                                        train_dataset,
                                        model,
                                        tokenizer,
                                        device,
                                        verbalizer,
                                        dataset_size = args.test_batch_size,
                                        )
                rewards_tensor = [torch.Tensor([r]) for r in rewards]
                
                
                
                if args.debug_mode:
                    print('query : ',meta_text)
                    #print('response : ',output)
                    print('reward : ',rewards)
                    print('used prompt : ',used_prompt)
                
                reward_np = np.array(rewards)
                wandb.log({'mean' : np.mean(reward_np), 'std' : np.std(reward_np), 'max' : np.max(reward_np)})
                print('mean : ',np.mean(reward_np))
                print('std : ',np.std(reward_np))
                print('max : ',np.max(reward_np))
                
                for i in range(len(rewards_tensor)):
                    changed = queue.add(rewards[i],used_prompt[i])
                    
                #step for ppo 
                stats = ppo_trainer.step([query_tensors.view(-1) for i in range(bs)],[response for response in response_tensors],rewards_tensor)
                
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
                    li_new = []
                    for l in li:
                        print('score : ',l[0], 'text : ',l[2])
                        li_new.append(l[0])
                    li_new_np = np.array(li_new)
                    wandb.log({'real_mean' : np.mean(li_new_np), 'real_std' : np.std(li_new_np), 'real_max' : np.max(li_new_np)})
                    print('\n\n')
        sys.stdout = original_stdout
if __name__ == '__main__':
    main()