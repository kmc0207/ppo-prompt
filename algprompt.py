import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig,AutoModelForCausalLMWithValueHead
import argparse
import numpy as np
import wandb
import copy
import random
import heapq
import utils
from dataset_utils import load_all_dataset,dataset_dicts
from peft import LoraConfig
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',type=str,default='microsoft/phi-2')
    parser.add_argument('--agent_model',type=str,default='microsoft/phi-2')
    parser.add_argument('--task',type=str,default='classification')
    parser.add_argument('--dataset',type=str,default='sst2')
    parser.add_argument(
        '--verbalizer',
        type = str,
        nargs = '+',
        default = None
    )
    parser.add_argument('--cache_dir',type=str,default='./')
    parser.add_argument('--batch_size',type=int,default=2)
    parser.add_argument('--max_prompt_length',type=int,default=100)
    parser.add_argument('--train_data_per_labels',type=int,default=10)
    parser.add_argument('--num_example',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--meta_prompt',type=str,
                        default = '''I want to give the appropriate instruction to help
                        a friend who needs to look at the input and guess the output.
                        Plase write instruction to help my friends. Here are the input-output pairs:
                        ''',)
    parser.add_argument('--prompt_per_example',type=int,default=2)

    args = parser.parse_args()
    return args

def main():
    
    args = parser_args()
    device= 'cuda:0'
    wandb.init(project='ALGprompt', 
               config=args,
               name = args.task + '_' + args.dataset + '_' + args.agent_model + '_' + args.target_model)
    
    
    if args.verbalizer is None:
        verbalizer = dataset_dicts(args.dataset)
    num_labels = len(verbalizer)
    print('Verbalizer : ', verbalizer)
    
    #load dataset
    if args.task == 'classification':
        dataset = load_all_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        train_dataset,validation_dataset = utils.create_balanced_subset_and_validation(train_dataset,
                                                                                       args.train_data_per_labels * num_labels,
                                                                                       )
    else:
        #TODO
        pass
        
    #make dataloader
    test_dataloader = DataLoader(test_dataset,batch_size = 4,shuffle = True)
    train_dataloader = DataLoader(train_dataset,batch_size = 4,shuffle = True)
    
    
        #load agent model
    config = PPOConfig(
        model_name = args.agent_model,
        learning_rate = 1e-4,
        batch_size = args.batch_size,
        mini_batch_size= args.batch_size,
        log_with='wandb',
    )
    lora_config = LoraConfig(
        r= 16,
        lora_alpha = 32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    agent_tokenizer = AutoTokenizer.from_pretrained(args.agent_model,cache_dir = args.cache_dir)
    agent_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map = 'auto',
        peft_config = lora_config,
        cache_dir = args.cache_dir
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map = 'auto',
        peft_config = lora_config,
        cache_dir = args.cache_dir
    )
    agent_tokenizer.pad_token = agent_tokenizer.eos_token
    ppo_trainer = PPOTrainer(config,agent_model,ref_model,agent_tokenizer)
    
    #load target model
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model,cache_dir = args.cache_dir)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model,
                                                        cache_dir = args.cache_dir,
                                                        device_map='auto')
    target_model.config.pad_token_id = target_tokenizer.eos_token_id
    target_tokenizer.pad_token = target_tokenizer.eos_token
    
    
    

    
    #generation kwargs setting
    generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": agent_tokenizer.eos_token_id,
    "max_new_tokens":args.max_prompt_length,
    "min_length": -1,
    }
    
    
    #setting verbalizer ids
    verbalizer_ids=  []
    for i in range(len(verbalizer)):
        verbalizer_ids.append(agent_tokenizer.convert_tokens_to_ids(verbalizer[i]))
    
    queue = utils.TopAccuracyTextsNoDuplicates(max_size=5)
    #start training
    for ep in tqdm(range(args.epochs)):
        max_total_loss = 0
        min_total_loss = 0
        mean_total_loss = 0
        sum_total_loss = 0
        
        
        
        
        for batch in train_dataloader:
            inputs = batch['text']
            labels = batch['label']
            examples = utils.got_example(validation_dataset,verbalizer,shot=args.num_example)
            query_text = [
                {"role" : "user", "content" : args.meta_prompt + '\n' + examples},
                {"role": "assistant","content" : "The Instruction is : "}
            ]
            
            query_encoded = agent_tokenizer.apply_chat_template(
                query_text,
                return_tensors='pt'
            ).view(-1).to(device)
            
            response_tensors =ppo_trainer.generate(
                query_encoded,
                **generation_kwargs,
                return_prompt=False,
                num_return_sequences = args.prompt_per_example
            )
            
            used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in response_tensors]
            
            #나온 프롬프트 중 너무 길이가 짧은게 많으면 종료
            if sum([len(p) for p in used_prompt]) < args.prompt_per_example * 10:
                break
            
            rewards = []
            losses = []
            with torch.no_grad(): 
                softmax_diff,accuracys,real_rewards = utils.evaluation_soft(
                    used_prompt,
                    inputs,
                    labels,
                    target_model,
                    target_tokenizer,
                    'cuda:0',
                    verbalizer.values(),
                    debug=True,
                    return_reward = True
                )
            rewards = [  softmax_diff[i] + accuracys[i] for i in range(len(used_prompt))]
            np_rewards = np.array(rewards)
            np_acc = np.array(accuracys)
            rewards = [ torch.tensor(reward) for reward in rewards]
            for i in range(len(rewards)):
                print('reward : ', rewards[i].item(),'acc :', accuracys[i],' prompt : ', used_prompt[i], '\n')
                queue.add(rewards[i].item(),used_prompt[i],ep)
            bs = len(np_rewards)
            #print([query_encoded.view(-1) for i in range(bs)],response_tensors,[torch.tensor(reward) for reward in rewards])
            stats = ppo_trainer.step([query_encoded.view(-1) for i in range(bs)],
                         [response for response in response_tensors],
                         rewards)
            rewards = torch.stack(rewards)
            mean_reward = torch.mean(rewards)
            max_reward = torch.max(rewards)
            wandb.log({
                'rewards' : rewards,
                'mean_reward' : mean_reward,
                'max_reward' : max_reward,
            })
    print('Final test Start')
    prompt_queue = queue.get_top_texts()
    new_acc = utils.evaluation(
        [prompt[1] for prompt in prompt_queue],
        test_dataset,
        target_model,
        target_tokenizer,
        device,
        verbalizer.values(),
    )
    for i in range(len(prompt_queue)):
        print('prompt : ',prompt_queue[i][1],'acc : ',new_acc[i])
            
if __name__ == '__main__':
    main()
                
                    
                    
    
    
    