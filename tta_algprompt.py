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
    parser.add_argument('--target_model',type=str,default='google/gemma-1.1-2b-it')
    parser.add_argument('--agent_model',type=str,default='google/gemma-1.1-2b-it')
    parser.add_argument('--task',type=str,default='classification')
    parser.add_argument('--dataset',type=str,default='sst2')
    parser.add_argument(
        '--verbalizer',
        type = str,
        nargs = '+',
        default = None
    )
    parser.add_argument('--cache_dir',type=str,default='./')
    parser.add_argument('--batch_size',type=int,default=3)
    parser.add_argument('--max_prompt_length',type=int,default=100)
    parser.add_argument('--train_data_per_labels',type=int,default=10)
    parser.add_argument('--num_example',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--meta_prompt',type=str,
                        default = '''I want to give the appropriate instruction to help
                        a friend who needs to look at the input and guess the output.
                        Plase write instruction to help my friends. Here are the input-output pairs:
                        ''',)
    parser.add_argument('--prompt_per_example',type=int,default=3)
    args = parser.parse_args()
    return args

def main():
    #torch.backends.cuda.enable_mem_efficient_sdp(False)
    #torch.backends.cuda.enable_flash_sdp(False)
    args = parser_args()
    device=  'cuda:0'
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
        test_dataset = utils.create_balanced_subset(test_dataset,20)
    else:
        #TODO
        pass
        
    #make dataloader
    test_dataloader = DataLoader(test_dataset,batch_size = 1,shuffle = True)
    train_dataloader = DataLoader(train_dataset,batch_size = 1,shuffle = True)
    
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
        torch_dtype=torch.float32,
        device_map = ['cuda:0','cuda:1','cuda:2','cuda:3'],
        peft_config = lora_config,
        cache_dir = args.cache_dir
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.agent_model,
        torch_dtype=torch.float32,
        device_map = ['cuda:0','cuda:1','cuda:2','cuda:3'],
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
                {"role":"assistant","content" : "Sure. Let me see what input friend sees "},
                {"role" : "user", "content" : "The input that friend sees : " + inputs[0]},
                {"role": "assistant","content" : "The Instruction is : "}
            ]
            
            query_encoded = agent_tokenizer.apply_chat_template(
                query_text,
                return_tensors='pt'
            ).view(-1)
            
            response_tensors =ppo_trainer.generate(
                query_encoded.to(device),
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
                for prompt in used_prompt:
                    template = prompt + "Input : " + inputs[0] + "Output : "
                    prompt_encoded = target_tokenizer(template,return_tensors='pt').to(device)
                    #print(prompt_encoded)
                    outputs = target_model(**prompt_encoded)
                    logits = outputs.logits
                    verbalizer_logits = logits[:, -1, verbalizer_ids]
                    label = torch.tensor(labels).to(device)
                    loss = -torch.nn.functional.cross_entropy(verbalizer_logits,label).item()
                    rewards.append(loss - len(prompt) * 0.00001)
                    losses.append(loss)
            np_rewards = np.array(rewards)
            pt_rewards = [torch.tensor(reward_) for reward_ in rewards]
            bs = len(pt_rewards)
            stats = ppo_trainer.step(
                [query_encoded] * bs,
                [response for response in response_tensors],
                pt_rewards,
            )
            max_total_loss += max(losses)
            min_total_loss += min(losses)
            mean_total_loss += np.mean(losses)
            sum_total_loss += sum(losses)
            #print(losses,used_prompt)
        print('Max Total Loss : ', max_total_loss)
        print('Min Total Loss : ', min_total_loss)
        print('Mean Total Loss : ', mean_total_loss)
        print('Sum Total Loss : ', sum_total_loss)
        wandb.log({
            'max_total_loss' : max_total_loss,
            'min_total_loss' : min_total_loss,
            'mean_total_loss' : mean_total_loss,
            'sum_total_loss' : sum_total_loss
        })
        
        #start evaluation
        if ep % 5 == 0:
            test_acc = 0
            test_total= 0
            print('start test')
            #evaluation
            for batch in tqdm(test_dataloader):
                with torch.no_grad():
                    inputs = batch['text']
                    labels = batch['label']
                    examples = utils.got_example(validation_dataset,verbalizer,shot=args.num_example)
                    query_text = [
                        {"role" : "user", "content" : args.meta_prompt + '\n' + examples},
                        {"role":"assistant","content" : "Sure. Let me see what input friend sees "},
                        {"role" : "user", "content" : "The input that friend sees : " + inputs[0]},
                        {"role": "assistant","content" : "The Instruction is : "}
                    ]
                    query_encoded = agent_tokenizer.apply_chat_template(
                        query_text,
                        return_tensors='pt'
                    ).view(-1)
                    response_tensors =ppo_trainer.generate(
                        query_encoded,
                        **generation_kwargs,
                        return_prompt=False,
                        num_return_sequences = 1
                    )
                    used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in response_tensors]
                    prompt = used_prompt[0]
                    template = prompt + "Input : " + inputs[0] + "Output : "
                    prompt_encoded = target_tokenizer(template,return_tensors='pt').to(device)
                    outputs = target_model(**prompt_encoded)
                    logits = outputs.logits
                    verbalizer_logits = logits[:, -1, verbalizer_ids]
                    label= labels
                    if torch.argmax(verbalizer_logits).item() == label:
                        test_acc += 1
                    #print(torch.argmax(verbalizer_logits).item(),label,test_acc)
                    test_total+=1
            print('Test Accuracy : ', test_acc / test_total)
            wandb.log({
                'test_acc' : test_acc / test_total
            })
                
        
            
if __name__ == '__main__':
    main()
                
                    
                    
    
    
    