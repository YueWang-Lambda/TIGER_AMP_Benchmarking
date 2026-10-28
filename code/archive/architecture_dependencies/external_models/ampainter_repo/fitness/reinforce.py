import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import logging
logging.set_verbosity_error()

import ankh
import pathlib
import random
import torch
import numpy as np
import pandas as pd
from time import time
from tqdm.auto import tqdm

from network import Agent
from func import *

seed = 1234
torch.manual_seed(seed)
random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Available device:', device)

# hyperparameters
learning_rate = 3e-4
batch_size = 16
epoches = 2

# generation parameters
masking_ratio = 0.2     # 0.2 for toy data, 0.5 for 7316 training
max_length = 41
num_beams = 10
temperature = 2.0


def train_agent(save_path, device, 
                model, tokenizer, 
                max_length, agent_path, 
                learning_rate, batch_size, batchs_seqs, n_steps, iterations, 
                experience_replay, early_stop):
    
    start_time = time()

    agent = Agent(model, tokenizer, max_length, device)
    # load trained agent weights
    agent.mutator.load_state_dict(torch.load(agent_path, map_location = device))
    
    optimizer = torch.optim.Adam(agent.mutator.parameters(), lr=learning_rate)
    agent.mutator.eval()
    experience = Inception()

    # Information for the logger
    step_log = [[], [], []]

    best_loss, early_stop_count = 100, 0
    for step in tqdm(range(n_steps)):
        mean_train_loss = []
        # mannualy define seqs per batch
        batch_num = len(batchs_seqs) // batch_size
        if (len(batchs_seqs) % batch_size) > 0:
            batch_num = batch_num + 1
        
        for bn in range(batch_num):
            start_idx = bn * batch_size
            if bn < batch_num:
                end_idx = (bn + 1) * batch_size
                batch_seqs = batchs_seqs[start_idx:end_idx]
            else:
                batch_seqs = batchs_seqs[start_idx:]

            # Sample from Agent
            seqs, agent_likelihood, batch_ite_memory = agent.sample(batch_seqs, iterations, bn)
            if len(seqs) == 1:
                seqs = seqs[0]

            # Get rewards
            score = scoring_function(seqs)

            # Calculate loss (== gamma = 1)
            loss = - agent_likelihood * Variable(score)

            # Then add new experience
            step_numbers = list(np.repeat(step+1, len(score)))
            experience.add_experience(seqs, score, agent_likelihood, step_numbers)

            # Calculate loss
            loss = loss.mean()
            '''
            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_train_loss.append(loss.detach().item())
            '''
            # Convert to numpy arrays so that we can print them
            agent_likelihood = agent_likelihood.data.cpu().numpy()

            if bn == 0:
                step_likelihood = agent_likelihood
                step_scores = score
                step_seqs = seqs
                ite_memory = batch_ite_memory
            else:           
                step_likelihood = np.concatenate((step_likelihood, agent_likelihood), 0)
                step_scores = np.concatenate((step_scores, score), 0)
                step_seqs = step_seqs + seqs
                ite_memory = ite_memory._append(batch_ite_memory)
        # save iteration memory
        ite_memory.to_csv(save_path + str(step+1) + 'step_iteration_memory.csv')

        # Print some information for this step
        time_elapsed = (time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Time elapsed: {:.2f}h Time left: {:.2f}h".format(step, time_elapsed, time_left))
        print("  Agent   Score              Seq")
        for i in range(4):
            print(" {:6.2f}   {:6.2f}             {}".format(step_likelihood[i],
                                                            step_scores[i],
                                                            step_seqs[i]))
        
        # save step outputs for evaluation
        save_step_record(step_seqs, step_scores, step_likelihood, os.path.join(save_path, str(step+1) + 'step_record.csv'))
        # mean_train_loss = np.array(mean_train_loss).mean()

        # Need this for Vizard plotting
        step_log[0].append(step + 1)
        step_log[1].append(np.mean(step_scores))
        # step_log[2].append(mean_train_loss)
        # print(f'Step [{step+1}/{n_steps}]: Train loss: {mean_train_loss:.4f}')

        if (step+1) % 5 == 0:
        # Save files
            experience.save_memory(os.path.join(save_path, "memory.csv"))
            with open(os.path.join(save_path, 'step_score.csv'), 'w') as f:
                f.write("step,score,loss\n")
                for s1, s2, s3 in zip(step_log[0], step_log[1], step_log[2]):
                    f.write(str(s1) + ',' + str(s2) + ',' + str(s3) + "\n")


if __name__ == "__main__":
    # Load Ankh-base model
    model, tokenizer = ankh.load_base_model(generation=True)

    # Fine-tuned model
    finetune_dir = '../../AMPainterV0/finetune/model_1by1/'
    name = 'lr' + str(learning_rate) + '_ba' + str(batch_size) + '_epo' + str(epoches) + '_temp' + str(temperature) + '_beam' + str(num_beams)
    path = os.path.join(finetune_dir, name)
    model = model.from_pretrained(path)
    tokenizer = tokenizer.from_pretrained(path)
    print('load fine-tuned model: ', name)

    model.eval()
    model.to(device=device)

    # load input seqs
    df_seq = pd.read_csv('./top100.txt', header=None)
    initial_seq = df_seq.iloc[:100, 0].values.tolist()
    print('input sequence numbers: ', len(initial_seq))

    # For evolution: load trained agents.ckpt
    agent_path = './Agent_fitness_ite8.ckpt'

    t0 = time()
    train_agent(save_path = './outputs/',
                device = device, 
                model = model, 
                tokenizer = tokenizer, 
                agent_path = agent_path, 
                max_length = max_length, 
                learning_rate = 1e-3, batch_size = 128, 
                batchs_seqs = initial_seq, n_steps = 10, iterations = 8, 
                experience_replay = False, early_stop = 20)
    t1 = time()
    print("Use time: {:.4f}s".format(t1 - t0))
