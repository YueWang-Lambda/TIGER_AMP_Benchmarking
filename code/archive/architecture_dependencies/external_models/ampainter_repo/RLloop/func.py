import torch
import random
import numpy as np
import pandas as pd

import sys
sys.path.append('../HyperAMP')
from predict import *

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return (tensor).cuda()
    return torch.autograd.Variable(tensor)

def get_masked_seqs(batch_mask_aa, seqs):
    seqs_mask = []
    for i, seq in enumerate(seqs):
        seqls = list(seq)
        mask_aa = batch_mask_aa[i]
        if len(mask_aa) == 0:
            seqs_mask.append(seqs)
        else:
            for k, n in enumerate(mask_aa):
                n = int(n) 
                seqls[n] = f"<extra_id_{k}>"    # replace masked aa to masking token
            seqs_mask.append(''.join(token for token in seqls))
    return seqs_mask

def embed_dataset(model, tokenizer, device, sequences, maximum_length = 41, num_beams = 10, temperature = 2.0):
    outputs = []
    with torch.no_grad():
        for (i, sample) in enumerate(sequences):
            ids = tokenizer.batch_encode_plus([sample], add_special_tokens=True, 
                                              padding=True, is_split_into_words=False, 
                                              return_tensors="pt")
            input_ids = ids['input_ids'].to(device)
            generated = model.generate(input_ids=input_ids, temperature = temperature,
                                max_length = maximum_length,
                                num_beams = num_beams,
                                do_sample=True if temperature > 0 else False)
            output_ids = generated[0].squeeze()
            # print('output_ids: ',output_ids)
            generated_tokens = list(tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            outputs.append(generated_tokens)
    return outputs

def get_modified_seqs(initial_seqs, mask_aa, output_aa):
    output_seqs = []
    for i, seq in enumerate(initial_seqs):
        seqls = list(seq)
        mutations = output_aa[i]
        if len(mutations) >= len(mask_aa[i]):
            for k, n in enumerate(mask_aa[i]):
                seqls[n] = mutations[k]
        else:
            for k, __ in enumerate(mutations):
                n = mask_aa[i][k]
                seqls[n] = mutations[k]
            print(f'output AA for seq {seq} less than masks!')
        output_seqs.append(''.join(token for token in seqls))
    return output_seqs
    

def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """
    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss


def scoring_function(seqs):
    # transform the input format for load_data (HyperAMP/utils.py)
    target_keys = []
    y = []
    for i, __ in enumerate(seqs):
        target_keys.append('seq' + str(i))
        y.append(0)
    df_data = pd.DataFrame(zip(target_keys, seqs, y))
    df_data.rename(columns={0:'name', 1: 'seq', 2:'label'}, inplace=True)

    # predicted logMIC values (func test from predict.py)
    model_path= '../HyperAMP/hyp_best_model_V1'
    preds = test(df_data, model_path)
    # process outliers
    scores = []
    for pred in preds:
        if pred < 0:
            scores.append(0)
        elif pred > 1:
            scores.append(1)
        else:
            scores.append(pred)
    return np.array(scores, dtype=np.float32)


class Inception(object):
    def __init__(self, memory_max_size=1000):
        self.memory: pd.DataFrame = pd.DataFrame(columns=['seqs', 'score', 'likelihood'])
        self.memory_max_size = memory_max_size

    def add_experience(self, seqs, score, likelihood):
        df = pd.DataFrame({"seqs": seqs, "score": score, "likelihood": likelihood.detach().cpu().numpy()})
        self.memory = self.memory._append(df)
        self._purge_memory()

    def _purge_memory(self):
        unique_df = self.memory.drop_duplicates(subset=["seqs"])
        sorted_df = unique_df.sort_values('score', ascending=False)
        self.memory = sorted_df.head(self.memory_max_size)

    def sample(self, sample_size):
        sample_size = min(len(self.memory), sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size)
            seqs = sampled["seqs"].to_list()
            scores = sampled["score"].to_list()
            agent_likelihood = sampled["likelihood"].to_list()
            return seqs, np.array(scores), np.array(agent_likelihood)
        return [], [], []
    
    def save_memory(self, path):
        self.memory[['seqs', 'score']].to_csv(path, index=None)
        
    def __len__(self):
        return len(self.memory)

def save_step_record(seqs, scores, likelihood, path):
    df = pd.DataFrame(zip(seqs, scores, likelihood))
    df.to_csv(path, header=False)

def random_mutation(batch_mask_aa, seqs):
    AAs = 'ACDEFGHIKLMNPQRSTVWY'
    outputs = []
    for i, seq in enumerate(seqs):
        seqls = list(seq)
        mask_aa = batch_mask_aa[i]
        if len(mask_aa) == 0:
            outputs.append(seqs)
        else:
            for k, n in enumerate(mask_aa):
                n = int(n) 
                seqls[n] = random.choice(AAs)    # replace masked aa to random residue
            outputs.append(''.join(token for token in seqls))
    return outputs
