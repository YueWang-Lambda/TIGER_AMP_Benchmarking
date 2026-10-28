import torch
import torch.nn as nn
import torch.nn.functional as F
from func import *

# same role as get_masked_seqs
class Mutator(nn.Module):
    def __init__(self, seq_length):
        super(Mutator, self).__init__()
        self.length = seq_length
        self.hidden = 128
        self.emb = nn.Embedding(self.length, self.length)
        self.attention_in = nn.Linear(self.length, self.hidden)
        self.attention_out = nn.Linear(self.hidden, self.length)
        
    def forward(self, batch_seqs, input_ids):
        # batch_seqs: (batch_size * sequence_length)
        # get masked position
        batch_size = input_ids.shape[0]
        embedding = self.emb(input_ids)
        att = self.attention_in(embedding)
        att = self.attention_out(att)      # [batch_size, max_length, max_length]
        att_prob = F.softmax(att, dim = 1)
        att_prob = att_prob[:,:,:1].squeeze()  # [batch_size, max_length]

        return att_prob

# the agent network
class Agent():
    def __init__(self, model, tokenizer, max_length, device):
        # should load pretrained/fine-tuned model and tokenizer here
        self.max_length = max_length
        self.device = device

        self.mutator = Mutator(self.max_length)
        self.model = model
        self.tokenizer = tokenizer
        self.mutator.to(self.device)

    def sample(self, batch_seqs, iterations):
        """
            Sample a batch of sequences

            Args:
                batch_size : Number of sequences to sample 
                max_length:  Maximum length of the sequences

            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
        """
        batch_size = len(batch_seqs)
        sequences = []
        log_probs = Variable(torch.zeros(batch_size))

        idnumber = list(range(batch_size))
        ite_memory = pd.DataFrame({'no.': idnumber, 'seqs': batch_seqs})

        for ites in range(iterations):
            if ites != 0:
                batch_seqs = output_seqs
            ids = self.tokenizer.batch_encode_plus(batch_seqs, add_special_tokens=True, 
                                    padding='max_length', is_split_into_words=False, 
                                    max_length = self.max_length, pad_to_max_length = True, 
                                    return_tensors="pt")
            input_ids = ids['input_ids'].to(self.device)

            att_prob = self.mutator(batch_seqs, input_ids)
            log_prob = att_prob.log()
            att_prob_zero = torch.where((input_ids!=0) & (input_ids!=1), att_prob, torch.zeros_like(att_prob))
            # index for seqs
            seqid = torch.arange(self.max_length, device = self.device) + 1 # to avoid the first 0 be filtered as mask
            seqids = torch.broadcast_to(seqid, (batch_size, self.max_length))
            seqids = torch.where((input_ids!=0) & (input_ids!=1), seqids, torch.zeros_like(seqids))

            mask_aa = torch.multinomial(att_prob_zero, num_samples=1)

            # replace mask by fine-tuned Ankh
            seqs_mask = get_masked_seqs(mask_aa, batch_seqs)
            outputs = embed_dataset(self.model, self.tokenizer, self.device, seqs_mask)
            output_seqs = get_modified_seqs(batch_seqs, mask_aa, outputs)

            # replace mask randomly for ablation
            # output_seqs = random_mutation(mask_aa, batch_seqs)
            
            # whether end the episode
            # output_score = scoring_function(output_seqs)
            # distance = levenshtein_distance_matrix(output_seqs, batch_seqs)
            
            # end the episode for repeated seqs
            endseq_idx = []
            for i, seq in enumerate(output_seqs):
                prev_seqs = ite_memory[ite_memory['no.'] == i]['seqs'].values.tolist()
                if (seq in prev_seqs):
                    endseq_idx.append(i)

            df = pd.DataFrame({'no.': idnumber, 'seqs': output_seqs})
            ite_memory = ite_memory._append(df)

            if len(endseq_idx) > 0:
                log_prob[endseq_idx] = log_prob[endseq_idx] * 0.5
                # log_prob[endseq_idx] = log_prob[endseq_idx]
                # print('end the iterations of seq')
            log_probs += NLLLoss(log_prob, mask_aa.view(-1))
        sequences.append(output_seqs)
        return sequences, log_probs
