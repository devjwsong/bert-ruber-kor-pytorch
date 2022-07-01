from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
from tqdm import tqdm

import json
import torch
import numpy as np
import random


class CustomDataset(Dataset):
    def __init__(self, args, file, tokenizer):
        self.query_ids, self.res_ids, self.labels = [], [], []
        query_lens, res_lens = [], []
        
        # Loading the data.
        with open(file, 'r') as f:
            data = json.load(f)
        random.seed(args.seed)
        data = data[:10]

        # Iterating & pre-processing each dialogue.
        for d, dial in enumerate(tqdm(data)):
            # Making candidate index list for negative sampling.
            dial_ids = list(range(len(data)))
            dial_ids = dial_ids[:d] + dial_ids[d+1:]

            assert d not in dial_ids, f"The current dialogue should not be in the candidate dialogue list."
            assert len(dial_ids) == len(data)-1, f"The candidate size should be the total size - 1."
            
            # Pre-tokenization.
            hists = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(turn)) for turn in dial]

            # Processing each turn.
            for t in range(1, len(hists)):
                query, pos = hists[t-1], hists[t]

                # Negative sampling.
                dial_id = random.choice(dial_ids)
                neg = random.choice(data[dial_id])
                neg = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(neg))

                # Making positive & negative sample.
                pos_ids = [args.cls_id] + pos + [args.sep_id]
                neg_ids = [args.cls_id] + neg + [args.sep_id]
                if len(pos_ids) > args.max_len:
                    pos_ids = pos_ids[:args.max_len]
                    pos_ids[-1] = args.sep_id
                if len(neg_ids) > args.max_len:
                    neg_ids = neg_ids[:args.max_len]
                    neg_ids[-1] = args.sep_id

                # Next, the context for user query. If no extra history is used, this becomes a single [CLS] token.
                query_context = [args.cls_id]
                for s in range(max(t-1-args.num_hists, 0), t-1):
                    memory = list(chain.from_iterable(hists[s:t-1]))
                    seq_len = 1 + len(memory) + 1 + len(query) + 1
                    if seq_len <= args.max_len:
                        query_context += (memory + [args.sep_id])
                        break

                query_ids = query_context + query + [args.sep_id]
                if len(query_ids) > args.max_len:
                    query_ids = query_ids[:args.max_len]
                    query_ids[-1] = args.sep_id

                # The final pre-processed results.
                self.query_ids.append(query_ids)
                self.res_ids.append(pos_ids)
                self.labels.append(1)
                query_lens.append(len(query_ids))
                res_lens.append(len(pos_ids))

                self.query_ids.append(query_ids)
                self.res_ids.append(neg_ids)
                self.labels.append(0)
                query_lens.append(len(query_ids))
                res_lens.append(len(neg_ids))
        
        assert len(self.query_ids) == len(self.res_ids), "The numbers of queries and responses are different."
        assert len(self.query_ids) == len(self.labels), "The numbers of queries and labels are different."
        assert len(self.query_ids) == len(query_lens), "The numbers of queries and query lengths are different."
        assert len(self.query_ids) == len(res_lens), "The numbers of queries and response lengths are different."

        print(f"The maximum length of queries: {np.max(query_lens)}")
        print(f"The minimum length of queries: {np.min(query_lens)}")
        print(f"The average length of queries: {np.mean(query_lens)}")
        print(f"The maximum length of responses: {np.max(res_lens)}")
        print(f"The minimum length of responses: {np.min(res_lens)}")
        print(f"The average length of responses: {np.mean(res_lens)}")
        
    def __len__(self):
        return len(self.query_ids)
    
    def __getitem__(self, idx):
        return self.query_ids[idx], self.res_ids[idx], self.labels[idx]

    
class PadCollate():
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad_collate(self, batch):
        # Extracting parsed objs from __getitem__.
        query_ids, res_ids, labels = [], [], []
        for obj in batch:
            query_ids.append(torch.LongTensor(obj[0]))
            res_ids.append(torch.LongTensor(obj[1]))
            labels.append(obj[2])
        
        # Padding.
        query_ids = pad_sequence(query_ids, batch_first=True, padding_value=self.pad_id)
        res_ids = pad_sequence(res_ids, batch_first=True, padding_value=self.pad_id)
        labels = torch.LongTensor(labels)
        
        # Setting contiguous for memory efficiency.
        return query_ids.contiguous(), res_ids.contiguous(), labels.contiguous()
