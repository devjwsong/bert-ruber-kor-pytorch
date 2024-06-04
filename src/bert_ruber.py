from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import functional as F
from itertools import chain
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import torch
import numpy as np


class MLPNetwork(nn.Module):
    def __init__(self, 
        hidden_size: int, 
        w1_size: int, 
        w2_size: int, 
        w3_size: int,
        num_classes: int
    ) -> None:
        super(MLPNetwork, self).__init__()
        
        self.m = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.w1 = nn.Linear(2 * hidden_size + 1, w1_size)
        self.w2 = nn.Linear(w1_size, w2_size)
        self.w3 = nn.Linear(w2_size, w3_size)
        self.output_layer = nn.Linear(w3_size, num_classes)
        
    def init_params(self):
        nn.init.xavier_uniform_(self.m)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        
    def forward(self, ctx_embs: torch.Tensor, res_embs: torch.Tensor) -> torch.Tensor:
        ctx_embs, res_embs = ctx_embs.unsqueeze(1), res_embs.unsqueeze(2)  # (B, 1, d_h), (B, d_h, 1)
        feats = torch.bmm(torch.matmul(ctx_embs, self.m), res_embs).squeeze(2)  # (B, 1)
        ctx_embs, res_embs = ctx_embs.squeeze(1), res_embs.squeeze(2)

        v = torch.cat([ctx_embs, feats, res_embs], dim=-1)  # (B, 2*d_h+1)
        v = torch.tanh(self.w1(v))  # (B, w1)
        v = torch.tanh(self.w2(v))  # (B, w2)
        v = torch.tanh(self.w3(v))  # (B, w3)
        logits = self.output_layer(v) # (B, C)

        return logits


class BertRuber(nn.Module):
    def __init__(self, ckpt_path, device=torch.device('cpu')):
        super(BertRuber, self).__init__()

        # Setting arguments.
        print("Setting arguments...")
        infos = ckpt_path.split('/')[-1].split('-')
        self.max_len, self.pooling = int(infos[-2]), infos[-1]
        assert self.pooling in ["cls", "mean", "max"], "The pooling method must be among 'cls', 'mean', or 'max'."

        # Loading tokenizer, model and parameters.
        print("Loading tokenizer, model and parameters...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.bert = AutoModel.from_pretrained(ckpt_path).to(self.device)
        mlp_net_path = hf_hub_download(repo_id=ckpt_path, filename="MLPNetwork.pt")

        self.mlp_net = MLPNetwork(
            self.bert.config.hidden_size,
            self.bert.config.w1_size,
            self.bert.config.w2_size,
            self.bert.config.w3_size,
            num_classes=2,
        )
        self.mlp_net.load_state_dict(torch.load(mlp_net_path))
        self.mlp_net = self.mlp_net.to(self.device)
        
        # Inference setting.
        self.bert.eval()
        self.mlp_net.eval()
        
        # Special tokens
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        
        vocab = self.tokenizer.get_vocab()
        self.cls_id = vocab[self.cls_token]
        self.sep_id = vocab[self.sep_token]
        self.pad_id = vocab[self.pad_token]
        self.unk_id = vocab[self.unk_token]
        
        print("BERT-RUBER ready.")
        
    def encode(self, input_ids):
        # input_ids: (B, L)
        padding_masks = (input_ids != self.pad_id).float()  # (B, L)
        
        # Getting the pooled embedding from each input.
        hidden_states = self.bert(input_ids=input_ids, attention_mask=padding_masks)[0]  # (B, L, d_h)
        if self.pooling == 'cls':
            pooled = hidden_states[:, 0, :]  # (B, d_h)
        elif self.pooling == 'mean':
            pooled = torch.mean(hidden_states, dim=1)  # (B, d_h)
        elif self.pooling == 'max':
            pooled = torch.max(hidden_states, dim=1).values  # (B, d_h)
        
        return pooled
    
    def get_unref_scores(self, query_embs, res_embs):
        # query_embs: (B, d_h), res_embs: (B, d_h)
        
        # Conducting the MLPNetwork's forward function.
        logits = self.mlp_net(query_embs, res_embs)  # (B, C)
        scores = F.softmax(logits, dim=-1)[:, 1]  # (B)
        
        return scores
    
    def get_ref_scores(self, ref_embs, res_embs):
        # ref_embs: (B, d_h), res_embs: (B, d_h)
        
        # Getting the cosine similarity between reference & prediction.
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        return torch.clamp(cos(ref_embs, res_embs), 0.0, 1.0)  # (B)
    
    def get_scores(self, query_ids, ref_ids, res_ids):
        # query_ids: (B, L), ref_ids: (B, L), res_ids: (B, L)
        with torch.no_grad():
            query_embs, ref_embs, res_embs = self.encode(query_ids), self.encode(ref_ids), self.encode(res_ids)  # (B, d_h), (B, d_h), (B, d_h)
            ref_scores = self.get_ref_scores(ref_embs, res_embs)  # (B)
            unref_scores = self.get_unref_scores(query_embs, res_embs)  # (B)

        return unref_scores, ref_scores
    
    def make_query(self, query, hists=[]):
        # Pre-tokenizing.
        query = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(query))
        hists = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(hist)) for hist in hists]

        # Finding the starting turn to make each query length is equal or shorther than max length.
        query_context = [self.cls_id]
        for s in range(len(hists)):
            memory = list(chain.from_iterable(hists[s:]))
            if 1 + len(memory) + 1 + len(query) + 1 <= self.max_len:
                query_context += (memory + [self.sep_id])
                break
        query_ids = query_context + query + [self.sep_id]
        if len(query_ids) > self.max_len:
            query_ids = query_ids[:self.max_len]
            query_ids[-1] = self.sep_id
        
        return torch.LongTensor(query_ids)  # (L)
    
    def make_res(self, res) -> torch.Tensor:
        # Pre-tokenizing.
        res = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(res))

        # Making response token ids.
        res_ids = [self.cls_id] + res + [self.sep_id]
        if len(res_ids) > self.max_len:
            res_ids = res_ids[:self.max_len]
            res_ids[-1] = self.sep_id
            
        return torch.LongTensor(res_ids)  # (L)
    
    def normalize_scores(self, scores):
        min_score, max_score = np.min(scores), np.max(scores)
        scores = (scores - min_score) / (max_score - min_score)
        
        return scores
    
    def forward(self, queries, answers, predictions, hists, batch_size=1, ref_weight=0.5, normalize=True):
        # Checking the number of samples.
        assert len(queries) == len(answers), "The number of samples should be consistent between queries and answers"
        assert len(queries) == len(predictions), "The number of samples should be consistent between queries and predictions."
        assert len(queries) == len(hists), "The number of samples should be consistent between queries and dialogue histories."
        assert ref_weight >= 0.0 and ref_weight <= 1.0, "The reference score weights should be [0.0, 1.0]."
        
        # Pre-processing the data.
        print("Pre-processing data...")
        query_ids_list, ref_ids_list, pred_ids_list = [], [], []
        num_samples = len(queries)
        for i in tqdm(range(num_samples)):
            query_ids = self.make_query(queries[i], hists[i])  # (L)
            ref_ids = self.make_res(answers[i])  # (L)
            pred_ids = self.make_res(predictions[i])  # (L)

            query_ids_list.append(query_ids)
            ref_ids_list.append(ref_ids)
            pred_ids_list.append(pred_ids)
        
        print(f"The number of samples to evaluate: {len(query_ids_list)}")
        
        # Batch processing.
        print("Processing each batch...")
        batch_unref_scores, batch_ref_scores = [], []
        for b in tqdm(range(0, num_samples, batch_size)):
            start, end = b, b+batch_size
            query_ids, ref_ids, pred_ids = query_ids_list[start:end], ref_ids_list[start:end], pred_ids_list[start:end]
            query_ids = pad_sequence(query_ids, batch_first=True, padding_value=self.pad_id).to(self.device)  # (B, L)
            ref_ids = pad_sequence(ref_ids, batch_first=True, padding_value=self.pad_id).to(self.device)  # (B, L)
            pred_ids = pad_sequence(pred_ids, batch_first=True, padding_value=self.pad_id).to(self.device)  # (B, L)

            unref_scores, ref_scores = self.get_scores(query_ids, ref_ids, pred_ids)  # (B), (B)
            
            batch_unref_scores.append(unref_scores.detach())
            batch_ref_scores.append(ref_scores.detach())
            
        assert len(batch_unref_scores) == len(batch_ref_scores)
        
        # Wrapping-up the scores!
        print("Wrapping-up the scores...")
        total_unref_scores, total_ref_scores = np.array([]), np.array([])
        for b in tqdm(range(len(batch_unref_scores))):
            total_unref_scores = np.append(total_unref_scores, batch_unref_scores[b].cpu().numpy())
            total_ref_scores = np.append(total_ref_scores, batch_ref_scores[b].cpu().numpy())
        assert total_unref_scores.size == total_ref_scores.size
        
        if normalize:
            total_unref_scores = self.normalize_scores(total_unref_scores)
            total_ref_scores = self.normalize_scores(total_ref_scores)
        
        total_scores = (1.0 - ref_weight) * total_unref_scores + ref_weight * total_ref_scores
        
        print("FINISHED.")
        
        return total_scores, total_unref_scores, total_ref_scores
        