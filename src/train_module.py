from transformers import AutoModel, AutoTokenizer
from transformers import get_polynomial_decay_schedule_with_warmup
from pytorch_lightning import seed_everything
from argparse import Namespace
from torch import nn
from sklearn.metrics import f1_score
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from bert_ruber import MLPNetwork

import torch
import pytorch_lightning as pl
import numpy as np


class TrainModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        
        # Setting the arguments.
        if isinstance(args, dict):
            args = Namespace(**args)
            
        self.args = args
        seed_everything(self.args.seed, workers=True)
        
        # Loading the tokenizer & model.
        print("Loading the model & the tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        self.bert = AutoModel.from_pretrained(self.args.model_path)
        
        # Setting essential arguements.
        self.args.max_len = min(self.args.max_len, self.bert.config.max_position_embeddings)
        self.args.hidden_size = self.bert.config.hidden_size

        self.args.cls_token = self.tokenizer.cls_token
        self.args.sep_token = self.tokenizer.sep_token
        self.args.pad_token = self.tokenizer.pad_token
        self.args.unk_token = self.tokenizer.unk_token

        vocab = self.tokenizer.get_vocab()
        self.args.cls_id = vocab[self.args.cls_token]
        self.args.sep_id = vocab[self.args.sep_token]
        self.args.pad_id = vocab[self.args.pad_token]
        self.args.unk_id = vocab[self.args.unk_token]
        
        # Defining additional layers
        self.mlp_net = MLPNetwork(
            self.args.hidden_size,
            self.args.w1_size,
            self.args.w2_size,
            self.args.w3_size,
            num_classes=2,
        )
        self.mlp_net.init_params()
        
        # Defining loss functions
        self.loss_func = nn.CrossEntropyLoss()
            
        # Saving arguments for the later usage of a checkpoint.
        self.save_hyperparameters(self.args)
        
    def forward(self, query_ids, res_ids):
        # query_ids: (B, L), res_ids: (B, L)
        query_embs, res_embs = self.encode(query_ids), self.encode(res_ids)  # (B, d_h), (B, d_h)
        logits = self.mlp_net(query_embs, res_embs)  # (B, C)
        
        return logits
        
    def encode(self, input_ids):
        # input_ids: (B, L)
        padding_masks = (input_ids != self.args.pad_id).float()  # (B, L)
        
        # Getting the pooled embedding from each input.
        hidden_states = self.bert(input_ids=input_ids, attention_mask=padding_masks)[0]  # (B, L, d_h)
        if self.args.pooling == 'cls':
            pooled = hidden_states[:, 0, :]  # (B, d_h)
        elif self.args.pooling == 'mean':
            pooled = torch.mean(hidden_states, dim=1)  # (B, d_h)
        elif self.args.pooling == 'max':
            pooled = torch.max(hidden_states, dim=1).values  # (B, d_h)
        
        return pooled
    
    def training_step(self, batch, batch_idx):
        # Loading batch.
        query_ids, res_ids, labels = batch  # (B, L), (B, L), (B)
        logits = self.forward(query_ids, res_ids)  # (B, C)
        
        loss = self.loss_func(logits, labels)  # ()
        preds = torch.max(logits, dim=-1).indices  # (B)
        
        return {
            'loss': loss, 'preds': preds.detach(), 'labels': labels.detach()
        }
    
    def training_epoch_end(self, training_step_outputs):
        # Parsing the results from each training step.
        train_losses, train_preds, train_labels = [], [], []
        for result in training_step_outputs:
            train_losses.append(result['loss'].item())
            train_preds += result['preds'].tolist()
            train_labels += result['labels'].tolist()
        
        train_loss = np.mean(train_losses)
        train_f1 = f1_score(train_labels, train_preds, average='micro')
        
        # Logging train loss & f1 score.
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def validation_step(self, batch, batch_idx):
        # Loading batch.
        query_ids, res_ids, labels = batch  # (B, L), (B, L), (B)
        logits = self.forward(query_ids, res_ids)  # (B, C)
        
        loss = self.loss_func(logits, labels)  # (B)
        preds = torch.max(logits, dim=-1).indices  # (B)
        
        return {
            'loss': loss, 'preds': preds.detach(), 'labels': labels.detach()
        }
    
    def validation_epoch_end(self, validation_step_outputs):
        # Parsing the results from each validation step.
        valid_losses, valid_preds, valid_labels = [], [], []
        for result in validation_step_outputs:
            valid_losses.append(result['loss'].item())
            valid_preds += result['preds'].tolist()
            valid_labels += result['labels'].tolist()
        
        valid_loss = np.mean(valid_losses)
        valid_f1 = f1_score(valid_labels, valid_preds, average='micro')
        
        # Logging validation loss & f1 score.
        self.log('valid_loss', valid_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_f1', valid_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        # Loading batch.
        query_ids, res_ids, labels = batch  # (B, L), (B, L), (B)
        logits = self.forward(query_ids, res_ids)  # (B, C)
        
        loss = self.loss_func(logits, labels)  # (B)
        preds = torch.max(logits, dim=-1).indices  # (B)
        
        return {
            'loss': loss, 'preds': preds.detach(), 'labels': labels.detach()
        }
    
    def test_epoch_end(self, test_step_outputs):
        # Parsing the results from each test step.
        test_losses, test_preds, test_labels = [], [], []
        for result in test_step_outputs:
            test_losses.append(result['loss'].item())
            test_preds += result['preds'].tolist()
            test_labels += result['labels'].tolist()
        
        test_loss = np.mean(test_losses)
        test_f1 = f1_score(test_labels, test_preds, average='micro')
        
        # Logging test loss & f1 score.
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', test_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
   
    def configure_optimizers(self):        
        optim = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        if self.args.num_warmup_steps < 0.0:
            return [optim]
        else:
            sched = {
                'scheduler': get_polynomial_decay_schedule_with_warmup(
                    optim,
                    num_warmup_steps=self.args.num_warmup_steps,
                    num_training_steps=self.args.num_training_steps,
                    power=2
                ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1
            }

            return [optim], [sched]
        
        