import os
import torch
import torch.nn.functional as F
import logging
import numpy as np
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from data.utils import get_dataloader
from utils.metrics import AverageMeter, Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from .loss import neuralNDCG

__all__ = ['LGSRR']

class LGSRR:

    def __init__(self, args, data, model):
             
        self.logger = logging.getLogger(args.logger_name)
        self.device, self.model = model.device, model._set_model(args)
        self.model.to(self.device)
        self.optimizer, self.scheduler = self._set_optimizer(args, self.model)

        mm_data = data.data
        mm_dataloader = get_dataloader(args, mm_data)
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
            
        self.args = args
        args.feat_size = args.text_feat_dim
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)
        
        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)
            
    def _set_optimizer(self, args, model):
        
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
        
        num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps= int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        
        return optimizer, scheduler

    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                desc_feats = batch['desc_feats'].to(self.device).permute(0, 2, 1, 3)
                label_ids = batch['label_ids'].to(self.device)
                rank_data = batch['rank_data'].to(self.device)

                zeros = torch.zeros_like(rank_data).to(self.device)
                importance_orders = torch.cat([zeros[:, 0].unsqueeze(1), rank_data + 1], dim=1)
                relevance_values = torch.tensor([5.0, 3.0, 2.0, 1.0], device=self.device).unsqueeze(0).expand_as(importance_orders)    
                relevance_values = torch.gather(relevance_values, 1, importance_orders).to(self.device)
                
                with torch.set_grad_enabled(True):

                    logits, _, weights = self.model(text_feats, desc_feats, args)
                    
                    loss = self.criterion(logits, label_ids) + args.gamma * neuralNDCG(weights.squeeze(), relevance_values, args)
                    self.optimizer.zero_grad()
                    
                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))   

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    self.optimizer.step()
                    self.scheduler.step()
            
            outputs = self._get_outputs(args, self.eval_dataloader)
            eval_score = outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'eval_score': round(eval_score, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model  
        
        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)   

    def _get_outputs(self, args, dataloader, show_results = False):

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.text_feat_dim)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            desc_feats = batch['desc_feats'].to(self.device).to(self.device).permute(0, 2, 1, 3)
            label_ids = batch['label_ids'].to(self.device)

            with torch.set_grad_enabled(False):
                
                logits, features, _ = self.model(text_feats, desc_feats, args)
                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_logit = total_logits.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_prob = total_maxprobs.cpu().numpy()
        y_feat = total_features.cpu().numpy()
        
        outputs = self.metrics(y_true, y_pred, show_results = show_results)

        outputs.update(
            {
                'y_prob': y_prob,
                'y_logit': y_logit,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_feat': y_feat
            }
        )

        return outputs

    def _test(self, args):
        
        test_results = {}
        
        ind_outputs = self._get_outputs(args, self.test_dataloader, show_results = True)
        if args.train:
            ind_outputs['best_eval_score'] = round(self.best_eval_score, 4)
        
        test_results.update(ind_outputs)
        
        return test_results