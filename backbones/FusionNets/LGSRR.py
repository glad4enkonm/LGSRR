import imp
import torch
from torch import nn
import torch.nn.functional as F

from ..SubNets import text_backbones_map

__all__=['LGSRR']

class LGSRR(nn.Module):

    def __init__(self, args):
        super(LGSRR, self).__init__()
        
        text_backbone = text_backbones_map[args.text_backbone]
        self.text_subnet = text_backbone(args)

        text_feat_dim = args.text_feat_dim

        self.weight_net = nn.Sequential(
            nn.Linear(text_feat_dim, args.weight_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.weight_dropout),
            nn.Linear(args.weight_hidden_dim, 1),
        )
        
        self.classifier = nn.Linear(text_feat_dim, args.num_labels)
    
    def forward(self, text_feats, desc_feats_all, args):
    
        bz, num_desc, _, seq_len = desc_feats_all.shape
        desc_feats_all = desc_feats_all.reshape(bz * num_desc, _, seq_len)
        
        if args.mean_pooling:
            text_feats = self.text_subnet(text_feats).mean(dim=1)
            desc_feats_all = self.text_subnet(desc_feats_all).mean(dim=1)
        else:
            text_feats = self.text_subnet(text_feats)[:, 0]
            desc_feats_all = self.text_subnet(desc_feats_all)[:, 0] 

        hidden_dim = desc_feats_all.shape[-1]
        
        desc_feats = desc_feats_all.view(bz, num_desc, hidden_dim)

        action_feats = desc_feats[:, 0, :]
        expression_feats = desc_feats[:, 1, :]
        interaction_feats = desc_feats[:, 2, :]

        sim_text_action = F.cosine_similarity(text_feats, action_feats, dim=-1).unsqueeze(1)
        sim_text_expr = F.cosine_similarity(text_feats, expression_feats, dim=-1).unsqueeze(1)
        sim_text_inter = F.cosine_similarity(text_feats, interaction_feats, dim=-1).unsqueeze(1)

        incon_text_action = F.mse_loss(text_feats, action_feats)
        incon_text_expr = F.mse_loss(text_feats, expression_feats)
        incon_text_inter = F.mse_loss(text_feats, interaction_feats)

        weights = torch.cat([self.weight_net(text_feats).unsqueeze(1), self.weight_net(desc_feats)], dim=1)
        normalized_weights = F.softmax(weights, dim=1)

        combined_feats = text_feats * normalized_weights[:, 0, :] + action_feats * normalized_weights[:, 1, :] * sim_text_action + expression_feats * normalized_weights[:, 2, :] * sim_text_expr + interaction_feats * normalized_weights[:, 3, :] * sim_text_inter

        inconsistency_penalty = incon_text_action * (text_feats - action_feats) + incon_text_expr * (text_feats - expression_feats) + incon_text_inter * (text_feats - interaction_feats)
        
        combined_feats = combined_feats - inconsistency_penalty
        
        logits = self.classifier(combined_feats)
    
        return logits, combined_feats, normalized_weights