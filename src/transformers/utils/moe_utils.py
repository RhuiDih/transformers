from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

def make_moe_modules(moe_class_partial, config_class):

    class MoeModules(nn.Module):

        def __init__(self, config: config_class, out_features: int):

            super().__init__()
            self.config = config
            self.num_experts = config.num_local_experts
            self.top_k = config.num_experts_per_tok
            self.always_on_idx = config.always_on_idx
            self.deep_router = config.deep_router
            
            self.experts = nn.ModuleList([moe_class_partial() for _ in range(self.num_experts)])
            self.out_features = out_features

            if self.deep_router:
                inter_dim = config.hidden_size // 4
                self.router = nn.Sequential(
                    nn.Linear(config.hidden_size, inter_dim, bias=False),
                    nn.Hardswish(inplace=True),
                    nn.Linear(inter_dim, out_features*self.num_experts, bias=False)
                )
            else:
                self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
            
            # storing purpose, not to disrupt original forward
            self.router_logits = None

        def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:

            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.router(hidden_states)

            if self.deep_router:
                router_logits = router_logits.view(-1, self.out_features, self.num_experts)

            if self.config.output_router_logits:
                self.router_logits = router_logits
            
            if self.always_on_idx == -1:
                routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
            else:
                always_on_constant, _ = torch.topk(router_logits, self.top_k, dim=-1)
                router_logits[:,self.always_on_idx] += always_on_constant.detach().sum(-1)
                routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, self.out_features),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            
            # selected_experts #bseq, (dim), topk
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts) # bseq, (dim), topk, exp
            if self.deep_router:
                expert_mask = expert_mask.permute(3,2,0,1) # exp, topk, bseq, out_feat
                #expert_mask.view(self.num_experts, self.top_k, -1) 
            else:
                expert_mask = expert_mask.permute(2,1,0) # exp, topk, bseq
            
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                #import pdb; pdb.set_trace()

                if not self.deep_router:
                    topk_idx, bseq_idx = torch.where(expert_mask[expert_idx])
                    if bseq_idx.shape[0] == 0:
                        continue
                    topk_idx_list = topk_idx.tolist()
                    bseq_idx_list = bseq_idx.tolist()
                    expert_weight = routing_weights[bseq_idx_list, topk_idx_list, None]
                else:
                    topk_idx, bseq_idx, feat_idx = torch.where(expert_mask[expert_idx])
                    if feat_idx.shape[0] == 0:
                        continue
                    topk_idx_list = topk_idx.tolist()
                    bseq_idx_list = bseq_idx.tolist()
                    feat_idx_list = feat_idx.tolist()
                    expert_weight = routing_weights[bseq_idx_list, feat_idx_list, topk_idx_list, None]

                current_state = hidden_states[None, bseq_idx_list].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(
                    current_state, *args, **kwargs
                ) * expert_weight

                final_hidden_states.index_add_(0, bseq_idx, current_hidden_states.to(hidden_states.dtype))

            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.out_features)

            return final_hidden_states
        
    return MoeModules

def make_moa_modules(attention_class, config_class):

    class MixtureOfAttention(attention_class):

        def __init__(self, config: config_class, layer_idx:int):
            
            super().__init__(config, layer_idx)
            self.num_experts = config.num_local_experts
            self.top_k = config.num_experts_per_tok
            self.always_on_idx = config.always_on_idx
            self.moe_query = config.moe_query
            self.moe_key = config.moe_key
            self.moe_value = config.moe_value

            # replacing
            if self.moe_query:
                LINEAR_CLS = make_moe_modules(partial(
                    nn.Linear,
                    in_features=self.q_proj.in_features,
                    out_features=self.q_proj.out_features,
                    bias=self.q_proj.bias),
                    config_class,)
                out_features = self.q_proj.out_features
                delattr(self, "q_proj")
                self.q_proj = LINEAR_CLS(config=config, out_features=out_features)
            if self.moe_key:
                LINEAR_CLS = make_moe_modules(partial(
                    nn.Linear,
                    in_features=self.k_proj.in_features,
                    out_features=self.k_proj.out_features,
                    bias=self.k_proj.bias),
                    config_class
                    )
                out_features = self.k_proj.out_features
                delattr(self, "k_proj")
                self.k_proj = LINEAR_CLS(config=config, out_features=out_features)
            if self.moe_value:
                LINEAR_CLS = make_moe_modules(partial(
                    nn.Linear,
                    in_features=self.v_proj.in_features,
                    out_features=self.v_proj.out_features,
                    bias=self.v_proj.bias),
                    config_class)
                out_features = self.v_proj.out_features
                delattr(self, "v_proj")
                self.v_proj = LINEAR_CLS(config=config, out_features=out_features)
        
        @property
        def router_logits(self):
            router_logits = ()
            if self.moe_query:
                router_logits += (self.q_proj.router_logits,)
            if self.moe_key:
                router_logits += (self.k_proj.router_logits,)
            if self.moe_value:
                router_logits += (self.v_proj.router_logits,)
            return router_logits

    return MixtureOfAttention