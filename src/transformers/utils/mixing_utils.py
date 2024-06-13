import os, logging, shutil, inspect
from typing import List
from tqdm import tqdm
import gc

import torch
from transformers import (
    set_seed,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM
)
from transformers.modeling_utils import no_init_weights
from datasets import load_from_disk

from transformers import (
    MoeLlamaConfig,
    MoeLlamaForCausalLM,
    MoeMistralForCausalLM,
    MoeMistralConfig
)

MODELS_MOE_CLASS = {
    LlamaForCausalLM: (MoeLlamaForCausalLM, MoeLlamaConfig),
    MistralForCausalLM: (MoeMistralForCausalLM, MoeMistralConfig),
}

ROUTERS_NAME = ".router."
EXPERTS_NAME = ".experts."


@torch.no_grad()
def mix(
    base_model:str,
    ingredients:List[str],
    modules_to_mix:List[str],
    positive_tokens:List[str]=[],
    num_samples:int=1000,
    num_experts_per_tok:int=2,
    deep_router:bool=False
):
    
    set_seed(1399)

    logging.info("loading base model...")
    config_base = AutoConfig.from_pretrained(base_model)
    model_base = AutoModelForCausalLM.from_pretrained(base_model)
    model_base_type = type(model_base)
    sd_base = model_base.state_dict()
    MOE_MODEL_CLS, MOE_CFG_CLS = MODELS_MOE_CLASS[model_base_type]
    #MOE_CFG_CLS.register_for_auto_class()
    #MOE_MODEL_CLS.register_for_auto_class("AutoModelForCausalLM")
    
    # SUPPORT CHECK
    assert num_experts_per_tok <= len(ingredients)

    assert len(modules_to_mix)>0, \
        "Modules to mix must have at least 'mlp'!"
    
    assert any([isinstance(model_base, supported_model) for supported_model in MODELS_MOE_CLASS.keys()]), \
        "Model not supported! Only supports {}!".format(MODELS_MOE_CLASS.keys())
    
    if positive_tokens:
        assert len(positive_tokens) == len(ingredients)

    # /SUPPORT CHECK
    
    logging.info("creating base model...")
    config_base.torch_dtype = torch.float16
    config = MOE_CFG_CLS(
        num_local_experts= len(ingredients),
        moe_mlp ="mlp" in modules_to_mix, 
        moe_query="q_proj" in modules_to_mix,
        moe_key = "k_proj" in modules_to_mix,
        moe_value="v_proj" in modules_to_mix,
        num_experts_per_tok=num_experts_per_tok,
        deep_router=deep_router,
        **config_base.to_dict()
    )
    
    logging.info(config)
    logging.info("creating moe model...")

    with no_init_weights():
        moe_model = MOE_MODEL_CLS(config) 
    moe_sd = moe_model.state_dict()
    
    experts_keys = [] # to be replaced with ingredients weights later
    routers_keys = [] # to be replaced later, if positive tokens are provided
    base_keys = [] # no use currently

    stem_param = 0
    experts_param = 0
    routers_param = 0

    for key in moe_sd:

        has_key_in_modules_to_mix = any([f"{x}{EXPERTS_NAME}" in key for x in modules_to_mix])

        # stem
        if not has_key_in_modules_to_mix and not ROUTERS_NAME in key:
            logging.info(f"copying {key} from base...")
            moe_sd[key].copy_(sd_base.pop(key))
            base_keys.append(key)
            stem_param += moe_sd[key].numel()

        # router
        elif ROUTERS_NAME in key:
            if len(positive_tokens):
                logging.info(f"zeroing {key}...")
                torch.nn.init.zeros_(moe_sd[key])
            else:
                logging.info(f"randomizing {key}...")
                torch.nn.init.normal_(moe_sd[key], mean=0, std=moe_model.config.initializer_range)
            routers_keys.append(key)
            routers_param += moe_sd[key].numel()
        
        #  experts
        elif has_key_in_modules_to_mix:
            experts_keys.append(key)
            experts_param += moe_sd[key].numel()

        else:
            raise Exception("Something wrong!")
    
    if positive_tokens:
        for expert_idx in range(len(ingredients)):
            tokens_path = positive_tokens[expert_idx]
            tokens = load_from_disk(tokens_path)
            if isinstance(tokens, dict): tokens = tokens["train"]
            tokens = tokens["input_ids"][:num_samples]
            #ingred_model.cuda().eval()
            model_base.cuda().eval()
            logging.info("Computing hidden states using positive tokens from {}".format(tokens_path))
            for token_idx in tqdm(range(len(tokens))):
                #_hidden_states: List = ingred_model(
                _hidden_states: List = model_base(
                    torch.tensor(tokens[token_idx]).unsqueeze(0).cuda(),
                    output_hidden_states=True,
                    return_dict=True
                ).hidden_states[:-1]
                _hidden_states = torch.stack(_hidden_states, dim=0).mean(-2) # average across sequence
                hidden_states = _hidden_states.clone() if not token_idx else hidden_states + _hidden_states
            hidden_states = hidden_states.mean(1) # average across batch

            # for each specified module
            for module_name in modules_to_mix:

                keyword = f"{module_name}{EXPERTS_NAME}"
                if "." in keyword: keyword = keyword[:keyword.find(".")]
                keyword += ROUTERS_NAME
                matched_keys = [x for x in routers_keys if keyword in x]
                #NOTE: assume `routers_keys` are layer ordered

                for layer_idx, key in enumerate(matched_keys):

                    logging.info("Replacing {}[{}] using hidden states computed.".format(key, expert_idx))
                    router_weight =  hidden_states[layer_idx]
                    router_weight /= router_weight.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                    moe_sd[key][expert_idx] += router_weight.cpu()

                    # for record
                    if expert_idx == len(ingredients)-1:
                        routers_keys.remove(key)

    del model_base
    del sd_base
    gc.collect()
    
    ## loading each ingredient models and and copy the weights to respectivce experts
    # all `experts_keys` should be overwritten with weightsafter this loop
    for expert_idx, path in enumerate(ingredients):

        logging.info("loading expert {} from {}...".format(expert_idx, path))
        ingred_model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=moe_model.config.torch_dtype
        )
        ingred_sd = ingred_model.state_dict()

        # for each specified module
        for module_name in modules_to_mix:

            keyword = f"{module_name}{EXPERTS_NAME}{expert_idx}"
            matched_keys = [x for x in experts_keys if keyword in x]
            assert matched_keys, keyword

            # for each matched experts weight
            for key in matched_keys:
                key_cand = key.replace(keyword, module_name)

                logging.info("copying {} from expert {} to MOE {}...".format(key_cand, expert_idx, key))
                moe_sd[key].copy_(ingred_sd[key_cand])

                # for record
                experts_keys.remove(key)
        
        del ingred_model
        del ingred_sd
        gc.collect()

    # END CHECK
    # ensure no weights are left empty/uncopied
    assert len(experts_keys) == 0, "Cannot match {}".format(experts_keys)

    if len(positive_tokens): assert len(routers_keys) == 0, "Cannot match {}".format(routers_keys)
    # /END CHECK

    # parameters
    model_info = {
        "stem_param": stem_param,
        "experts_param": experts_param,
        "routers_param": routers_param,
        "total_param": stem_param + experts_param + routers_param,
        "active_param": stem_param + routers_param + int(experts_param/len(ingredients)*num_experts_per_tok)
    }
    logging.info("Stem parameters: {}".format(model_info["stem_param"]))
    logging.info("Experts parameters: {}".format(model_info["experts_param"]))
    logging.info("Routers parameters: {}".format(model_info["routers_param"]))
    logging.info("MOE total parameters (numel): {}".format(
        sum(p.numel() for p in moe_model.parameters())))
    logging.info("MOE total parameters : {}".format(model_info["total_param"]))
    logging.info("MOE active parameters: {}".format(model_info["active_param"]))

    return moe_model.to(torch.float16), model_info


if __name__ == "__main__":

    import argparse, os, inspect
    from transformers import AutoTokenizer
    
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ingredients', nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--always_on_idx', type=int, default=-1)
    parser.add_argument('--modules', nargs='+', default=["mlp"])
    parser.add_argument('--positive_tokens', nargs='+', default=[])
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_experts_per_tok', type=int, default=2)
    parser.add_argument('--deep_router', action="store_true", default=False)
    args = parser.parse_args()
    
    model, model_info = mix(
        args.model_path,
        args.ingredients,
        args.modules,
        args.positive_tokens,
        args.num_samples,
        deep_router=args.deep_router,
        num_experts_per_tok=args.num_experts_per_tok
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.save_pretrained(args.output_dir)
    
    import json
    with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f)