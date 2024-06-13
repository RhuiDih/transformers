import logging, traceback
import torch

from transformers import MoeMistralConfig, MoeMistralForCausalLM
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO)

dtype = torch.float16
simple_arch = dict(
    hidden_size=128,
    intermediate_size=1024,
    num_hidden_layers=4,
    num_attention_heads=8,
    torch_dtype=dtype
)

configs = [
    # (name, cpu_ok, config)
    ("Basic Mistral (no mixture)", True, 
        MoeMistralConfig(
            moe_mlp=False,
            output_router_logits=False,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MoeMistral with MLP MOE", True,
        MoeMistralConfig(
            num_local_experts=4,
            output_router_logits=True,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MoeMistral with MLP MOE always-on expert", True,
        MoeMistralConfig(
            num_experts_per_tok=2,
            num_local_experts=4,
            always_on_idx=0,
            output_router_logits=True,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MoeMistral with QUERY MOE", True,
        MoeMistralConfig(
            num_experts_per_tok=2,
            num_local_experts=4,
            moe_query=True,
            output_router_logits=True,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MoeMistral with ALL MOE", True,
        MoeMistralConfig(
            num_experts_per_tok=2,
            num_local_experts=4,
            moe_query=True,
            moe_key=True,
            moe_value=True,
            output_router_logits=True,
            pretraining_tp=False,
            **simple_arch
        )),
    ("MoeMistral with ALL MOE with flash attn", False,
        MoeMistralConfig(
            num_experts_per_tok=2,
            num_local_experts=4,
            moe_query=True,
            moe_key=True,
            moe_value=True,
            output_router_logits=True,
            pretraining_tp=False,
            _attn_implementation="flash_attention_2",
            **simple_arch
        )),
]

def simpleforward(cls, dtype, device):
    dummy = torch.ones((1,8)).long().to(device)
    model = cls(config).to(dtype).to(device)
    o = model(dummy)
    o = model.generate(dummy)
    optimizer = AdamW(model.parameters())
    out = model(
        input_ids=dummy, labels=dummy
    )
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad()

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    results = []
    max_mems = []

    for idx, (name, cpu_ok, config) in enumerate(configs):

        try :
            logging.info("### TEST [{}] {}".format(idx, name))
            if not torch.cuda.is_available() and not cpu_ok:
                results.append("This test requires GPU")
                logging.info("This test requires GPU, skipping...")
                continue
            simpleforward(MoeMistralForCausalLM, dtype, device)
            logging.info("### TEST [{}] Passed!".format(idx))
            results.append("Pass")
    
        except Exception as e:

            logging.info("### TEST [{}] Failed!".format(idx))
            traceback.print_exc()
            results.append("Fail: {}".format(str(e)))

        if torch.cuda.is_available():
            max_mems.append(torch.cuda.max_memory_allocated()/1024/1024)
            torch.cuda.reset_peak_memory_stats()
        else:
            max_mems.append("null")
        
    print("\n\n\n")
    for idx, (name, cpu_ok, config) in enumerate(configs):
        report = "TEST [{}]\tName: {}\tStatus: {}".format(idx, name, results[idx])
        if torch.cuda.is_available():
            report += "\tMaxMem: {}".format(max_mems[idx])
        logging.info(report)