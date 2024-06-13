import transformers
from transformers import MistralConfig
from transformers.utils import logging

from packaging import version

logger = logging.get_logger(__name__)

class MoeMistralConfig(MistralConfig):

    model_type = "moemistral"

    def __init__(
        self,
        moe_mlp=True,
        moe_query=False,
        moe_key=False,
        moe_value=False,
        num_experts_per_tok=2,
        num_local_experts=4,
        output_router_logits=False,
        router_aux_loss_coef=0.01,
        always_on_idx=-1,
        deep_router=False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        assert version.parse(transformers.__version__) >= version.parse('4.40.0'), \
            "MoeMistral is implemented for transformers>=4.40.0!"
        self.moe_mlp = moe_mlp
        self.moe_query = moe_query
        self.moe_key = moe_key
        self.moe_value = moe_value
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.always_on_idx = always_on_idx
        self.deep_router = deep_router