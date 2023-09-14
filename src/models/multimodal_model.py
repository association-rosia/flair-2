import torch
from torch import nn
from typing import Optional, Union
from transformers import SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput


class MultiModalSegformer(SegformerForSemanticSegmentation):
    def __init__(self, config):
        super().__init__(config)
        
        self.sen_encoder = nn.Sequential(
            nn.LazyConv3d(16, kernel_size=3, padding=1),
            nn.LazyBatchNorm3d(),
            nn.LazyConv3d(16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.LazyConv3d(32, kernel_size=3, padding=1),
            nn.LazyBatchNorm3d(),
            nn.LazyConv3d(32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d((16, 16, 16))
        )
        
        self.aerial_sen_norm = nn.LayerNorm((512, 16, 16), eps=1e-05, elementwise_affine=True)
         
    def forward(
        self,
        aerial: torch.FloatTensor,
        sen: torch.FloatTensor,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            aerial,
            output_attentions=None,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )
        
        output_sen = self.sen_encoder(sen)
        output_sen = output_sen.flatten(start_dim=1, end_dim=2)
        
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        
        encoder_hidden_states = list(encoder_hidden_states)
        encoder_hidden_states[-1] += output_sen
        encoder_hidden_states[-1] = self.aerial_sen_norm(encoder_hidden_states[-1])
        
        logits = self.decode_head(encoder_hidden_states)
        
        return nn.functional.interpolate(
            input=logits,
            size=aerial.shape[-2:],
            mode='bilinear',
            align_corners=False
        )