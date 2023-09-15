import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation


class MultiModalSegformer(SegformerForSemanticSegmentation):
    def __init__(self, config):
        super().__init__(config)
        
        decoder_hidden_size = int(config.decoder_hidden_size)
        
        self.sen_encoder = nn.Sequential(
            nn.Conv3d(10, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((decoder_hidden_size // 32, 16, 16))
        )
        
        self.aerial_sen_norm = nn.LayerNorm(decoder_hidden_size)
         
    def forward(
        self,
        aerial: torch.FloatTensor,
        sen: torch.FloatTensor,
    ):
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
        
        last_encoder_hidden_state = output_sen + encoder_hidden_states[-1]
        last_encoder_hidden_state = last_encoder_hidden_state.permute(0, 2, 3, 1)
        last_encoder_hidden_state = self.aerial_sen_norm(last_encoder_hidden_state)
        encoder_hidden_states[-1] = last_encoder_hidden_state.permute(0, 3, 1, 2)
        
        logits = self.decode_head(encoder_hidden_states)
        
        return nn.functional.interpolate(
            input=logits,
            size=aerial.shape[-2:],
            mode='bilinear',
            align_corners=False
        )