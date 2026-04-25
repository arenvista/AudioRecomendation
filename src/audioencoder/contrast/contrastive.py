import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoFeatureExtractor, ASTModel, CLIPModel, CLIPProcessor
from typing import Optional, cast, Any

class ASTCLIPAlignmentModel(nn.Module):
    def __init__(
        self,
        ast_model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        clip_model_id: str = "openai/clip-vit-base-patch32",
        shared_dim: int = 512,
    ):
        super().__init__()
        self.audio_extractor = AutoFeatureExtractor.from_pretrained(ast_model_id)
        self.ast = ASTModel.from_pretrained(ast_model_id) 
        
        self.clip = CLIPModel.from_pretrained(clip_model_id)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        
        self.audio_projection = nn.Linear(self.ast.config.hidden_size, shared_dim)
        
        initial_logit = float(np.log(1 / 0.07))
        self.logit_scale = nn.Parameter(torch.as_tensor(initial_logit))

    def forward(
        self,
        audio_tensors: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        image_tensors: Optional[torch.Tensor] = None,
    ):
        ast_outputs = self.ast(audio_tensors)
        raw_audio_embeds = cast(torch.Tensor, ast_outputs.pooler_output)
        audio_embeds = self.audio_projection(raw_audio_embeds)
        audio_embeds = F.normalize(audio_embeds, p=2, dim=-1)

        text_embeds, image_embeds = None, None

        if text_tokens is not None:
            text_outputs = cast(
                torch.Tensor, self.clip.get_text_features(input_ids=text_tokens)
            )
            text_embeds = F.normalize(text_outputs, p=2, dim=-1)

        if image_tensors is not None:
            image_outputs = cast(
                torch.Tensor,
                self.clip.get_image_features(pixel_values=cast(Any, image_tensors)),
            )
            image_embeds = F.normalize(image_outputs, p=2, dim=-1)

        return audio_embeds, text_embeds, image_embeds

    def compute_loss(self, audio_embeds: torch.Tensor, other_embeds: torch.Tensor) -> torch.Tensor:
        """
        Computes the symmetric contrastive loss between audio and text/image.
        """
        # Cosine similarity scaled by learned temperature
        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * audio_embeds @ other_embeds.t()
        logits_per_other = logits_per_audio.t()

        batch_size = audio_embeds.shape[0]
        # Labels are simply the diagonal (0, 1, 2, ..., batch_size - 1)
        labels = torch.arange(batch_size, device=audio_embeds.device)

        # Symmetric cross-entropy
        loss_audio = F.cross_entropy(logits_per_audio, labels)
        loss_other = F.cross_entropy(logits_per_other, labels)
        
        return (loss_audio + loss_other) / 2
