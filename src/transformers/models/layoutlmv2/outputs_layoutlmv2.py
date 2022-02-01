""" PyTorch Relation Extraction LayoutLMv2 outputs """
from ...file_utils import ModelOutput
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch


@dataclass
class RegionExtractionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    entities: Optional[Dict] = None
    relations: Optional[Dict] = None
    pred_relations: Optional[Dict] = None
