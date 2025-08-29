import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys

# Import the PCT model components
from PCT.networks.cls.pct0 import  Point_Transformer_lstm


class SimpleSkeletonModel(nn.Module):
    
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.transformer = Point_Transformer_lstm()
    
    def execute(self, vertices: jt.Var,normal):
        x = self.transformer(vertices, normal)
        return x
    
class SimpleSkeletonModel4096(nn.Module):
    
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.transformer = Point_Transformer_lstm4096()
    
    def execute(self, vertices: jt.Var,normal):
        x = self.transformer(vertices, normal)
        return x
# Factory function to create models
def create_model(model_name='pct', **kwargs):
    if model_name == "pct":
        return SimpleSkeletonModel(feat_dim=256)
    raise NotImplementedError()
