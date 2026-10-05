import math
import torch
from torch import nn
from torch.nn import functional as F

class SinusoidalPosition(nn.Module):
    def __init__(self,width,max_length=2048):
        super().__init__(); pos=torch.arange(max_length).float().unsqueeze(1); div=torch.exp(torch.arange(0,width,2).float()*(-math.log(10000.)/width)); pe=torch.zeros(max_length,width); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div); self.register_buffer("pe",pe)
    def forward(self,x):
        if x.shape[1]>len(self.pe): raise ValueError("sequence exceeds positional encoding capacity")
        return x+self.pe[:x.shape[1]]

class TextMotionModel(nn.Module):
    def __init__(self,text_dim,motion_dim,latent=10,width=128):
        super().__init__(); self.text=nn.Sequential(nn.Linear(text_dim,width),nn.GELU(),nn.Linear(width,latent)); self.motion_in=nn.Linear(motion_dim,width); self.position=SinusoidalPosition(width)
        layer=nn.TransformerEncoderLayer(width,4,256,batch_first=True,activation="gelu"); self.motion=nn.TransformerEncoder(layer,2); self.motion_out=nn.Linear(width,latent)
    def forward(self,text,motion):
        if motion.ndim!=3 or text.ndim!=2 or len(text)!=len(motion): raise ValueError("expected paired text [B,D] and motion [B,F,D]")
        return F.normalize(self.text(text),dim=-1),F.normalize(self.motion_out(self.motion(self.position(self.motion_in(motion))).mean(1)),dim=-1)

def contrastive(text,motion,temperature=.07):
    logits=text@motion.T/temperature; labels=torch.arange(len(text),device=text.device)
    return (F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
