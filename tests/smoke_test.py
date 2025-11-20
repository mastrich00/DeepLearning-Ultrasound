import torch
from src.ultra_spatial.models.generator import RetinexLowRankVT
from src.ultra_spatial.models.discriminator import PatchDiscriminator
def test():
    B,T,H,W=2,16,64,64
    G=RetinexLowRankVT(); x=torch.rand(B,T,1,H,W); out=G(x); assert out["corrected"].shape==(B,1,H,W)
    D=PatchDiscriminator(); d=D(out["corrected"]); assert d.ndim==4
    print("OK")
if __name__=="__main__": test()
