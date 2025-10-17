
import os, argparse, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from dataset import DeepSonarDataset
from backbone import SRBackbone
from behavior import pack_features
from model import Detector
from tqdm import tqdm

class ARGS:
    def __init__(self, real_dir, fake_dir, ckpt_dir, epochs = 20, lr = 0.001, max_len = 4196):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.ckpt_dir = ckpt_dir
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len


def train(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = SRBackbone(device=device)
    ds = DeepSonarDataset(args.fake_dir, args.real_dir, bb, device=device)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    x0,y0 = next(iter(dl))
    feats = bb.forward_with_hooks(x0.to(device))
    fvec = pack_features(feats).to(device)
    in_dim = fvec.numel()
    clf = Detector(in_dim).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    best = 0.0
    
    for epoch in range(args.epochs):
        clf.train(); tot=0; corr=0; n=0
        for x,y in tqdm(dl):
            x=x[0].to(device); y=y.to(device)
            # x = x.squeeze()
            # feats = bb.forward_with_hooks(x)
            f = pack_features(feats)
            if f is None: continue
            f = f.view(1,-1).to(device)
            logits = clf(f)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            pred = logits.argmax(-1)
            corr += (pred==y).sum().item(); n+=y.size(0); tot += loss.item()
        acc = corr/max(n,1)
        torch.save({"model":clf.state_dict(),"in_dim":in_dim}, os.path.join(args.ckpt_dir,"last.pt"))
        if acc>best:
            best=acc
            torch.save({"model":clf.state_dict(),"in_dim":in_dim}, os.path.join(args.ckpt_dir,"best.pt"))
        print(f"epoch {epoch} acc {acc:.4f} loss {tot/max(n,1):.4f}")
    print("best",best)

if __name__=="__main__":
    # ap=argparse.ArgumentParser()
    # ap.add_argument("--data_dir",type=str,default="data")
    # ap.add_argument("--index_csv",type=str,required=True)
    # ap.add_argument("--ckpt_dir",type=str,default="ckpts")
    # ap.add_argument("--epochs",type=int,default=5)
    # ap.add_argument("--lr",type=float,default=2e-4)
    # ap.add_argument("--max_len",type=float,default=10.0)
    # args=ap.parse_args()

    fake_dir = "/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Dataset/inTheWildAudioDeekfake/fake"
    real_dir = "/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Dataset/inTheWildAudioDeekfake/real"
    ckpt_dir = "/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Code/ckpt"
    epochs = 200
    lr = 0.001
    max_len = 4196
    args = ARGS(real_dir=real_dir, fake_dir=fake_dir, ckpt_dir=ckpt_dir, epochs=epochs, lr=lr, max_len=max_len)
    train(args)
