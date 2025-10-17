
import os, argparse, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from dataset import WavIndex
from backbone import SRBackbone
from behavior import pack_features
from model import Detector
from sklearn.metrics import classification_report, roc_auc_score

def eval_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = WavIndex(args.index_csv, max_len=args.max_len, sr=16000)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    bb = SRBackbone(device=device)
    ckpt = torch.load(args.ckpt, map_location=device)
    in_dim = ckpt["in_dim"]
    clf = Detector(in_dim).to(device)
    clf.load_state_dict(ckpt["model"]); clf.eval()
    ys=[]; ps=[]
    with torch.no_grad():
        for x,y in dl:
            x=x[0].to(device)
            feats = bb.forward_with_hooks(x)
            f = pack_features(feats)
            if f is None: continue
            f = f.view(1,-1).to(device)
            logits = clf(f)
            prob = torch.softmax(logits, dim=-1)[:,1].item()
            ys.append(y.item()); ps.append(prob)
    yhat = [1 if p>=0.5 else 0 for p in ps]
    rep = classification_report(ys,yhat,output_dict=True)
    auc = roc_auc_score(ys, ps) if len(set(ys))>1 else float("nan")
    df = pd.DataFrame(rep).transpose()
    df["auc"]=auc
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    df.to_csv(args.report, index=True)
    print("AUC",auc)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--index_csv",type=str,required=True)
    ap.add_argument("--ckpt",type=str,required=True)
    ap.add_argument("--report",type=str,default="reports/eval.csv")
    ap.add_argument("--max_len",type=float,default=10.0)
    args=ap.parse_args()
    eval_run(args)
