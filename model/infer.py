
import argparse, torch, soundfile as sf, torchaudio
from backbone import SRBackbone
from behavior import pack_features
from model import Detector

def load_audio(path, target_sr=16000):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim>1: x=x.mean(-1)
    if sr!=target_sr:
        x = torchaudio.functional.resample(torch.tensor(x), sr, target_sr).numpy()
    return torch.tensor(x)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",type=str,required=True)
    ap.add_argument("--wav",type=str,required=True)
    args=ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = SRBackbone(device=device)
    ckpt = torch.load(args.ckpt, map_location=device)
    in_dim = ckpt["in_dim"]
    clf = Detector(in_dim).to(device); clf.load_state_dict(ckpt["model"]); clf.eval()
    x = load_audio(args.wav, 16000)
    T = int(10.0*16000)
    if x.numel()>T: x=x[:T]
    else:
        pad = T-x.numel()
        if pad>0: x = torch.nn.functional.pad(x,(0,pad))
    with torch.no_grad():
        feats = bb.forward_with_hooks(x.to(device))
        f = pack_features(feats).view(1,-1).to(device)
        logits = clf(f)
        prob = torch.softmax(logits, dim=-1)[0,1].item()
    print("fake_prob",prob)
