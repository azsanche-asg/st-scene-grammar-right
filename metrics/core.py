import numpy as np
from typing import Dict, Any, List, Tuple
def compression_ratio(grammar: Dict[str, Any], seq_len: int) -> float:
    parts = len(grammar.get('parts', []))
    rules = len(grammar.get('rules', []))
    denom = max(1, seq_len * 5)
    return (parts + rules) / denom
def purity(grammar: Dict[str, Any]) -> float:
    tracks = [p['track'] for p in grammar.get('parts', [])]
    if not tracks: return 0.0
    ok = 0
    for tr in tracks:
        sizes_w = [w for (_,_,w,h) in tr]
        sizes_h = [h for (_,_,w,h) in tr]
        ok += int(np.std(sizes_w) < 1e-3 and np.std(sizes_h) < 1e-3)
    return ok / max(1,len(tracks))
def ade_fde(tracks_gt: List[List[Tuple[int,int,int,int]]], tracks_pred: List[List[Tuple[int,int,int,int]]]):
    def center(r): x,y,w,h = r; return (x+w/2, y+h/2)
    if not tracks_gt or not tracks_pred: return 0.0, 0.0
    T = min(len(tracks_pred[0]), len(tracks_gt[0]))
    if T == 0: return 0.0, 0.0
    N = min(len(tracks_gt), len(tracks_pred))
    ade, fde = 0.0, 0.0
    for i in range(N):
        gt = tracks_gt[i][:T]; pr = tracks_pred[i][:T]
        errs = [np.linalg.norm(np.array(center(a))-np.array(center(b))) for a,b in zip(gt,pr)]
        ade += float(np.mean(errs)); fde += float(errs[-1])
    ade /= max(1,N); fde /= max(1,N)
    return ade, fde
def repeat_accuracy(grammar: Dict[str, Any], gt_regular: bool) -> float:
    rules = grammar.get('rules', [])
    has_rep = any(r.get('type')=='repeat' for r in rules)
    approx = any(r.get('type')=='repeat' and r.get('approx_equal_spacing', False) for r in rules)
    return float((gt_regular and approx) or ((not gt_regular) and (not has_rep)))
