import os, csv, json, time
from typing import Dict, Any
from ..utils.misc import ensure_dir, set_seed
from ..utils.config import cfg_hash
from ..datasets.synthetic import gen_sequence
from ..backbones.identity import IdentityBackbone
from ..policy.heuristic import HeuristicPolicy
from ..metrics.core import compression_ratio, purity, ade_fde, repeat_accuracy

def run_experiment(cfg: Dict[str, Any], out_dir: str):
    set_seed(cfg.get('seed', 42))
    ensure_dir(out_dir)
    # Data
    if cfg['data']['name'] == 'synthetic':
        seq = gen_sequence(**cfg['data'].get('params', {}), save_dir=None)
    else:
        raise NotImplementedError("Real datasets are stubbed; use data.name=synthetic for now.")
    # Backbone (placeholder)
    _ = IdentityBackbone()
    # Policy
    policy = HeuristicPolicy(iou_thresh=cfg['policy'].get('iou_thresh', 0.3))
    grammar = policy.induce(seq)
    # Metrics
    cr = compression_ratio(grammar, seq_len=len(seq['frames']))
    pur = purity(grammar)
    ade, fde = ade_fde([p['track'] for p in grammar['parts']], [p['track'] for p in grammar['parts']])
    rep_acc = repeat_accuracy(grammar, gt_regular=seq['gt']['regular'])
    # Save
    tag = cfg_hash(cfg)
    with open(os.path.join(out_dir, f"grammar_{tag}.json"), "w") as f:
        json.dump(grammar, f, indent=2)
    csv_path = os.path.join(out_dir, "results.csv")
    header = ["tag","cr","purity","ade","fde","rep_acc","time"]
    row = [tag, cr, pur, ade, fde, rep_acc, time.time()]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)
    return {"tag": tag, "cr": cr, "purity": pur, "ade": ade, "fde": fde, "rep_acc": rep_acc}
