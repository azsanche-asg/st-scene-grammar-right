import os, csv, json, time
from importlib import import_module
from typing import Dict, Any
from utils.misc import ensure_dir, set_seed
from utils.config import cfg_hash
from metrics.core import compression_ratio, purity, ade_fde, repeat_accuracy


def build_backbone(cfg):
    name = cfg['backbone']['name']
    module = import_module(f'backbones.{name}')
    cls = getattr(module, f'{name.capitalize()}Backbone')
    return cls(**cfg['backbone'].get('params', {}))


def build_policy(cfg):
    name = cfg['policy']['name']
    module = import_module(f'policy.{name}')
    cls = getattr(module, f'{name.capitalize()}Policy')
    return cls(**cfg['policy'].get('params', {}))

def run_experiment(cfg: Dict[str, Any], out_dir: str):
    set_seed(cfg.get('seed', 42))
    ensure_dir(out_dir)
    # Data
    data_cfg = cfg.get('data', {})
    data_name = data_cfg.get('name', 'synthetic')
    data_params = data_cfg.get('params', {})

    from datasets.synthetic import gen_sequence

    def _gen_synth(custom_params=None):
        params = custom_params or {}
        return gen_sequence(**params, save_dir=None)

    seq = None
    if data_name == 'synthetic':
        seq = _gen_synth(data_params)
    elif data_name == 're10k':
        root = data_params.get('root')
        if not root or not os.path.isdir(root):
            print(f"Warning: dataset root '{root}' not found; falling back to synthetic data.")
            seq = _gen_synth()
        else:
            from datasets.real_stubs import RE10KDataset

            dataset = RE10KDataset(**data_params)
            seq = next(iter(dataset), None)
            if seq is None:
                print("Warning: RE10K dataset produced no sequences; falling back to synthetic data.")
                seq = _gen_synth()
    elif data_name == 'scannetpp':
        root = data_params.get('root')
        if not root or not os.path.isdir(root):
            print(f"Warning: dataset root '{root}' not found; falling back to synthetic data.")
            seq = _gen_synth()
        else:
            from datasets.real_stubs import ScanNetPPDataset

            dataset = ScanNetPPDataset(**data_params)
            seq = next(iter(dataset), None)
            if seq is None:
                print("Warning: ScanNetPP dataset produced no sequences; falling back to synthetic data.")
                seq = _gen_synth()
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    if seq is None:
        seq = _gen_synth()
    # Backbone
    backbone = build_backbone(cfg)
    # Policy
    policy = build_policy(cfg)
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
