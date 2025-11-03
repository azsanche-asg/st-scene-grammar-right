from eval.runner import run_experiment
from utils.config import load_config
def test_synthetic_runs():
    cfg = load_config('configs/synthetic_demo.yaml')
    cfg['policy']['name'] = 'heuristic'
    cfg['policy']['params'] = {'iou_thresh': 0.35}
    res = run_experiment(cfg, cfg['save']['out_dir'])
    assert 'tag' in res and res['cr'] >= 0.0
    assert 0.0 <= res['purity'] <= 1.0


def test_learned_policy_runs():
    cfg = load_config('configs/synthetic_demo.yaml')
    cfg['policy']['name'] = 'learned'
    res = run_experiment(cfg, cfg['save']['out_dir'])
    assert 'tag' in res
