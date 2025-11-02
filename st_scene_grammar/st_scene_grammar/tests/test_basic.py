from st_scene_grammar.eval.runner import run_experiment
from st_scene_grammar.utils.config import load_config
def test_synthetic_runs():
    cfg = load_config('st_scene_grammar/configs/synthetic_demo.yaml')
    res = run_experiment(cfg, cfg['save']['out_dir'])
    assert 'tag' in res and res['cr'] >= 0.0
    assert 0.0 <= res['purity'] <= 1.0
