import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('core_summary', ROOT / 'scripts/summarize_results.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_summary_rejects_incomplete_or_inconsistent_cells(tmp_path):
    path = tmp_path / 'results.json'
    result = dict(protocol_kind='paper', benchmark='pusht', method='sage', seed=32, horizon=25,
                  metrics=dict(episode_successes=[True] * 35 + [False] * 15, success_rate=70.))
    path.write_text(json.dumps(result))
    assert module.load_result(path, 'pusht', 'sage', 32, 25) == 70.
    result['metrics']['success_rate'] = 90.
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='episode outcomes'):
        module.load_result(path, 'pusht', 'sage', 32, 25)
    result['metrics']['episode_successes'].pop()
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='50 boolean'):
        module.load_result(path, 'pusht', 'sage', 32, 25)


def test_dino_references_are_not_lewm_results():
    pusht = json.loads((ROOT / 'configs/paper_dinowm_k128_cem6.json').read_text())
    cube = json.loads((ROOT / 'configs/paper_dinowm_cube_k128_cem6.json').read_text())
    assert set(pusht['expected_success_percent']) == {'pusht'}
    assert set(cube['expected_success_percent']) == {'cube'}
    assert pusht['expected_success_percent']['pusht']['sage'][0] == 70.
    assert cube['expected_success_percent']['cube']['sage'][0] == 100.
