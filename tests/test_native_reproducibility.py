import json
from pathlib import Path
import subprocess
import sys
import unittest

from sage.runtime.eval_resume import validate_resume

ROOT = Path(__file__).resolve().parents[1]


class NativeReproducibilityTests(unittest.TestCase):
    def test_robotwin_group_seeds(self):
        result = subprocess.run([sys.executable, '-m', 'sage.reproduce_native',
            '--suite', 'robotwin_a2b', '--robotwin-root', 'runtime-root',
            '--data-root', 'data-root', '--out', 'test-dry-run-unused',
            '--methods', 'sage', '--horizons', '30', '--dry-run'],
            cwd=ROOT, text=True, capture_output=True, check=True)
        commands = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([c[c.index('--seed')+1] for c in commands], ['42', '43', '44'])
        self.assertEqual([c[c.index('--query-start')+1] for c in commands], ['0', '50', '100'])

    def test_resume_requires_rng(self):
        with self.assertRaisesRegex(ValueError, 'RNG'):
            validate_resume({'protocol': {}, 'results': [{'query_id': 'a'}]}, {}, ['a', 'b'])

    def test_resume_rejects_changed_inputs(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            validate_resume({'protocol': {'sha': 'old'}}, {'sha': 'new'}, [])

    def test_resume_rejects_non_prefix_and_errors(self):
        for rows in ([{'query_id': 'b'}], [{'query_id': 'a', 'error': 'bad replay'}]):
            with self.assertRaises(ValueError):
                validate_resume({'protocol': {}, 'results': rows}, {}, ['a', 'b'])

    def test_resume_accepts_recorded_state(self):
        saved = {'protocol': {}, 'results': [{'query_id': 'a'}], 'proposal_rng_state': [0, 1, 255]}
        self.assertEqual(validate_resume(saved, {}, ['a', 'b']), [0, 1, 255])


if __name__ == '__main__':
    unittest.main()
