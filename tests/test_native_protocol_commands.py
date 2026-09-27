import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NativeProtocolCommandsTest(unittest.TestCase):
    def command(self, suite, group):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, '-m', 'sage.reproduce_native',
                '--suite', suite, '--data-root', directory, '--out', directory,
                '--methods', 'sage', '--groups', str(group), '--dry-run']
            if suite == 'robotwin_a2b':
                command += ['--robotwin-root', directory, '--horizons', '30']
            else:
                command += ['--libero-root', directory, '--full-episode']
            output = subprocess.check_output(command,
                cwd=Path(__file__).resolve().parents[1], text=True)
        return json.loads(output)

    def test_libero_uses_historical_planner_seeds_not_group_ids(self):
        for suite, seed in [('libero_scene2', 20260829), ('libero_caddy', 42)]:
            for group in range(3):
                with self.subTest(suite=suite, group=group):
                    command = self.command(suite, group)
                    self.assertEqual(command[command.index('--seed') + 1], str(seed))
                    self.assertEqual(command[command.index('--query-start') + 1], str(50 * group))

    def test_robotwin_keeps_group_seeds_and_recycles_processes(self):
        for group in range(3):
            command = self.command('robotwin_a2b', group)
            self.assertEqual(command[command.index('--seed') + 1], str(42 + group))
            self.assertEqual(command[command.index('--max-new-queries') + 1], '8')


if __name__ == '__main__':
    unittest.main()
