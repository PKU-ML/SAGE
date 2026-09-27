"""Aggregate complete locked native cells; never count runtime errors as failures."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from sage.provenance import sha256_file


def validate_group(payload, queries, *, robotwin, method, group, manifest_sha, suite=None):
    protocol = payload['protocol']
    expected_method = 'gaussian_lewm' if robotwin and method == 'base' else method
    if protocol.get('controller' if robotwin else 'method') != expected_method:
        raise ValueError('Result method mismatch')
    expected_seed = 42 + group if robotwin else {'libero_scene2': 20260829, 'libero_caddy': 42}[suite]
    if not robotwin and protocol.get('suite') != suite:
        raise ValueError('Result suite mismatch')
    if protocol.get('seed') != expected_seed:
        raise ValueError('Result sampling seed mismatch')
    if protocol.get('query_start') != group * 50 or protocol.get('num_queries') != 50:
        raise ValueError('Result query slice mismatch')
    if (protocol if robotwin else payload).get('query_manifest_sha256') != manifest_sha:
        raise ValueError('Result manifest mismatch')
    expected = {'chunk_raw_actions': 15, 'recovery_raw_actions': 120, 'cem_rounds': 1} if robotwin else {
        'commitment': 15, 'recovery': 120, 'rank_rounds': 1, 'K': 64}
    if robotwin:
        expected['num_candidates'] = 1 if method == 'prior_top' else 64
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError('Result planner protocol mismatch')
    rows = payload['results']
    key = 'query_id' if robotwin else 'query_index'
    if len(rows) != 50 or len(queries) != 50:
        raise ValueError('Incomplete result: require 50 queries per group')
    if [row.get(key) for row in rows] != [q[key] for q in queries]:
        raise ValueError('Missing, duplicate, or reordered query IDs')
    if any('error' in row for row in rows):
        raise ValueError('Runtime errors are not task failures')
    success_key = 'official_success' if robotwin else 'official_success_ever'
    if any(type(row.get(success_key)) is not bool for row in rows):
        raise ValueError('Missing or invalid official success flag')
    return sum(row[success_key] for row in rows)


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True, choices=('libero_scene2', 'libero_caddy', 'robotwin_a2b'))
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--methods', nargs='+', choices=('base', 'prior_top', 'sage'), default=['base', 'prior_top', 'sage'])
    p.add_argument('--horizons', nargs='+', type=int, choices=(30, 60, 90, 120), default=[30, 60, 90, 120])
    p.add_argument('--full-episode', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    robotwin = args.suite == 'robotwin_a2b'
    if robotwin and args.full_episode:
        p.error('RoboTwin A2B has no paper full-episode cell')
    manifest = root/'data/manifests'/args.suite/('full.json' if args.full_episode else 'tail.json')
    queries = json.loads(manifest.read_text())['queries']
    manifest_sha = sha256_file(manifest)
    suites = json.loads((root/'configs/suites.json').read_text())
    assets = json.loads((root/'configs/assets.json').read_text())
    suite = suites[args.suite]
    summary = []
    for method in args.methods:
        for horizon in ([None] if args.full_episode else args.horizons):
            label = 'full' if horizon is None else f'h{horizon}'
            chosen = [q for q in queries if horizon is None or q.get('horizon_raw_actions', q.get('horizon')) == horizon]
            successes = []
            for group in range(3):
                path = args.results/method/label/f'group{group}.json'
                payload = json.loads(path.read_text())
                identities = payload['protocol']['input_sha256'] if robotwin else payload['checkpoints_sha256']
                for component, filename in (('world_model', 'world_model/weights.pt'), ('generator', 'generator.pt'), ('prior', 'prior.pt')):
                    if identities.get(component if robotwin else filename) != assets[suite[component]]['sha256']:
                        raise ValueError(f'{path}: wrong {component} checkpoint')
                successes.append(validate_group(payload, chosen[group*50:(group+1)*50],
                    robotwin=robotwin, method=method, group=group, manifest_sha=manifest_sha, suite=args.suite))
            percentages = [100*s/50 for s in successes]
            summary.append({'suite': args.suite, 'method': method, 'horizon': label,
                'successes': sum(successes), 'total': 150, 'success_percent': 100*sum(successes)/150,
                'group0': successes[0], 'group1': successes[1], 'group2': successes[2],
                'group_std_percent': statistics.stdev(percentages)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
