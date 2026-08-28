# Data Layout

SAGE does not redistribute the underlying PushT or OGBench trajectories.

Expected local inputs:

```text
PushT:       pusht_expert_train.lance
OGBench Cube: cube_single_expert.h5
```

The public split and evaluation files are:

```text
data/
  splits/
    pusht_episode_split_seed42.json
    ogbench_cube_single_split_seed42.json
  manifests/
    pusht/seed{32,42,52}/h{25,50,75,100,125,150}.json
    cube/seed{32,42,52}/h{25,50,75,100,125,150}.json
```

Splits are episode-disjoint. Evaluation manifests are immutable query sets,
not generated afresh by the evaluation scripts. This guarantees that changing
the order of requested seeds does not alter the records assigned to a seed.

To sanitize newly generated manifests:

```bash
python scripts/sanitize_manifests.py \
  --source /path/to/raw/manifests \
  --destination /path/to/sanitized/manifests
```

Always run `python scripts/audit_release.py` afterward.
