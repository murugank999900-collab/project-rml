# Frozen splits

Each JSON file records a dataset split as sample indices only (no samples).
Once committed, a split file must never be edited or regenerated in place.

`rml2016.10a_seed42.json` (not yet created) is produced on a machine with the
dataset by:

```bash
python scripts/prepare_rml2016.py --write-split
```

Format: per (modulation, SNR) group, the within-group `val` and `test` indices;
`train` is the complement. Global index = group offset (groups sorted by
modulation, then SNR) + within-group index. `sha256` fields hash the global
index arrays as little-endian int64 and are checked on load.
