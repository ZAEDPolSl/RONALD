"""Finalize a completed cache after grouping duplicate scans across frozen splits."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from .prepare import validate_splits
from .preparation_state import PreparationState, preparation_lock, atomic_json


def group_splits(cases):
    parent = list(range(len(cases)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, case in enumerate(cases):
        for field in ('patient_id', 'image', 'image_sha256'):
            if not case.get(field):
                raise ValueError(f'Missing {field} in completed case')
            key = field, case[field]
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, case in enumerate(cases):
        groups[root(i)].append(case)
    priority = {'training': 0, 'validation': 1, 'test': 2}
    splits = {key: [] for key in priority}
    changes = []
    duplicates = []
    for group in groups.values():
        target = max((c['split'] for c in group), key=priority.__getitem__)
        if len(group) > 1:
            duplicates.append({'case_ids': sorted(c['case_id'] for c in group), 'split': target})
        for case in group:
            if case['split'] != target:
                changes.append({'case_id': case['case_id'], 'from': case['split'], 'to': target})
            splits[target].append(dict(case, split=target))
    for cases_in_split in splits.values():
        cases_in_split.sort(key=lambda c: c['case_id'])
    validate_splits(splits)
    return splits, changes, duplicates


def deduplicate_splits(splits):
    """Keep lowest case_id per native voxel hash, after all identity links are grouped."""
    result, removed = {}, []
    for split, cases in splits.items():
        retained = {}
        for case in sorted(cases, key=lambda c: c['case_id']):
            key = case['image_sha256']
            if key in retained:
                removed.append({'split': split, 'image_sha256': key,
                                'retained_case_id': retained[key]['case_id'],
                                'removed_case_id': case['case_id'],
                                'retained_dataset': retained[key]['dataset'],
                                'removed_dataset': case['dataset']})
            else:
                retained[key] = case
        result[split] = list(retained.values())
    validate_splits(result)
    return result, removed


def finalize(cache, accept_timestamps=False, deduplicate=False):
    cache = Path(cache).resolve()
    with preparation_lock(cache):
        saved = json.loads((cache / 'prepare_state.json').read_text())
        settings = saved['identity']['settings']
        args = argparse.Namespace(**settings, output=str(cache), resume=True,
                                  adopt_legacy_cache=False,
                                  accept_cache_timestamp_changes=accept_timestamps)
        state = PreparationState(cache, args)
        with open(settings['inventory'], newline='') as f:
            rows = list(csv.DictReader(f))
        cases, marker_hashes = [], {}
        from .preparation_state import sha256_file
        for row in rows:
            case_id = row['case_id']
            if Path(case_id).name != case_id:
                raise ValueError('Invalid case ID')
            marker = cache / case_id / 'complete.json'
            record = json.loads(marker.read_text())
            checked = state.completed(case_id, record['sources'])
            case = checked['case']
            if any(case[k] != row[k] for k in ('case_id', 'patient_id', 'dataset', 'split')):
                raise ValueError('Completed case disagrees with frozen inventory')
            cases.append(case)
            marker_hashes[case_id] = sha256_file(marker)
        if len({c['case_id'] for c in cases}) != len(cases):
            raise ValueError('Duplicate case IDs')
        splits, changes, groups = group_splits(cases)
        removed = []
        if deduplicate:
            splits, removed = deduplicate_splits(splits)
        report = {'policy': 'transitive patient/image/hash groups; test > validation > training',
                  'changes': changes, 'linked_groups': groups,
                  'counts': {k: dict(Counter(c['dataset'] for c in v)) for k, v in splits.items()},
                  'unique_images': {k: len({c['image_sha256'] for c in v}) for k, v in splits.items()},
                  'completion_sha256': marker_hashes,
                  'limitations': 'Original sources not reread; uses completed cache provenance. Duplicate copies retained within one split; not independent observations.'}
        if deduplicate:
            report['deduplication'] = {'policy': 'lowest case_id per image_sha256 after transitive grouping', 'removed': removed}
            report['limitations'] = 'Original sources not reread; native voxel hash identity only. Biological patient aliases remain unverified. Cached arrays are preserved.'
        manifest = {'version': 1, 'classes': {'0': 'background', '1': 'airway_with_wall', '2': 'vessel'},
                    'preprocessing': {'spacing_xyz': settings['spacing'], 'margin_mm': settings['margin_mm'], 'hu': settings['hu']},
                    'preparation': dict(settings, output=str(cache)), 'splits': splits,
                    'split_policy': report['policy']}
        if deduplicate:
            manifest['deduplication_policy'] = report['deduplication']['policy']
        target = cache / ('manifest-deduplicated.json' if deduplicate else 'manifest-grouped.json')
        if target.exists() and json.loads(target.read_text()) != manifest:
            raise ValueError('Existing grouped manifest differs; preserve it and investigate')
        atomic_json(cache / ('split-deduplication-audit.json' if deduplicate else 'split-repair-audit.json'), report)
        if not target.exists():
            atomic_json(target, manifest)
        print(json.dumps({'manifest': str(target), 'counts': report['counts'],
                          'unique_images': report['unique_images'], 'moved_cases': len(changes), 'removed_duplicate_copies': len(removed)}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--accept-cache-timestamp-changes', action='store_true')
    p.add_argument('--deduplicate', action='store_true', help='Write a separate manifest retaining one copy per native voxel hash')
    args = p.parse_args()
    finalize(args.cache, args.accept_cache_timestamp_changes, args.deduplicate)


if __name__ == '__main__':
    main()
