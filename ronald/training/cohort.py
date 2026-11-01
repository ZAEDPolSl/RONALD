"""Freeze DUKE, LUNA16 and Moltest 1 inventories with patient-grouped splits."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import logging
from pathlib import Path
import shutil
import tempfile

from .inventory import FIELDS
from .prepare import split_cases

DATASETS = ('DUKE', 'LUNA16', 'Moltest 1')
LOG = logging.getLogger(__name__)


def build(source, output, seed=42, validation_fraction=.15, test_fraction=.15,
          patient_map=None, resume=False):
    source, output = Path(source), Path(output)
    raw = source.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    required = {'case_id', 'patient_id', 'dataset', 'image', 'airways', 'vessels', 'lungs'}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError('Inventory lacks required columns')
    original = list(reader)
    rows = [{key: r.get(key, '') or '' for key in FIELDS}
            for r in original if r['dataset'] in DATASETS]
    if set(r['dataset'] for r in rows) != set(DATASETS):
        raise ValueError('Each of DUKE, LUNA16 and Moltest 1 must be represented')
    overrides = {}
    map_hash = None
    if patient_map:
        data = Path(patient_map).read_bytes()
        map_hash = hashlib.sha256(data).hexdigest()
        for r in csv.DictReader(io.StringIO(data.decode('utf-8-sig'))):
            if not r['patient_id'].strip() or r['case_id'] in overrides:
                raise ValueError('Empty identity or duplicate case in patient map')
            overrides[r['case_id']] = r['patient_id'].strip()
    seen_ids, seen_images = set(), set()
    for row in rows:
        if any(not row[k].strip() for k in required):
            raise ValueError('Selected inventory row has a blank required field')
        if row['case_id'] in seen_ids or row['image'] in seen_images:
            raise ValueError('Duplicate case ID or image path in selected cohort')
        seen_ids.add(row['case_id'])
        seen_images.add(row['image'])
        row['patient_id'] = overrides.get(row['case_id'], row['patient_id'])
    # Group across cohorts using supplied global identities. No nodule availability
    # assumption is made before preparation; stratification here is by cohort only.
    cases = [dict(r, has_nodules=False) for r in sorted(rows, key=lambda r: r['case_id'])]
    splits = split_cases(cases, seed, validation_fraction, test_fraction)
    if not splits['test']:
        raise ValueError('Need a nonempty protected test set')
    files = {}
    combined = []
    counts = {}
    for split, cases in splits.items():
        selected = []
        for case in sorted(cases, key=lambda r: r['case_id']):
            row = {k: case[k] for k in FIELDS}
            row['split'] = split
            selected.append(row)
        combined.extend(selected)
        counts[split] = {'scans': len(selected),
                         'patient_ids': len({r['patient_id'] for r in selected}),
                         'by_dataset': dict(Counter(r['dataset'] for r in selected))}
        files[split + '.csv'] = selected
    files['inventory.csv'] = combined
    audit = {
        'version': 1, 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'patient_map_sha256': map_hash, 'seed': seed,
        'validation_fraction': validation_fraction, 'test_fraction': test_fraction,
        'source_rows': len(original), 'selected_rows': len(rows),
        'excluded_other_datasets': dict(Counter(r['dataset'] for r in original
                                               if r['dataset'] not in DATASETS)),
        'splits': counts, 'existing_splits_preserved': all(r['split'] for r in rows),
        'identity_overrides_applied': len(seen_ids & overrides.keys()),
        'limitations': [
            'Disjoint supplied patient IDs; biological identity and cross-cohort aliases remain unverified.',
            'Default DUKE and LUNA inventory IDs derive from scans/series; provide verified patient linkage when needed.',
            'Raw files, geometry and mask quality are not checked here. Preparation reads files and checks native voxel hashes across splits.',
            'No malignancy stratification; LUNA class=1 denotes a nodule candidate, not malignancy.',
        ],
    }
    payloads = {}
    for name, records in files.items():
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
        payloads[name] = buffer.getvalue().encode()
    audit['csv_sha256'] = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    payloads['audit.json'] = (json.dumps(audit, indent=2) + '\n').encode()
    if output.exists():
        if not resume:
            raise FileExistsError('Cohort output exists; use --resume to verify unchanged inputs/settings')
        if any(not (output / name).is_file() or (output / name).read_bytes() != data
               for name, data in payloads.items()):
            raise ValueError('Existing cohort differs; preserve it and choose a new output directory')
        return audit
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.cohort-', dir=output.parent))
    try:
        for name, data in payloads.items():
            (staging / name).write_bytes(data)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validation-fraction', type=float, default=.15)
    p.add_argument('--test-fraction', type=float, default=.15)
    p.add_argument('--patient-map', help='Verified global identity CSV: case_id,patient_id')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    audit = build(args.inventory, args.output, args.seed, args.validation_fraction,
                  args.test_fraction, args.patient_map, args.resume)
    LOG.info('Frozen cohort: %s', audit['splits'])
    for limitation in audit['limitations']:
        LOG.warning('%s', limitation)


if __name__ == '__main__':
    main()
