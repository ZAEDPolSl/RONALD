"""Discover complete NAS cases and write a Linux-path inventory without reading CT pixels."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
from pathlib import Path
import unicodedata
from .imaging import resolve_dicom_series

LOG = logging.getLogger(__name__)
FIELDS = ['case_id', 'patient_id', 'dataset', 'image', 'airways', 'vessels', 'lungs',
          'mediastinum', 'nodules', 'luna_seriesuid', 'series_uid', 'split']


def normalized_identity(text):
    text = text.strip().casefold().replace('ł', 'l')
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))


def plcs_relative_path(value):
    value = value.replace('\\', '/')
    # The verified table contains X:\OPPRP\PLCS\<study UID> paths.
    parts = value.split('/')
    if 'OPPRP' not in parts:
        raise ValueError(f'Unrecognized PLCS path format: {value}')
    relative = Path(*parts[parts.index('OPPRP'):])
    if '..' in relative.parts:
        raise ValueError('Parent traversal in PLCS path')
    return relative


def candidate_scans(dataset, mount):
    raw = mount / dataset['raw']
    if dataset['kind'] == 'plcs4':
        with (mount / dataset['selection']).open(newline='') as f:
            for row in csv.DictReader(f):
                path = mount / plcs_relative_path(row['dicom_path'])
                identity = row.get('patient_key') or row.get('patient_id')
                if not identity:
                    raise ValueError('PLCS4 selection row lacks patient identity')
                yield path.name, path, identity
    else:
        paths = raw.glob(dataset['pattern'])
        for path in sorted(paths):
            if dataset['kind'] == 'dicom':
                scan = path.parent.name
            else:
                scan = path.name[:-7] if path.name.endswith('.nii.gz') else path.stem
            yield scan, path, scan


def build_inventory(config, read_mount=None, patient_map=None):
    """read_mount allows inspecting the macOS NAS mount while emitting Linux paths."""
    mount = Path(read_mount or config['mount'])
    target = Path(config['mount'])
    rows, report, seen = [], [], set()
    overrides = patient_map or {}
    for dataset in config['datasets']:
        LOG.info('Discovering %s', dataset['name'])
        root = mount / dataset['raw']
        masks = mount / dataset['masks']
        if not root.is_dir() or not masks.is_dir():
            raise FileNotFoundError(f'Missing configured roots for {dataset["name"]}: {root}, {masks}')
        for scan, image, identity in candidate_scans(dataset, mount):
            case_id = dataset['key'] + '_' + hashlib.sha256(scan.encode()).hexdigest()[:16]
            if case_id in seen:
                raise ValueError(f'Duplicate scan identifier in {dataset["name"]}: {scan}; resolve before preparing')
            seen.add(case_id)
            folder = masks / scan
            required_masks = ('airways_final.nrrd', 'vessels_final.nrrd', 'lungs.nrrd')
            for candidate in [folder, *[mount / extra / scan for extra in dataset.get('additional_masks', [])]]:
                if all((candidate / name).is_file() for name in required_masks):
                    folder = candidate
                    break
            paths = {'image': image, **{k: folder / filename for k, filename in
                     [('airways', 'airways_final.nrrd'), ('vessels', 'vessels_final.nrrd'), ('lungs', 'lungs.nrrd')]}}
            missing = [key for key, path in paths.items() if not path.exists()]
            if missing:
                report.append({'dataset': dataset['name'], 'scan': scan, 'status': 'missing', 'fields': missing})
                continue
            if dataset['kind'] in ('dicom', 'plcs4'):
                selected = resolve_dicom_series(image)
                if selected is None:
                    report.append({'dataset': dataset['name'], 'scan': scan, 'status': 'no_dicom_series', 'image': str(paths['image'])})
                    LOG.warning('%s / %s: ctools found no DICOM series under %s', dataset['name'], scan, image)
                    continue
                image, series_uid = selected
                paths['image'] = image
            else:
                series_uid = ''
            for key, filename in [('mediastinum', 'mediastinum.nrrd'), ('nodules', 'region_grow.nrrd')]:
                path = folder / filename
                if path.is_file() and not (key == 'nodules' and dataset['key'] == 'luna16'):
                    paths[key] = path
            # Do not infer the semantics of intermediate/mimseg/ROI variants from
            # their filenames. Only the configured canonical region-grow mask is used.
            row = {key: '' for key in FIELDS}
            row.update({key: str(target / path.relative_to(mount)) for key, path in paths.items()})
            patient_key = dataset['identity_group'] + ':' + normalized_identity(identity)
            row.update(case_id=case_id, patient_id=overrides.get(case_id, hashlib.sha256(patient_key.encode()).hexdigest()),
                       dataset=dataset['name'], series_uid=series_uid,
                       luna_seriesuid=scan if dataset['key'] == 'luna16' else '')
            rows.append(row)
            report.append({'dataset': dataset['name'], 'scan': scan, 'case_id': case_id,
                           'image': row['image'], 'series_uid': series_uid,
                           'series_selection': 'ctools_most_slices' if series_uid else 'volume_file',
                           'status': 'included', 'identity_source': 'override' if case_id in overrides else dataset['identity_group']})
    if not rows:
        raise ValueError('No complete cases found')
    return rows, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/training/pmanas.json')
    p.add_argument('--output', required=True)
    p.add_argument('--read-mount', help='Optional local mount; emitted paths retain configured Linux prefix')
    p.add_argument('--patient-map', help='Optional CSV with case_id,patient_id for verified cross-cohort linkage')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    overrides = {}
    if args.patient_map:
        with open(args.patient_map, newline='') as f:
            for row in csv.DictReader(f):
                if not row['patient_id'] or row['case_id'] in overrides:
                    raise ValueError('Empty/duplicate patient-map entry')
                overrides[row['case_id']] = row['patient_id']
    rows, report = build_inventory(config, args.read_mount, overrides)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    audit = {'included_by_dataset': dict(Counter(r['dataset'] for r in rows)),
             'status_counts': dict(Counter(r['status'] for r in report)), 'cases': report}
    out.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2))
    LOG.info('Included cases: %s', audit['included_by_dataset'])
    LOG.info('Statuses: %s', audit['status_counts'])
    LOG.warning('Review patient linkage: Moltest uses normalized folder IDs; PLCS4 uses patient_key; '
                'DUKE uses scan IDs; LUNA uses series IDs. Use --patient-map for verified aliases.')


if __name__ == '__main__':
    main()
