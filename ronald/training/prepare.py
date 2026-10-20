"""CSV inventory -> audited, disk-cached, patient-disjoint manifest."""
import argparse
import csv
import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from .imaging import read_image, binary_on_reference, prepare_image, grow_nodules
from .progress import dataset_progress

LOG = logging.getLogger(__name__)


def select_positive_nodules(annotations, candidates_path):
    """Map class=1 candidates to reference nodules; grow once per reference.

    Reference centers and diameters define growth, rather than multiple slightly
    displaced detection candidates. Class=0 candidates never initiate growth.
    """
    selected = defaultdict(set)
    positive_count = 0
    with open(candidates_path, newline='') as f:
        reader = csv.DictReader(f)
        if not {'seriesuid', 'coordX', 'coordY', 'coordZ', 'class'}.issubset(reader.fieldnames or []):
            raise ValueError('LUNA candidates require seriesuid, coordX/Y/Z, and class')
        for row in reader:
            label = float(row['class'])
            if label not in (0, 1):
                raise ValueError('LUNA candidate class must be 0 or 1')
            if label != 1:
                continue
            positive_count += 1
            uid = row['seriesuid']
            refs = annotations.get(uid, [])
            xyz = np.array([float(row[k]) for k in ('coordX', 'coordY', 'coordZ')])
            if not np.isfinite(xyz).all() or not refs:
                raise ValueError(f'{uid}: positive candidate has invalid coordinates or no reference annotations')
            centers = np.array([[float(r[k]) for k in ('coordX', 'coordY', 'coordZ')] for r in refs])
            radii = np.array([float(r['diameter_mm']) / 2 for r in refs])
            distance = np.linalg.norm(centers - xyz, axis=1)
            matching = np.flatnonzero(distance <= radii + 1e-3)
            if len(matching) != 1:
                raise ValueError(f'{uid}: positive candidate matches {len(matching)} reference nodules; check annotations')
            selected[uid].add(int(matching[0]))
    if not positive_count:
        raise ValueError('No class=1 LUNA candidates found')
    result = defaultdict(list, {uid: [annotations[uid][i] for i in sorted(indices)]
                               for uid, indices in selected.items()})
    audit = {'positive_candidates': positive_count, 'selected_nodules': sum(map(len, result.values())),
             'reference_nodules': sum(map(len, annotations.values()))}
    return result, audit


def split_cases(cases, seed=42, validation_fraction=.15, test_fraction=.15):
    if not 0 < validation_fraction < 1 or not 0 <= test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError('Invalid split fractions')
    explicit = [bool(c.get('split')) for c in cases]
    if any(explicit) and not all(explicit):
        raise ValueError('Set split for all rows or none')
    if all(explicit):
        splits = {k: [c for c in cases if c['split'] == k] for k in ('training', 'validation', 'test')}
        if sum(map(len, splits.values())) != len(cases):
            raise ValueError('split must be training, validation, or test')
    else:
        # Global patient IDs link repeat scans across datasets. Stratify by dataset
        # membership AND availability of usable nodule supervision, at patient level.
        patients = defaultdict(list)
        for c in cases:
            patients[c['patient_id']].append(c)
        strata = defaultdict(list)
        for patient, scans in patients.items():
            key = (tuple(sorted({s['dataset'] for s in scans})), any(s['has_nodules'] for s in scans))
            strata[key].append(patient)
        rng = np.random.default_rng(seed)
        splits = {k: [] for k in ('training', 'validation', 'test')}
        for key in sorted(strata):
            ids = sorted(strata[key])
            rng.shuffle(ids)
            n = len(ids)
            nv = max(1, round(n * validation_fraction)) if n >= 3 else 0
            nt = max(1, round(n * test_fraction)) if test_fraction and n >= 3 else 0
            if nv + nt >= n:
                nv, nt = 1, max(0, n - 2)
            for i, patient in enumerate(ids):
                group = 'validation' if i < nv else ('test' if i < nv + nt else 'training')
                splits[group].extend(patients[patient])
    validate_splits(splits)
    return splits


def validate_splits(splits):
    if not splits.get('training') or not splits.get('validation'):
        raise ValueError('Need nonempty training and validation; small cohorts require explicit patient-level split column')
    seen = {}
    for split, cases in splits.items():
        for c in cases:
            for kind in ('patient_id', 'image', 'image_sha256'):
                if not c.get(kind):
                    continue
                key = (kind, c[kind])
                if key in seen and seen[key] != split:
                    raise ValueError(f'Leakage across {seen[key]}/{split}: {key}')
                seen[key] = split


def prepare(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Use a new output directory; existing manifest will not be overwritten')
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(out / 'prepare.log')])
    inventory = Path(args.inventory).resolve()
    with inventory.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty inventory')
    annotations = defaultdict(list)
    if args.luna_annotations:
        with open(args.luna_annotations, newline='') as f:
            reader = csv.DictReader(f)
            required = {'seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('Use LUNA annotations.csv with diameter_mm, not the unfiltered candidate list')
            for row in reader:
                if 'class' in row:
                    if float(row['class']) not in (0, 1):
                        raise ValueError('Annotation class must be 0 or 1')
                    if float(row['class']) != 1:
                        continue
                annotations[row['seriesuid']].append(row)
    if args.luna_candidates:
        if not args.luna_annotations:
            raise ValueError('--luna-candidates requires --luna-annotations for reference diameters')
        annotations, selection_audit = select_positive_nodules(annotations, args.luna_candidates)
        (out / 'luna_selection.json').write_text(json.dumps(selection_audit, indent=2))
        LOG.info('LUNA class=1 selection: %s', selection_audit)
    seen_ids, seen_images, cases, audits = set(), set(), [], []
    for row in dataset_progress(rows):
        for key in ('case_id', 'patient_id', 'dataset', 'image', 'airways', 'vessels', 'lungs'):
            if not row.get(key, '').strip():
                raise ValueError(f'Missing {key}: {row}')
        case_id = row['case_id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', case_id) or case_id in seen_ids:
            raise ValueError(f'Unsafe/duplicate case_id: {case_id}')
        seen_ids.add(case_id)
        paths = {}
        for key in ('image', 'airways', 'vessels', 'lungs', 'mediastinum', 'nodules'):
            value = row.get(key, '').strip()
            paths[key] = str((inventory.parent / value).resolve()) if value else None
            if paths[key] and not Path(paths[key]).exists():
                raise FileNotFoundError(paths[key])
        image_key = (paths['image'], row.get('series_uid', ''))
        if image_key in seen_images:
            raise ValueError(f'Duplicate image in inventory: {image_key}')
        seen_images.add(image_key)
        LOG.info('Preparing %s (%s)', case_id, row['dataset'])
        image = read_image(paths['image'], row.get('series_uid') or None)
        # Hash native voxel data to detect renamed/reformatted duplicates in splits.
        image_hash = hashlib.sha256(sitk.GetArrayFromImage(image).tobytes()).hexdigest()
        ct, ref, box = prepare_image(image, paths['lungs'], paths['mediastinum'], args.spacing, args.margin_mm, args.hu)
        air = binary_on_reference(paths['airways'], ref)
        vessel = binary_on_reference(paths['vessels'], ref)
        audit = {'case_id': case_id, 'original_size_xyz': list(image.GetSize()), 'crop_shape_zyx': list(ct.shape)}
        for name, mask in [('airways', air), ('vessels', vessel)]:
            if not mask.any():
                raise ValueError(f'{case_id}: empty {name} mask (or physical geometry mismatch)')
            retained = int(mask[box].sum())
            audit[f'{name}_outside_crop'] = int(mask.sum()) - retained
            if retained == 0:
                raise ValueError(f'{case_id}: crop excludes all {name}')
            if audit[f'{name}_outside_crop']:
                LOG.warning('%s: %d %s voxels outside thoracic crop', case_id, audit[f'{name}_outside_crop'], name)
        source = 'unavailable'
        nodules = np.zeros(air.shape, dtype=bool)
        case_dir = out / case_id
        if case_dir.exists():
            raise FileExistsError(f'{case_dir}: use a fresh output directory after an interrupted prepare')
        case_dir.mkdir()
        if paths['nodules']:
            nodules = binary_on_reference(paths['nodules'], ref, allow_instance_labels=True)
            source = 'provided'
        elif row.get('luna_seriesuid') and args.luna_annotations:
            mask, grow_audit = grow_nodules(image, annotations[row['luna_seriesuid']], args.grow_lower_hu,
                                           args.grow_upper_hu, args.grow_tolerance_hu, args.grow_radius_scale)
            sitk.WriteImage(mask, str(case_dir / 'nodules_region_grow.nii.gz'))
            nodules = sitk.GetArrayFromImage(sitk.Resample(mask, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
            source = 'luna_region_grow_unreviewed'
            audit['region_grow'] = grow_audit
        audit['nodules_outside_crop'] = int(nodules.sum() - nodules[box].sum())
        if audit['nodules_outside_crop']:
            raise ValueError(f'{case_id}: crop excludes nodule voxels; correct lung mask or increase margin')
        air, vessel, nodules = air[box], vessel[box], nodules[box]
        labels = np.zeros(ct.shape, dtype=np.uint8)
        labels[air] = 1
        labels[vessel] = 2
        labels[air & vessel] = 255  # ambiguous pseudo-label overlap, ignored
        audit['overlap_ignored_voxels'] = int((air & vessel & ~nodules).sum())
        audit['nodule_foreground_conflicts'] = int(((air | vessel) & nodules).sum())
        labels[nodules] = 0
        has_nodules = bool(nodules.any())
        audit.update(nodule_voxels=int(nodules.sum()), nodule_source=source, has_nodules=has_nodules)
        np.save(case_dir / 'image.npy', ct.astype(np.float16))
        np.save(case_dir / 'label.npy', labels)
        np.save(case_dir / 'nodules.npy', nodules.astype(np.uint8))
        # Small fixed-size candidate pools avoid scanning entire volumes every patch.
        rng = np.random.default_rng(args.seed)
        pools = {}
        for key, mask in [('airway', labels == 1), ('vessel', labels == 2), ('nodule', nodules)]:
            indices = np.flatnonzero(mask)
            if len(indices) > 100000:
                indices = rng.choice(indices, 100000, replace=False)
            pools[key] = indices
        np.savez(case_dir / 'indices.npz', **pools)
        geometry = {'size_xyz': list(ref.GetSize()), 'spacing_xyz': list(ref.GetSpacing()),
                    'origin_xyz': list(ref.GetOrigin()), 'direction': list(ref.GetDirection()),
                    'crop_start_zyx': [s.start for s in box], 'crop_stop_zyx': [s.stop for s in box]}
        (case_dir / 'geometry.json').write_text(json.dumps(geometry, indent=2))
        cases.append({'case_id': case_id, 'patient_id': row['patient_id'], 'dataset': row['dataset'],
                      'cache': case_id, 'image': paths['image'], 'image_sha256': image_hash,
                      'has_nodules': has_nodules, 'nodule_source': source, 'split': row.get('split', '')})
        audits.append(audit)
        (out / 'audit.json').write_text(json.dumps(audits, indent=2))
    splits = split_cases(cases, args.seed, args.validation_fraction, args.test_fraction)
    manifest = {'version': 1, 'classes': {'0': 'background', '1': 'airway_with_wall', '2': 'vessel'},
                'preprocessing': {'spacing_xyz': args.spacing, 'margin_mm': args.margin_mm, 'hu': args.hu},
                'preparation': vars(args), 'splits': splits}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    LOG.info('Saved manifest: %s', {k: len(v) for k, v in splits.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--luna-annotations')
    p.add_argument('--luna-candidates', help='LUNA candidates CSV; only class=1 selects nodules for growing')
    p.add_argument('--spacing', nargs=3, type=float, default=[.8, .8, .8], help='XYZ mm; no fixed-cube resizing')
    p.add_argument('--margin-mm', type=float, default=15.)
    p.add_argument('--hu', nargs=2, type=float, default=[-1000., 500.])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validation-fraction', type=float, default=.15)
    p.add_argument('--test-fraction', type=float, default=.15)
    p.add_argument('--grow-lower-hu', type=float, default=-700.)
    p.add_argument('--grow-upper-hu', type=float, default=400.)
    p.add_argument('--grow-tolerance-hu', type=float, default=250.)
    p.add_argument('--grow-radius-scale', type=float, default=1.25)
    args = p.parse_args()
    if min(args.spacing) <= 0 or args.margin_mm < 0 or args.hu[0] >= args.hu[1]:
        p.error('Require positive spacing, nonnegative margin, increasing HU range')
    if args.grow_lower_hu >= args.grow_upper_hu or args.grow_tolerance_hu <= 0 or args.grow_radius_scale <= 0:
        p.error('Invalid region-growing parameters')
    prepare(args)


if __name__ == '__main__':
    main()
