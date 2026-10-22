"""Discover complete NAS cases and write a Linux-path inventory without reading CT pixels."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import unicodedata
from .imaging import resolve_dicom_series
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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
        for path in paths:
            if dataset['kind'] == 'dicom':
                scan = path.parent.name
            else:
                scan = path.name[:-7] if path.name.endswith('.nii.gz') else path.stem
            yield scan, path, scan


def build_inventory(config, read_mount=None, patient_map=None, completed=None, notify=None):
    """read_mount allows inspecting the macOS NAS mount while emitting Linux paths."""
    mount = Path(read_mount or config['mount'])
    target = Path(config['mount'])
    rows, report, seen = [], [], set()
    overrides = patient_map or {}
    completed = completed or {}
    notify = notify or (lambda *args: None)

    class Report(list):
        def append(self, entry):
            super().append(entry)
            notify("completed", {"case_id": case_id, "report": entry,
                                  "row": rows[-1] if entry["status"] == "included" else None})

    report = Report()
    for dataset in config['datasets']:
        LOG.info('Discovering %s', dataset['name'])
        notify('current', f"Checking roots / discovering {dataset['name']}")
        root = mount / dataset['raw']
        masks = mount / dataset['masks']
        if not root.is_dir() or not masks.is_dir():
            raise FileNotFoundError(f'Missing configured roots for {dataset["name"]}: {root}, {masks}')
        scans = candidate_scans(dataset, mount)
        included = skipped = 0
        with logging_redirect_tqdm(), tqdm(scans, desc=f"Inventory: {dataset['name']}",
                                          unit='scan', dynamic_ncols=True) as progress:
            for scan, image, identity in progress:
                case_id = dataset['key'] + '_' + hashlib.sha256(scan.encode()).hexdigest()[:16]
                if case_id in seen:
                    raise ValueError(f'Duplicate scan identifier in {dataset["name"]}: {scan}; resolve before preparing')
                seen.add(case_id)
                if case_id in completed:
                    entry = completed[case_id]
                    if entry['row'] is not None:
                        rows.append(entry['row'])
                        included += 1
                    else:
                        skipped += 1
                    list.append(report, entry['report'])
                    notify('current', f"Discovering {dataset['name']} after cached {scan}")
                    progress.set_postfix(included=included, skipped=skipped, refresh=False)
                    continue
                notify('scan', {'case_id': case_id, 'dataset': dataset['name'],
                                'scan': scan, 'image': str(image)})
                notify('current', f"Checking masks for {dataset['name']} / {scan}")
                folder = masks / scan
                required_masks = ('airways_final.nrrd', 'vessels_final.nrrd', 'lungs.nrrd')
                for candidate in [folder, *[mount / extra / scan for extra in dataset.get('additional_masks', [])]]:
                    notify('current', f'Checking mask files: {candidate}')
                    if all((candidate / name).is_file() for name in required_masks):
                        folder = candidate
                        break
                paths = {'image': image, **{k: folder / filename for k, filename in
                         [('airways', 'airways_final.nrrd'), ('vessels', 'vessels_final.nrrd'), ('lungs', 'lungs.nrrd')]}}
                notify('current', f'Checking required paths: {image}')
                missing = [key for key, path in paths.items() if not path.exists()]
                if missing:
                    report.append({'dataset': dataset['name'], 'scan': scan, 'status': 'missing', 'fields': missing})
                    skipped += 1
                    progress.set_postfix(included=included, skipped=skipped, refresh=False)
                    continue
                if dataset['kind'] in ('dicom', 'plcs4'):
                    notify('current', f'ctools DICOM discovery: {image}')
                    selected = resolve_dicom_series(image)
                    if selected is None:
                        report.append({'dataset': dataset['name'], 'scan': scan, 'status': 'no_dicom_series', 'image': str(paths['image'])})
                        LOG.warning('%s / %s: ctools found no DICOM series under %s', dataset['name'], scan, image)
                        skipped += 1
                        progress.set_postfix(included=included, skipped=skipped, refresh=False)
                        continue
                    image, series_uid = selected
                    paths['image'] = image
                else:
                    series_uid = ''
                notify('current', f'Checking optional masks: {folder}')
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
                included += 1
                progress.set_postfix(included=included, skipped=skipped, refresh=False)
    if not rows:
        raise ValueError('No complete cases found')
    return rows, list(report)


def _inventory_worker(connection, config, read_mount, overrides, completed):
    try:
        result = build_inventory(config, read_mount, overrides, completed,
                                 lambda kind, data: connection.send((kind, data)))
        connection.send(('done', result))
    except BaseException as exc:
        connection.send(('error', f'{type(exc).__name__}: {exc}'))
    finally:
        connection.close()


def resumable_inventory(config, read_mount, overrides, out, resume=False, timeout=120, skip_timeouts=False, retry_timeouts=False, state_dir=None):
    """Bound stalled filesystem/native-library calls in a disposable child process."""
    from .preparation_state import atomic_json
    import fcntl
    state_dir = Path(state_dir) if state_dir else out.with_suffix('.inventory-state')
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'version': 1, 'config': config, 'read_mount': str(read_mount) if read_mount else None,
                    'patient_map': overrides}
        settings = state_dir / 'settings.json'
        if settings.exists():
            if not resume:
                raise ValueError('Inventory checkpoint exists; use --resume')
            if json.loads(settings.read_text()) != identity:
                raise ValueError('Inventory settings changed; use a new output filename')
        else:
            if out.exists():
                raise ValueError('Existing inventory has no checkpoint; use a new output filename')
            atomic_json(settings, identity)
        completed = {p.stem: json.loads(p.read_text()) for p in state_dir.glob('case-*.json')}
        completed = {entry['case_id']: entry for entry in completed.values()}
        if retry_timeouts:
            for key, entry in list(completed.items()):
                if entry['report']['status'] == 'timeout':
                    del completed[key]
        while True:
            ctx = multiprocessing.get_context('spawn')
            receiver, sender = ctx.Pipe(duplex=False)
            worker = ctx.Process(target=_inventory_worker,
                                 args=(sender, config, read_mount, overrides, completed), daemon=True)
            current = 'Starting inventory worker'
            active_scan = None
            worker.start()
            sender.close()
            try:
                while True:
                    if not receiver.poll(timeout):
                        LOG.warning('Inventory timeout after %gs: %s (worker PID %s)',
                                    timeout, current, worker.pid)
                        if skip_timeouts and active_scan is not None:
                            entry = {'case_id': active_scan['case_id'], 'row': None,
                                     'report': dict(active_scan, status='timeout', stage=current,
                                                    timeout_seconds=timeout)}
                            LOG.warning('Saving timeout exclusion to %s', state_dir)
                            atomic_json(state_dir / ('case-' + entry['case_id'] + '.json'), entry)
                            completed[entry['case_id']] = entry
                            LOG.warning('Deferring timed-out scan: %s', current)
                            break
                        raise TimeoutError(f'Inventory stalled for {timeout:g}s: {current}. '
                                           f'Completed checks saved in {state_dir}; retry with --resume.')
                    try:
                        kind, data = receiver.recv()
                    except EOFError as exc:
                        raise RuntimeError(f'Inventory worker exited unexpectedly: {current}') from exc
                    if kind == 'scan':
                        active_scan = data
                    elif kind == 'current':
                        current = data
                        if current.startswith(('Discovering ', 'Checking roots /')):
                            active_scan = None
                        atomic_json(state_dir / 'current.json', {'stage': current, 'scan': active_scan})
                        LOG.info('%s', current)
                    elif kind == 'completed':
                        completed[data['case_id']] = data
                        active_scan = None
                        atomic_json(state_dir / ('case-' + data['case_id'] + '.json'), data)
                        current = 'Discovering next scan after ' + data['case_id']
                        atomic_json(state_dir / 'current.json', {'stage': current, 'scan': active_scan})
                    elif kind == 'error':
                        raise RuntimeError(data)
                    elif kind == 'done':
                        atomic_json(state_dir / 'current.json', {'stage': 'complete'})
                        return data
            finally:
                worker.join(timeout=0.5)
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=2)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=2)
                if worker.is_alive():
                    LOG.error('Worker PID %s remains alive after SIGKILL; possible blocked kernel I/O. '
                              'Stop this run and check the storage mount before restarting.', worker.pid)
                    # Do not accumulate blocked workers by restarting discovery.
                    raise RuntimeError('Inventory worker could not be stopped; inspect storage/process state')
                receiver.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/training/pmanas.json')
    p.add_argument('--output', required=True)
    p.add_argument('--read-mount', help='Optional local mount; emitted paths retain configured Linux prefix')
    p.add_argument('--patient-map', help='Optional CSV with case_id,patient_id for verified cross-cohort linkage')
    p.add_argument('--resume', action='store_true', help='Reuse completed inventory checks')
    p.add_argument('--timeout-seconds', type=float, default=120, help='Stop after this long without progress (default: 120)')
    p.add_argument('--skip-timeouts', action='store_true',
                   help='Audit and defer timed-out scans, then continue with remaining scans')
    p.add_argument('--retry-timeouts', action='store_true',
                   help='Recheck scans previously deferred for timeout')
    p.add_argument('--state-dir', help='Checkpoint directory; use local disk to isolate progress writes from NAS stalls')
    args = p.parse_args()
    if not 0 < args.timeout_seconds < float('inf'):
        p.error('--timeout-seconds must be positive and finite')
    logging.basicConfig(level=logging.INFO)
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    overrides = {}
    if args.patient_map:
        with open(args.patient_map, newline='') as f:
            for row in csv.DictReader(f):
                if not row['patient_id'] or row['case_id'] in overrides:
                    raise ValueError('Empty/duplicate patient-map entry')
                overrides[row['case_id']] = row['patient_id']
    rows, report = resumable_inventory(config, args.read_mount, overrides, out,
                                       args.resume, args.timeout_seconds, args.skip_timeouts, args.retry_timeouts, args.state_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + '.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["case_id"]))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, out)
    audit = {'included_by_dataset': dict(Counter(r['dataset'] for r in rows)),
             'status_counts': dict(Counter(r['status'] for r in report)), 'cases': report}
    out.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2))
    LOG.info('Included cases: %s', audit['included_by_dataset'])
    LOG.info('Statuses: %s', audit['status_counts'])
    LOG.warning('Review patient linkage: Moltest uses normalized folder IDs; PLCS4 uses patient_key; '
                'DUKE uses scan IDs; LUNA uses series IDs. Use --patient-map for verified aliases.')


if __name__ == '__main__':
    main()
