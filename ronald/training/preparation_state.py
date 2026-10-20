"""Atomic preparation state, input provenance, and scan completion records."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from zipfile import BadZipFile
import numpy as np

CACHE_VERSION = 1
OUTPUT_FILES = ('image.npy', 'label.npy', 'nodules.npy', 'indices.npz', 'geometry.json')


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preparation_identity(args):
    settings = {k: v for k, v in vars(args).items() if k not in ('output', 'resume', 'adopt_legacy_cache')}
    files = {}
    for key in ('inventory', 'luna_annotations', 'luna_candidates'):
        if settings.get(key):
            path = Path(settings[key]).resolve()
            settings[key] = str(path)
            files[key] = sha256_file(path)
    return {'cache_version': CACHE_VERSION, 'settings': settings, 'input_sha256': files}


def file_stat(path):
    stat = path.stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def source_snapshot(paths):
    """Stat source files without decoding CT pixels; include DICOM and MHD payloads."""
    result = {}
    for value in paths.values():
        if not value:
            continue
        path = Path(value)
        if path.is_dir():
            files = sorted(p for p in path.rglob('*') if p.is_file())
        else:
            files = [path]
            if path.suffix.lower() == '.mhd':
                lines = path.read_text().splitlines()
                payload = next((line.split('=', 1)[1].strip() for line in lines
                                if line.strip().startswith('ElementDataFile') and '=' in line), None)
                if payload and payload != 'LOCAL':
                    if payload.startswith('LIST') or '%' in payload:
                        raise ValueError(f'{path}: unsupported multi-file MHD provenance; convert to NIfTI/NRRD')
                    files.append((path.parent / payload).resolve())
        for file in files:
            result[str(file)] = file_stat(file)
    return result


@contextmanager
def preparation_lock(out):
    """Advisory lock releases on process exit, including interruption."""
    with (out / '.prepare.lock').open('a') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another preparation process is using {out}') from exc
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class PreparationState:
    def __init__(self, out, args):
        self.out = out
        self.identity = preparation_identity(args)
        self.token = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        state_path = out / 'prepare_state.json'
        if state_path.exists():
            if not args.resume:
                raise FileExistsError('Preparation already exists; use --resume with unchanged inputs/settings')
            saved = json.loads(state_path.read_text())
            if saved['identity'] != self.identity:
                raise ValueError('Preparation inputs/settings changed; use the original arguments or a new cache directory')
        else:
            existing = [p for p in out.iterdir() if p.name not in ('.prepare.lock', 'prepare_state.json.tmp')]
            if existing and not (args.resume and args.adopt_legacy_cache):
                raise ValueError('Legacy/nonempty cache has no preparation state. For an old cache with the SAME '
                                 'original inputs/settings, use --resume --adopt-legacy-cache; otherwise use a new directory')
            if existing:
                audit_path = out / 'audit.json'
                if not audit_path.exists():
                    raise ValueError('Cannot adopt legacy cache without audit.json')
                audit = json.loads(audit_path.read_text())
                atomic_json(out / 'legacy_audit.json', audit)
            saved = {'identity': self.identity, 'legacy_allowed': bool(existing)}
            atomic_json(state_path, saved)
        self.legacy = {}
        if saved.get('legacy_allowed'):
            self.legacy = {a['case_id']: a for a in json.loads((out / 'legacy_audit.json').read_text())}
        self.legacy_allowed = saved.get('legacy_allowed', False)

    def completed(self, case_id, sources):
        folder = self.out / case_id
        marker = folder / 'complete.json'
        if not marker.exists():
            return None
        saved = json.loads(marker.read_text())
        if saved['preparation_token'] != self.token or saved['sources'] != sources:
            raise ValueError(f'{case_id}: completed scan inputs or settings changed; use a new cache')
        for name, expected in saved['outputs'].items():
            if not (folder / name).is_file() or file_stat(folder / name) != expected:
                raise ValueError(f'{case_id}: completed cache file changed or missing: {name}; '
                                 'move this scan directory aside to rebuild it with --resume')
        return saved

    def staging(self, case_id):
        final = self.out / case_id
        if final.exists():
            # A legacy scan absent from audit.json was not completed. Preserve it.
            if not self.legacy_allowed:
                raise ValueError(f'{case_id}: unexpected unmarked scan directory')
            final.rename(self.out / f'.legacy-incomplete-{case_id}-{uuid.uuid4().hex[:8]}')
        parent = self.out / '.partial'
        parent.mkdir(exist_ok=True)
        folder = parent / case_id
        if folder.exists():
            shutil.rmtree(folder)  # owned staging output only; never a source path
        folder.mkdir()
        return folder

    def finish(self, folder, case, audit, sources):
        names = list(OUTPUT_FILES)
        if (folder / 'nodules_region_grow.nii.gz').exists():
            names.append('nodules_region_grow.nii.gz')
        # Flush all output files before committing the completion marker.
        for name in names:
            with (folder / name).open('rb') as f:
                os.fsync(f.fileno())
        saved = {'preparation_token': self.token, 'sources': sources, 'case': case, 'audit': audit,
                 'outputs': {name: file_stat(folder / name) for name in names}}
        atomic_json(folder / 'complete.json', saved)
        if folder != self.out / case['case_id']:
            os.replace(folder, self.out / case['case_id'])
        return saved

    def validate_legacy(self, case_id, spacing):
        """Validate old completed arrays; original HU/margin cannot be inferred.

        Adoption is an explicit user assertion that the supplied arguments and
        source files are those used by the old preparation.
        """
        if case_id not in self.legacy:
            return None
        folder = self.out / case_id
        audit = self.legacy[case_id]
        try:
            arrays = {n: np.load(folder / f'{n}.npy', mmap_mode='r', allow_pickle=False)
                      for n in ('image', 'label', 'nodules')}
            shape = tuple(audit['crop_shape_zyx'])
            if any(a.shape != shape for a in arrays.values()):
                return None
            if arrays['image'].dtype != np.float16 or any(arrays[n].dtype != np.uint8 for n in ('label', 'nodules')):
                return None
            geometry = json.loads((folder / 'geometry.json').read_text())
            if not np.allclose(geometry['spacing_xyz'], spacing):
                raise RuntimeError(f'{case_id}: legacy cache spacing differs from requested spacing')
            if tuple(np.subtract(geometry['crop_stop_zyx'], geometry['crop_start_zyx'])) != shape:
                return None
            voxels = int(np.prod(shape))
            with np.load(folder / 'indices.npz', allow_pickle=False) as pools:
                for key in ('airway', 'vessel', 'nodule'):
                    indices = pools[key]
                    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= voxels):
                        return None
            if audit['nodule_source'] == 'luna_region_grow_unreviewed' and not (folder / 'nodules_region_grow.nii.gz').is_file():
                return None
        except (OSError, EOFError, KeyError, ValueError, BadZipFile):
            return None
        return audit
