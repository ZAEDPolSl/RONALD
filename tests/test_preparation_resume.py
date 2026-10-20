"""Scan-level interruption, provenance checks, atomic publication, and old-cache migration."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pytest
import SimpleITK as sitk
import ronald.training.prepare as prep
from ronald.training.preparation_state import preparation_lock, source_snapshot


@pytest.fixture
def inputs(tmp_path):
    rows = []
    for i, split in enumerate(['training', 'validation', 'test']):
        folder = tmp_path / f'source{i}'
        folder.mkdir()
        rng = np.random.default_rng(i)
        arrays = {'image': rng.normal(-600, 20, (8, 8, 8)).astype(np.float32),
                  'lungs': np.ones((8, 8, 8), np.uint8),
                  'airways': np.zeros((8, 8, 8), np.uint8),
                  'vessels': np.zeros((8, 8, 8), np.uint8)}
        arrays['airways'][:, 1:3, 1:3] = 1
        arrays['vessels'][:, 5:7, 5:7] = 1
        row = dict(case_id=f'c{i}', patient_id=f'p{i}', dataset='test', split=split)
        for key, array in arrays.items():
            path = folder / f'{key}.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
            row[key] = str(path)
        rows.append(row)
    inventory = tmp_path / 'inventory.csv'
    with inventory.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args = argparse.Namespace(inventory=str(inventory), output=str(tmp_path / 'cache'),
                              resume=False, adopt_legacy_cache=False, luna_annotations=None, luna_candidates=None,
                              spacing=[1., 1., 1.], margin_mm=0., hu=[-1000., 500.], seed=42,
                              validation_fraction=.15, test_fraction=.15, grow_lower_hu=-700., grow_upper_hu=400.,
                              grow_tolerance_hu=250., grow_radius_scale=1.25)
    return args, rows


def test_interrupted_scan_rebuilt_and_completed_scan_not_read_again(inputs, monkeypatch):
    args, rows = inputs
    out = Path(args.output)
    original_save = np.save
    def fail_second_scan(file, *a, **kw):
        if Path(file).parent.name == 'c1' and Path(file).name == 'label.npy':
            raise KeyboardInterrupt('simulated interruption')
        return original_save(file, *a, **kw)
    monkeypatch.setattr(np, 'save', fail_second_scan)
    with pytest.raises(KeyboardInterrupt):
        prep.prepare(args)
    assert (out / 'c0/complete.json').exists()
    assert not (out / 'c1').exists()
    assert (out / '.partial/c1/image.npy').exists()
    assert not (out / 'manifest.json').exists()
    before = (out / 'c0/image.npy').stat().st_mtime_ns
    monkeypatch.setattr(np, 'save', original_save)
    original_read = prep.read_image
    reads = []
    def track(path, *a, **kw):
        reads.append(path)
        return original_read(path, *a, **kw)
    monkeypatch.setattr(prep, 'read_image', track)
    args.resume = True
    prep.prepare(args)
    assert rows[0]['image'] not in reads
    assert reads == [rows[1]['image'], rows[2]['image']]
    assert before == (out / 'c0/image.npy').stat().st_mtime_ns
    manifest_bytes = (out / 'manifest.json').read_bytes()
    reads.clear()
    prep.prepare(args)
    assert reads == []
    assert (out / 'manifest.json').read_bytes() == manifest_bytes
    assert len(json.loads((out / 'audit.json').read_text())) == 3


def test_reject_changed_settings_inventory_source_and_cache(inputs):
    args, rows = inputs
    prep.prepare(args)
    args.resume = True
    args.hu = [-900., 500.]
    with pytest.raises(ValueError, match='inputs/settings changed'):
        prep.prepare(args)
    args.hu = [-1000., 500.]
    inventory = Path(args.inventory)
    original = inventory.read_bytes()
    inventory.write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='inputs/settings changed'):
        prep.prepare(args)
    inventory.write_bytes(original)
    source = Path(rows[0]['airways'])
    source.touch()
    with pytest.raises(ValueError, match='completed scan inputs'):
        prep.prepare(args)


def test_changed_cache_and_concurrent_writer_rejected(inputs):
    args, _ = inputs
    prep.prepare(args)
    args.resume = True
    out = Path(args.output)
    (out / 'c0/image.npy').unlink()
    with pytest.raises(ValueError, match='cache file changed or missing'):
        prep.prepare(args)
    with preparation_lock(out):
        with pytest.raises(RuntimeError, match='Another preparation process'):
            prep.prepare(args)


def test_legacy_adoption_reuses_completed_arrays(inputs, monkeypatch):
    args, _ = inputs
    prep.prepare(args)
    out = Path(args.output)
    # Reproduce the previous cache format: audit and arrays, no completion records.
    (out / 'prepare_state.json').unlink()
    for marker in out.glob('*/complete.json'):
        marker.unlink()
    (out / 'manifest.json').unlink()
    audit = json.loads((out / 'audit.json').read_text())
    (out / 'audit.json').write_text(json.dumps(audit[:2]))
    (out / 'c2/label.npy').unlink()  # interrupted old scan
    timestamps = [(out / f'c{i}/image.npy').stat().st_mtime_ns for i in range(2)]
    args.resume = True
    with pytest.raises(ValueError, match='Legacy/nonempty'):
        prep.prepare(args)
    args.adopt_legacy_cache = True
    original_prepare = prep.prepare_image
    calls = []
    def track(*a, **kw):
        calls.append(1)
        return original_prepare(*a, **kw)
    monkeypatch.setattr(prep, 'prepare_image', track)
    prep.prepare(args)
    assert len(calls) == 1
    assert timestamps == [(out / f'c{i}/image.npy').stat().st_mtime_ns for i in range(2)]
    assert list(out.glob('.legacy-incomplete-c2-*'))
    assert all((out / f'c{i}/complete.json').exists() for i in range(3))
    args.adopt_legacy_cache = False
    prep.prepare(args)
    assert len(calls) == 1


def test_source_snapshot_tracks_mhd_payload_and_dicom_files(tmp_path):
    header = tmp_path / 'ct.mhd'
    header.write_text('ElementDataFile = ct.raw\n')
    raw = tmp_path / 'ct.raw'
    raw.write_bytes(b'payload')
    dicom = tmp_path / 'dicom'
    dicom.mkdir()
    (dicom / 'slice.dcm').write_bytes(b'dicom')
    first = source_snapshot({'image': str(header), 'other': str(dicom)})
    assert str(raw) in first and str(dicom / 'slice.dcm') in first
    raw.write_bytes(b'changed payload')
    assert source_snapshot({'image': str(header), 'other': str(dicom)}) != first


def test_completion_survives_interruption_before_aggregate_audit(inputs, monkeypatch):
    args, rows = inputs
    original_json = prep.atomic_json
    def fail_audit(path, value):
        if Path(path).name == 'audit.json':
            raise OSError('simulated disk interruption after scan commit')
        return original_json(path, value)
    monkeypatch.setattr(prep, 'atomic_json', fail_audit)
    with pytest.raises(OSError, match='simulated disk interruption'):
        prep.prepare(args)
    out = Path(args.output)
    assert (out / 'c0/complete.json').exists()
    assert not (out / 'audit.json').exists()
    monkeypatch.setattr(prep, 'atomic_json', original_json)
    original_read = prep.read_image
    def forbid_completed(path, *a, **kw):
        assert path != rows[0]['image']
        return original_read(path, *a, **kw)
    monkeypatch.setattr(prep, 'read_image', forbid_completed)
    args.resume = True
    prep.prepare(args)
    assert len(json.loads((out / 'audit.json').read_text())) == 3
