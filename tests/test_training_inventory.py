import json
from pathlib import Path
import numpy as np
import pytest
import SimpleITK as sitk
from ronald.training.inventory import build_inventory, candidate_scans, plcs_relative_path
from ronald.training.imaging import binary_on_reference


def test_inventory_remaps_mount_and_filters_incomplete_cases(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for name in ['complete', 'incomplete']:
        (raw / f'{name}.nrrd').touch()
        folder = tmp_path / 'masks' / name
        folder.mkdir(parents=True)
        for filename in ['airways_final.nrrd', 'lungs.nrrd']:
            (folder / filename).touch()
    (tmp_path / 'masks/complete/vessels_final.nrrd').touch()
    (tmp_path / 'masks/complete/region_grow_roi.nrrd').touch()
    config = {'mount': '/mnt/pmanas', 'datasets': [dict(name='Moltest 1', key='moltest1', raw='raw',
              masks='masks', kind='volume', pattern='*.nrrd', identity_group='moltest')]}
    rows, report = build_inventory(config, tmp_path)
    assert len(rows) == 1
    assert rows[0]['image'] == '/mnt/pmanas/raw/complete.nrrd'
    assert rows[0]['nodules'] == ''  # Unverified ROI variants are not automatically selected.
    assert any(r['status'] == 'missing' for r in report)
    (tmp_path / 'masks/complete/region_grow.nrrd').touch()
    assert build_inventory(config, tmp_path)[0][0]['nodules'].endswith('/region_grow.nrrd')


def test_luna_discovery_excludes_lung_segmentations(tmp_path):
    for folder in ['raw/subset0', 'raw/seg-lungs-LUNA16']:
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / 'uid.mhd').touch()
    scans = list(candidate_scans(dict(raw='raw', kind='volume', pattern='subset*/**/*.mhd'), tmp_path))
    assert len(scans) == 1
    assert scans[0][0] == 'uid'


def test_plcs_windows_path_and_patient_linkage(tmp_path):
    assert str(plcs_relative_path(r'X:\OPPRP\PLCS\2.25.1')) == 'OPPRP/PLCS/2.25.1'
    (tmp_path / 'selected.csv').write_text('dicom_path,patient_key\nX:\\OPPRP\\PLCS\\2.25.1,p1\nX:\\OPPRP\\PLCSv2\\2.25.2,p1\n')
    rows = list(candidate_scans(dict(raw='OPPRP', kind='plcs4', selection='selected.csv'), tmp_path))
    assert rows[0][2] == rows[1][2] == 'p1'
    assert rows[0][1] == tmp_path / 'OPPRP/PLCS/2.25.1'


def test_nodule_instance_labels_are_unioned_but_invalid_masks_rejected(tmp_path):
    array = np.zeros((4, 4, 4), dtype=np.uint16)
    array[1, 1, 1], array[2, 2, 2] = 2, 7
    ref = sitk.GetImageFromArray(array)
    path = tmp_path / 'region_grow.nrrd'
    sitk.WriteImage(ref, str(path))
    assert binary_on_reference(path, ref, allow_instance_labels=True).sum() == 2
    with pytest.raises(ValueError):
        binary_on_reference(path, ref)


def test_additional_moltest_masks_are_used_only_when_primary_incomplete(tmp_path):
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'raw/scan.nrrd').touch()
    (tmp_path / 'masks/scan').mkdir(parents=True)
    fallback = tmp_path / 'masks400/scan'
    fallback.mkdir(parents=True)
    for filename in ['airways_final.nrrd', 'vessels_final.nrrd', 'lungs.nrrd']:
        (fallback / filename).touch()
    config = {'mount': '/mnt/pmanas', 'datasets': [dict(name='Moltest 2', key='moltest2', raw='raw',
              masks='masks', additional_masks=['masks400'], kind='volume', pattern='*.nrrd', identity_group='moltest')]}
    rows, _ = build_inventory(config, tmp_path)
    assert rows[0]['airways'] == '/mnt/pmanas/masks400/scan/airways_final.nrrd'


def write_dicom_series(directory, uid, slices):
    directory.mkdir(parents=True)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    for z in range(slices):
        image = sitk.GetImageFromArray(np.full((12, 12), z, dtype=np.int16))
        for tag, value in {'0008|0060': 'CT', '0020|000e': uid,
                           '0020|000d': '1.2.826.0.1.3680043.2.1125.999',
                           '0020|0032': f'0\\0\\{z}', '0020|0037': '1\\0\\0\\0\\1\\0',
                           '0020|0013': str(z + 1)}.items():
            image.SetMetaData(tag, value)
        writer.SetFileName(str(directory / f'{z:03d}.dcm'))
        writer.Execute(image)


def test_ctools_nested_inventory_and_reading(tmp_path):
    from ronald.training.imaging import read_image, resolve_dicom_series
    parent = tmp_path / 'raw/group/person/DICOM'
    selected_dir = parent / 'study/axial'
    uid = '1.2.826.0.1.3680043.2.1125.1234'
    write_dicom_series(selected_dir, uid, 12)
    write_dicom_series(parent / 'study/other', uid + '.1', 10)
    warnings_before = sitk.ProcessObject.GetGlobalWarningDisplay()
    resolved = resolve_dicom_series(parent)
    assert resolved == (selected_dir, uid)
    assert sitk.ProcessObject.GetGlobalWarningDisplay() == warnings_before
    assert read_image(parent).GetSize() == (12, 12, 12)
    assert read_image(selected_dir, uid).GetSize() == (12, 12, 12)
    with pytest.raises(ValueError, match='selected series_uid not found'):
        read_image(selected_dir, 'nonexistent')
    folder = tmp_path / 'masks/person'
    folder.mkdir(parents=True)
    for filename in ['airways_final.nrrd', 'vessels_final.nrrd', 'lungs.nrrd']:
        (folder / filename).touch()
    config = {'mount': '/mnt/pmanas', 'datasets': [dict(name='Moltest 2', key='moltest2', raw='raw',
              masks='masks', kind='dicom', pattern='*/*/DICOM', identity_group='moltest')]}
    rows, report = build_inventory(config, tmp_path)
    assert rows[0]['image'].endswith('/DICOM/study/axial')
    assert rows[0]['series_uid'] == uid
    assert report[0]['series_selection'] == 'ctools_most_slices'


def test_ctools_empty_directory_and_direct_series(tmp_path):
    from ronald.training.imaging import resolve_dicom_series
    assert resolve_dicom_series(tmp_path) is None
    uid = '1.2.826.0.1.3680043.2.1125.5678'
    write_dicom_series(tmp_path / 'direct', uid, 12)
    assert resolve_dicom_series(tmp_path / 'direct') == (tmp_path / 'direct', uid)


def test_inventory_checkpoint_survives_failure_and_resume(tmp_path):
    from ronald.training.inventory import resumable_inventory
    for folder in ['raw', 'masks/scan']:
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / 'raw/scan.nrrd').touch()
    for name in ['airways_final.nrrd', 'vessels_final.nrrd', 'lungs.nrrd']:
        (tmp_path / 'masks/scan' / name).touch()
    dataset = dict(name='one', key='one', raw='raw', masks='masks',
                   kind='volume', pattern='*.nrrd', identity_group='one')
    config = {'mount': str(tmp_path), 'datasets': [dataset, dict(dataset, name='two', key='two', raw='later')]}
    out = tmp_path / 'inventory.csv'
    with pytest.raises(RuntimeError, match='Missing configured roots'):
        resumable_inventory(config, None, {}, out, timeout=30)
    assert len(list(out.with_suffix('.inventory-state').glob('case-*.json'))) == 1
    # Completed mask checks must not run again. Source tree is otherwise assumed immutable.
    (tmp_path / 'masks/scan/vessels_final.nrrd').unlink()
    (tmp_path / 'later').mkdir()
    rows, report = resumable_inventory(config, None, {}, out, resume=True, timeout=30)
    assert len(rows) == len(report) == 1
    with pytest.raises(ValueError, match='settings changed'):
        resumable_inventory(config, None, {'x': 'y'}, out, resume=True)


def test_inventory_timeout_leaves_restartable_state(tmp_path):
    from ronald.training.inventory import resumable_inventory
    config = {'mount': str(tmp_path), 'datasets': []}
    out = tmp_path / 'inventory.csv'
    with pytest.raises(TimeoutError, match='Completed checks saved'):
        resumable_inventory(config, None, {}, out, timeout=0.00001)
    assert out.with_suffix('.inventory-state').joinpath('settings.json').exists()
    assert not out.exists()


def _stall_one_scan(connection, config, read_mount, overrides, completed):
    import time
    if 'blocked' not in completed and not (Path(config['mount']) / 'recovered').exists():
        connection.send(('scan', dict(case_id='blocked', dataset='synthetic',
                                      scan='blocked', image='synthetic')))
        connection.send(('current', 'ctools DICOM discovery: synthetic'))
        time.sleep(30)
    elif 'blocked' not in completed:
        entry = dict(case_id='blocked', row={'case_id': 'blocked'},
                     report={'case_id': 'blocked', 'status': 'included'})
        connection.send(('completed', entry))
        completed['blocked'] = entry
    connection.send(('done', ([{'case_id': 'healthy'}],
                              [entry['report'] for entry in completed.values()])))
    connection.close()


def test_defer_stalled_scan_and_retry(tmp_path, monkeypatch):
    import ronald.training.inventory as inventory
    monkeypatch.setattr(inventory, '_inventory_worker', _stall_one_scan)
    config = {'mount': str(tmp_path), 'datasets': []}
    out = tmp_path / 'inventory.csv'
    rows, report = inventory.resumable_inventory(config, None, {}, out,
                                                timeout=2, skip_timeouts=True)
    assert rows == [{'case_id': 'healthy'}]
    assert report[0]['status'] == 'timeout'
    assert report[0]['stage'] == 'ctools DICOM discovery: synthetic'
    # Ordinary resume preserves exclusions; an explicit retry rechecks them.
    (tmp_path / 'recovered').touch()
    _, report = inventory.resumable_inventory(config, None, {}, out, resume=True, timeout=5)
    assert report[0]['status'] == 'timeout'
    _, report = inventory.resumable_inventory(config, None, {}, out, resume=True,
                                              timeout=5, retry_timeouts=True)
    assert report[0]['status'] == 'included'


def test_inventory_local_state_directory(tmp_path, monkeypatch):
    import ronald.training.inventory as inventory
    monkeypatch.setattr(inventory, '_inventory_worker', _stall_one_scan)
    (tmp_path / 'recovered').touch()
    local = tmp_path / 'local-state'
    out = tmp_path / 'unavailable-output-parent' / 'inventory.csv'
    config = {'mount': str(tmp_path), 'datasets': []}
    inventory.resumable_inventory(config, None, {}, out, timeout=5, state_dir=local)
    assert (local / 'case-blocked.json').exists()
    assert not out.parent.exists()
    inventory.resumable_inventory(config, None, {}, out, resume=True, timeout=5, state_dir=local)


def test_retry_missing_recovers_new_masks(tmp_path):
    from ronald.training.inventory import resumable_inventory
    (tmp_path / 'raw').mkdir()
    for scan in ['ready', 'later']:
        (tmp_path / 'raw' / (scan + '.nrrd')).touch()
        folder = tmp_path / 'masks' / scan
        folder.mkdir(parents=True)
        for name in ['lungs.nrrd', 'vessels_final.nrrd']:
            (folder / name).touch()
    (tmp_path / 'masks/ready/airways_final.nrrd').touch()
    config = {'mount': str(tmp_path), 'datasets': [
        dict(name='test', key='test', raw='raw', masks='masks',
             kind='volume', pattern='*.nrrd', identity_group='test')]}
    out = tmp_path / 'inventory.csv'
    rows, report = resumable_inventory(config, None, {}, out, timeout=30)
    assert len(rows) == 1
    (tmp_path / 'masks/later/airways_final.nrrd').touch()
    rows, _ = resumable_inventory(config, None, {}, out, resume=True, timeout=30)
    assert len(rows) == 1
    rows, report = resumable_inventory(config, None, {}, out, resume=True,
                                       timeout=30, retry_missing=True)
    assert len(rows) == 2
    assert all(entry['status'] == 'included' for entry in report)
