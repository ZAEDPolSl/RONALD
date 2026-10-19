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
