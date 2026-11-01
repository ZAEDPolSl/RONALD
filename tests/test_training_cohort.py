import csv
import pytest
from ronald.training.cohort import build, DATASETS
from ronald.training.inventory import FIELDS


def test_grouped_selection_and_resume(tmp_path):
    rows = []
    for dataset in (*DATASETS, 'Moltest 2'):
        for i in range(20):
            key = dataset + str(i)
            row = dict.fromkeys(FIELDS, '')
            row.update(case_id=key, patient_id=str(i), dataset=dataset,
                       image='/' + key, airways='/a', vessels='/v', lungs='/l')
            rows.append(row)
    path = tmp_path / 'source.csv'
    def save():
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    save()
    out = tmp_path / 'cohort'
    audit = build(path, out)
    assert audit['selected_rows'] == 60
    assert [audit['splits'][k]['scans'] for k in ('training', 'validation', 'test')] == [42, 9, 9]
    ids = {}
    for split in ('training', 'validation', 'test'):
        with (out / (split + '.csv')).open() as f:
            selected = list(csv.DictReader(f))
        ids[split] = {r['patient_id'] for r in selected}
        assert all(r['dataset'] in DATASETS and r['split'] == split for r in selected)
    assert not ids['training'] & ids['validation']
    assert not ids['test'] & (ids['training'] | ids['validation'])
    assert build(path, out, resume=True) == audit
    with pytest.raises(ValueError, match='differs'):
        build(path, out, seed=12, resume=True)
    rows.reverse()
    save()
    other = tmp_path / 'other'
    build(path, other)
    assert (out / 'inventory.csv').read_bytes() == (other / 'inventory.csv').read_bytes()
    rows.reverse()
    rows[1]['image'] = rows[0]['image']
    save()
    with pytest.raises(ValueError, match='Duplicate'):
        build(path, tmp_path / 'bad')
    rows[1]['image'] = '/unique'
    for i, row in enumerate(rows):
        row['split'] = ('training', 'validation', 'test')[i % 3]
    save()
    with pytest.raises(ValueError, match='Leakage'):
        build(path, tmp_path / 'bad')
