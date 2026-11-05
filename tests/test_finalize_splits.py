from ronald.training.finalize_splits import group_splits


def test_transitive_duplicate_groups_preserve_holdout():
    def case(i, patient, image_hash, split):
        return dict(case_id=str(i), patient_id=patient, image=str(i), image_sha256=image_hash, split=split)
    cases = [case(0, 'a', 'x', 'training'), case(1, 'b', 'x', 'validation'),
             case(2, 'b', 'y', 'test'), case(3, 'c', 'z', 'training'),
             case(4, 'd', 'w', 'validation')]
    splits, changes, groups = group_splits(cases)
    assert [c['case_id'] for c in splits['test']] == ['0', '1', '2']
    assert len(changes) == 2
    assert cases[0]['split'] == 'training'
