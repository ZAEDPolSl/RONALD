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


def test_deduplicate_is_order_independent_and_preserves_distinct_repeat_scans():
    from ronald.training.finalize_splits import deduplicate_splits
    def case(i, patient, h, split):
        return dict(case_id=i, patient_id=patient, image=i, image_sha256=h, split=split, dataset='synthetic')
    cases = [case('b', 'p', 'same', 'training'), case('a', 'q', 'same', 'validation'),
             case('c', 'p', 'different', 'training'), case('d', 'r', 'train', 'training'),
             case('e', 's', 'test', 'test')]
    grouped, _, _ = group_splits(cases)
    result, removed = deduplicate_splits(grouped)
    assert [c['case_id'] for c in result['validation']] == ['a', 'c']
    assert removed[0]['retained_case_id'] == 'a'
    assert removed[0]['removed_case_id'] == 'b'
    assert sum(map(len, grouped.values())) == 5
    again, _ = deduplicate_splits({k: list(reversed(v)) for k, v in grouped.items()})
    assert again == result
