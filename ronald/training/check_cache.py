"""Inspect completed-cache metadata without changing files or reading large arrays."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .preparation_state import output_mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--case-id', help='Inspect one scan; omit to inspect all completion records')
    args = parser.parse_args()
    if not args.cache.is_dir():
        parser.error('Cache directory does not exist')
    if args.case_id and Path(args.case_id).name != args.case_id:
        parser.error('case-id must be a directory name')
    markers = ([args.cache / args.case_id / 'complete.json'] if args.case_id else
               args.cache.glob('*/complete.json'))
    counts = Counter()
    checked = 0
    for marker in markers:
        checked += 1
        try:
            saved = json.loads(marker.read_text())
            issues = output_mismatches(marker.parent, saved['outputs'])
        except (OSError, ValueError, KeyError) as exc:
            issues = [dict(reason='inspection_error', error=str(exc))]
        for issue in issues:
            counts[issue['reason']] += 1
        if issues:
            print(json.dumps(dict(case_id=marker.parent.name, issues=issues)), flush=True)
    print(json.dumps(dict(completion_records_checked=checked, issues=dict(counts))))


if __name__ == '__main__':
    main()
