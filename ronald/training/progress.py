"""Per-cohort preparation progress without changing inventory order."""
from collections import Counter
from contextlib import ExitStack
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def dataset_progress(rows):
    totals = Counter(row['dataset'] for row in rows)
    with ExitStack() as stack:
        stack.enter_context(logging_redirect_tqdm())
        bars = {name: stack.enter_context(tqdm(total=count, desc=f'Prepare: {name}',
                                               unit='scan', position=i, dynamic_ncols=True))
                for i, (name, count) in enumerate(totals.items())}
        for row in rows:
            yield row
            bars[row['dataset']].update(1)
