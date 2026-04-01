import os
import time

import numpy as np
import pandas as pd
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

from .wrapper_mmp_ad import run_MMPAD

METRIC_COLS = [
    'AUC-PR',
    'AUC-ROC',
    'VUS-PR',
    'VUS-ROC',
    'Standard-F1',
    'PA-F1',
    'Event-based-F1',
    'R-based-F1',
    'Affiliation-F',
]


def _sanitize_metric_value(value):
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        value_float = float(value)
        if not np.isfinite(value_float):
            return 0.0
        return value_float
    return value


def _load_data(file_path):
    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    return data, label


def run_submission_batch(dataset_dir, file_list, score_dir, save_dir,
                         ad_name, n_job, verbose, hp, run_kwargs=None):
    target_dir = os.path.join(score_dir, ad_name)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    run_kwargs = dict(run_kwargs or {})

    file_names = pd.read_csv(file_list)['file_name'].tolist()
    metric_csv_path = os.path.join(save_dir, f'{ad_name}.csv')
    csv_cols = ['file', 'Time'] + METRIC_COLS
    rows = []

    print('AD_Name:', ad_name)
    print('HP:', hp)
    print('dataset_dir:', dataset_dir)
    print('file_list:', file_list)
    print('score_dir:', score_dir)
    print('save_dir:', save_dir)
    print('n_job:', n_job)
    print('run_kwargs:', run_kwargs)

    for filename in file_names:
        print(f'Processing:{filename} by {ad_name}')
        file_path = os.path.join(dataset_dir, filename)
        score_path = os.path.join(
            target_dir,
            f'{os.path.splitext(filename)[0]}.npy',
        )

        data, label = _load_data(file_path)
        start_time = time.time()
        output = run_MMPAD(
            data,
            n_job=int(n_job),
            verbose=bool(verbose),
            **hp,
            **run_kwargs,
        )
        run_time = time.time() - start_time
        sliding_window = find_length_rank(data[:, 0], rank=1)
        metrics = get_metrics(output, label, slidingWindow=sliding_window)

        np.save(score_path, output)

        row = {
            'file': filename,
            'Time': run_time,
        }
        for metric_col in METRIC_COLS:
            row[metric_col] = _sanitize_metric_value(metrics.get(metric_col, 0))
        rows.append(row)
        pd.DataFrame(rows, columns=csv_cols).to_csv(metric_csv_path, index=False)
