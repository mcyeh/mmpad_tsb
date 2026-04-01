import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from mmpad_submission.submission_runner import run_submission_batch
from mmpad_submission.wrapper_mmp_ad import MMPAD_HP_M_0, MMPAD_HP_M_1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generating anomaly scores with MMPAD on multivariate data')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-M/')
    parser.add_argument('--file_list', type=str, default='../Datasets/File_List/TSB-AD-M-Eva.csv')
    parser.add_argument('--score_dir', type=str, default='eval/score/multi/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/multi/')
    parser.add_argument('--ad_name', type=str, default='MMPAD')
    parser.add_argument('--hp_version', type=int, default=0, choices=[0, 1])
    parser.add_argument('--n_job', type=int, default=1)
    parser.add_argument('--backend', dest='backend', type=str, default='cpu',
                        choices=['cpu', 'gpu'])
    parser.add_argument('--gpu_precision', dest='gpu_precision', type=str, default='float64',
                        choices=['float64', 'float32'])
    parser.add_argument('--gpu_execution', dest='gpu_execution', type=str, default='auto',
                        choices=['auto', 'gpu_pipeline', 'cpu_reference'])
    parser.add_argument('--gpu_device', dest='gpu_device', type=str, default=None)
    parser.add_argument('--gpu_reseed_period', dest='gpu_reseed_period', type=int, default=None)
    parser.add_argument('--gpu_allow_tf32', dest='gpu_allow_tf32', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.hp_version == 0:
        mmp_ad_hp = MMPAD_HP_M_0
    elif args.hp_version == 1:
        mmp_ad_hp = MMPAD_HP_M_1

    run_submission_batch(
        dataset_dir=args.dataset_dir,
        file_list=args.file_list,
        score_dir=args.score_dir,
        save_dir=args.save_dir,
        ad_name=args.ad_name,
        n_job=args.n_job,
        verbose=args.verbose,
        hp=mmp_ad_hp,
        run_kwargs={
            'backend': args.backend,
            'gpu_precision': args.gpu_precision,
            'gpu_execution': args.gpu_execution,
            'gpu_device': args.gpu_device,
            'gpu_reseed_period': args.gpu_reseed_period,
            'gpu_allow_tf32': args.gpu_allow_tf32,
        },
    )
