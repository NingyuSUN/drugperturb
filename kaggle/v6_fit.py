"""Kaggle kernel v6-fit (T4 x2): the v6 arm fits for one arm and its planned seeds; never reads a test outcome.

Decision D7, taken on the CPU-only report, cut the GPU budget to M1 at three
seeds (amendment A3), so ARM and SEEDS are set by kaggle/push.py. Each seed runs
a one-epoch smoke fit first and the kernel stops if one fails, so a broken
configuration costs minutes and not hours.
"""
import glob
import sys
import tarfile
import time

ARM, SEEDS = '__ARM__', '__SEEDS__'  # set by kaggle/push.py

archive = glob.glob('/kaggle/input/**/drugperturb-src.archive', recursive=True)
assert len(archive) == 1, archive
with tarfile.open(archive[0]) as tar:
    tar.extractall('/tmp/dp', filter='data')
sys.path.insert(0, '/tmp/dp/kaggle')
import support  # noqa: E402

prepared = sorted(glob.glob('/kaggle/input/**/prepared/prepare_receipt.json', recursive=True))
sealed = sorted(glob.glob('/kaggle/input/**/sealed/baselines_v6_predictions.npz', recursive=True))
assert len(prepared) == 1 and len(sealed) == 1, (prepared, sealed)
prepared, baselines = prepared[0].rsplit('/', 1)[0], sealed[0]
python = support.environment('cellmodels', 'locks/cellmodels-cuda.txt', '3.11', 'cu121')
env = support.python_env()
support.run([python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_v6_cellmodels.py'],
            log='fit_tests.log', env=env)

common = ['--prepared', prepared, '--baselines', baselines, '--manifests', support.CODE/'manifests/v6',
          '--protocol', support.CODE/'configs/protocol_v6.yaml', '--device', 'cuda']
for seed in SEEDS:  # a one-epoch smoke fit per seed before the real one
    support.run([python, 'scripts/v6_fit.py', '--arm', ARM, '--seed', seed, *common,
                 '--output-root', f'/tmp/smoke_{seed}', '--max-epochs', '1'],
                log=f'smoke_{ARM}_{seed}.log', env=env)
print(f'smoke fits passed for {ARM} seeds {list(SEEDS)}', flush=True)

started = time.perf_counter()
for seed in SEEDS:
    support.run([python, 'scripts/v6_fit.py', '--arm', ARM, '--seed', seed, *common,
                 '--output-root', support.WORKING], log=f'fit_{ARM}_{seed}.log', env=env)
    print(f'{ARM} seed {seed} done at {round((time.perf_counter()-started)/60, 1)} minutes', flush=True)
support.write_receipt(f'fit_{ARM}', dict(system=support.system_info(), arm=ARM, seeds=list(SEEDS),
                                         prepared=prepared, baselines=baselines, code_archive=archive[0],
                                         code_archive_sha256=support.file_sha256(archive[0]),
                                         minutes=round((time.perf_counter()-started)/60, 1)))
print('V6 fits complete')
