"""Kaggle kernel v4-K3 (GPU T4 x2, internet): fit every prespecified arm/seed of a dataset, then seal.

push.py substitutes DATASET ('op3' or 'sciplex') and SMOKE. Two worker threads pin
one fit at a time to each GPU. Inputs are only response-free prepare outputs
(v4 O2 for OP3, v3 K2 for sci-Plex3); no test truth exists there.
"""
import glob
import queue
import subprocess
import sys
import tarfile
import threading

DATASET = '__DATASET__'
SMOKE = __SMOKE__  # max epochs for an engineering smoke run, else None

archive = glob.glob('/kaggle/input/**/drugperturb-src.archive', recursive=True)
assert len(archive) == 1, archive
with tarfile.open(archive[0]) as tar:
    tar.extractall('/tmp/dp', filter='data')
sys.path.insert(0, '/tmp/dp/kaggle')
import support  # noqa: E402
import yaml  # noqa: E402

prepared = glob.glob('/kaggle/input/**/prepared/prepare_receipt.json', recursive=True)
assert len(prepared) == 1, prepared
prepared = prepared[0].rsplit('/', 1)[0]
python = support.environment('cellmodels', 'locks/cellmodels-cuda.txt', '3.11', 'cu121')
env = support.python_env()
support.run([python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_v4_cellmodels.py', 'tests/test_v4_core.py',
             'tests/test_v3_core.py'], log=f'k3_v4_{DATASET}_tests.log', env=env)
protocol = yaml.safe_load(open('/tmp/dp/configs/protocol_v4.yaml'))
settings = ['op3'] if DATASET == 'op3' else ['sciplex_Q2', 'sciplex_Q1']
# Longest fits first (pseudobulk arms, then larger settings) so the two GPUs finish together.
arms = ['I3_nb_plus_pseudobulk', 'I2_counterfactual_pseudobulk', 'I1_nb_likelihood', 'A0_v3_reference']
jobs = queue.Queue()
for setting in settings:
    for arm in arms:
        for seed in protocol['training']['seeds'][setting]:
            jobs.put((setting, arm, seed))
failures = []


def worker(gpu):
    while True:
        try:
            setting, arm, seed = jobs.get_nowait()
        except queue.Empty:
            return
        command = [python, 'scripts/v4_fit.py', '--setting', setting, '--arm', arm, '--seed', seed,
                   '--prepared', prepared, '--output-root', support.WORKING, '--device', 'cuda']
        if SMOKE:
            command += ['--max-epochs', SMOKE]
        try:
            support.run(command, log=f'fit_{arm}_{setting}_s{seed}.log', env={**env, 'CUDA_VISIBLE_DEVICES': str(gpu)})
        except SystemExit as error:
            failures.append(str(error))


gpus = [line for line in subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.splitlines()
        if line.startswith('GPU ')]
assert gpus, 'no GPU visible'
threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in range(len(gpus))]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
if failures:
    raise SystemExit('\n'.join(failures))
for setting in settings:
    command = [python, 'scripts/v4_seal.py', '--setting', setting, '--fit-root', support.WORKING/'fits',
               '--prepared', prepared, '--output-root', support.WORKING/'sealed']
    if SMOKE:
        command.append('--allow-smoke')
    support.run(command, log=f'seal_{setting}.log', env=env)
support.write_receipt(f'k3_v4_{DATASET}', dict(system=support.system_info(), smoke_max_epochs=SMOKE, prepared=prepared,
                                               code_archive=archive[0], code_archive_sha256=support.file_sha256(archive[0])))
print('K3 v4 complete')
