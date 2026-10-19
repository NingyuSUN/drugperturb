"""Kaggle kernel K3 (GPU T4 x2, internet): fit every prespecified candidate/seed, then seal.

push.py substitutes METHOD and SMOKE. Two worker threads pin one fit at a time to
each GPU. Inputs are only the response-free K2 outputs; no test truth exists there.
"""
import glob
import queue
import sys
import tarfile
import threading

METHOD = '__METHOD__'
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
support.run([python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_v3_cellmodels.py',
             'tests/test_v3_core.py'], log=f'k3_{METHOD}_tests.log', env=env)
protocol = yaml.safe_load(open('/tmp/dp/configs/protocol_v3.yaml'))
spec = protocol['models']['chemCPA' if METHOD == 'chemcpa' else 'biolord']
jobs = queue.Queue()
# Longest fits first so the two GPUs finish together.
for q in ['Q2', 'Q1']:
    for candidate in sorted(spec['candidates'], reverse=True):
        for seed in protocol['models']['seeds']:
            jobs.put((q, candidate, seed))
failures = []


def worker(gpu):
    while True:
        try:
            q, candidate, seed = jobs.get_nowait()
        except queue.Empty:
            return
        command = [python, 'scripts/v3_fit.py', '--method', METHOD, '--protocol', q, '--candidate', candidate,
                   '--seed', seed, '--prepared', prepared, '--output-root', support.WORKING, '--device', 'cuda']
        if SMOKE:
            command += ['--max-epochs', SMOKE]
        try:
            support.run(command, log=f'fit_{METHOD}_{q}_{candidate}_s{seed}.log',
                        env={**env, 'CUDA_VISIBLE_DEVICES': str(gpu)})
        except SystemExit as error:
            failures.append(str(error))


import subprocess  # noqa: E402
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
for q in ['Q1', 'Q2']:
    command = [python, 'scripts/v3_seal.py', '--method', METHOD, '--protocol', q, '--fit-root',
               support.WORKING/'fits', '--prepared', prepared, '--output-root', support.WORKING/'sealed']
    if SMOKE:
        command.append('--allow-smoke')
    support.run(command, log=f'seal_{METHOD}_{q}.log', env=env)
support.write_receipt(f'k3_{METHOD}', dict(system=support.system_info(), smoke_max_epochs=SMOKE,
                                          prepared=prepared, code_archive=archive[0]))
print('K3 complete')
