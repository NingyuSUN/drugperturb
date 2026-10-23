"""Kaggle kernel v5-F (GPU T4 x2, internet): fit one half of the prespecified v5 arm/seed jobs.

push.py substitutes PART ('a' or 'b'). Jobs are ordered longest arm first
(I3, I2, I1, A0, seeds in protocol order) and alternate between parts, so the
two kernels carry similar loads. Each kernel first runs a one-epoch engineering
smoke fit of every arm it holds (outputs under smoke/, never sealed) and stops
if any fails; then two worker threads pin one fit at a time to each GPU. Inputs
are only the response-free P1 outputs; no test truth exists there. Sealing
happens after both parts finish (scripts/v5_seal.py).
"""
import glob
import queue
import subprocess
import sys
import tarfile
import threading

PART = '__PART__'

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
support.run([python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_v5_cellmodels.py',
             'tests/test_v4_cellmodels.py'], log=f'v5_fit_{PART}_tests.log', env=env)
protocol = yaml.safe_load(open('/tmp/dp/configs/protocol_v5.yaml'))
seeds = protocol['training']['seeds']
jobs = [(arm, seed) for arm in ['I3_nb_plus_pseudobulk', 'I2_counterfactual_pseudobulk', 'I1_nb_likelihood']
        for seed in seeds['candidates']]+[('A0_v3_reference', seed) for seed in seeds['A0']]
mine = jobs[0::2] if PART == 'a' else jobs[1::2]
smoke = list(dict.fromkeys(arm for arm, _ in mine))
failures = []


def work(items, gpu, root, extra, label):
    while True:
        try:
            arm, seed = items.get_nowait()
        except queue.Empty:
            return
        command = [python, 'scripts/v5_fit.py', '--arm', arm, '--seed', seed, '--prepared', prepared,
                   '--output-root', root, '--device', 'cuda', *extra]
        try:
            support.run(command, log=f'{label}_{arm}_s{seed}.log', env={**env, 'CUDA_VISIBLE_DEVICES': str(gpu)})
        except SystemExit as error:
            failures.append(str(error))


def run_all(pairs, root, extra, label):
    items = queue.Queue()
    for pair in pairs:
        items.put(pair)
    threads = [threading.Thread(target=work, args=(items, gpu, root, extra, label)) for gpu in range(len(gpus))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


gpus = [line for line in subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.splitlines()
        if line.startswith('GPU ')]
assert gpus, 'no GPU visible'
run_all([(arm, next(s for a, s in mine if a == arm)) for arm in smoke], support.WORKING/'smoke', ['--max-epochs', '1'],
        'smoke')
if failures:
    raise SystemExit('smoke fits failed; no full fit was started\n'+'\n'.join(failures))
run_all(mine, support.WORKING, [], 'fit')
if failures:
    raise SystemExit('\n'.join(failures))
support.write_receipt(f'v5_fit_{PART}', dict(system=support.system_info(), part=PART, jobs=mine, prepared=prepared,
                                             code_archive=archive[0], code_archive_sha256=support.file_sha256(archive[0])))
print(f'v5 fit part {PART} complete')
