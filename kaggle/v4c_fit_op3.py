"""Kaggle kernel v4c (GPU T4 x2, internet): post hoc corrected OP3 fits, seal and selection.

Same as v4_fit.py for OP3, except that every fit and the seal run with
--descriptor-tolerance (configs/v4_corrected_posthoc.yaml). Inputs are only the
response-free v4 O2 prepare outputs; no test truth exists there.
"""
import glob
import queue
import subprocess
import sys
import tarfile
import threading

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
             'tests/test_v3_core.py', 'tests/test_descriptor_scaling.py'], log='k3_v4c_op3_tests.log', env=env)
protocol = yaml.safe_load(open('/tmp/dp/configs/protocol_v4.yaml'))
plan = yaml.safe_load(open('/tmp/dp/configs/v4_corrected_posthoc.yaml'))
assert plan['status'] == 'POST_HOC_PLAN_BEFORE_CORRECTED_FITS', plan['status']
# Longest fits first (pseudobulk arms) so the two GPUs finish together, as in v4.
arms = ['I3_nb_plus_pseudobulk', 'I2_counterfactual_pseudobulk', 'I1_nb_likelihood', 'A0_v3_reference']
jobs = queue.Queue()
for arm in arms:
    for seed in protocol['training']['seeds']['op3']:
        jobs.put((arm, seed))
failures = []


def worker(gpu):
    while True:
        try:
            arm, seed = jobs.get_nowait()
        except queue.Empty:
            return
        command = [python, 'scripts/v4_fit.py', '--setting', 'op3', '--arm', arm, '--seed', seed,
                   '--prepared', prepared, '--output-root', support.WORKING, '--device', 'cuda', '--descriptor-tolerance']
        try:
            support.run(command, log=f'fit_{arm}_op3_s{seed}.log', env={**env, 'CUDA_VISIBLE_DEVICES': str(gpu)})
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
support.run([python, 'scripts/v4_seal.py', '--setting', 'op3', '--fit-root', support.WORKING/'fits', '--prepared', prepared,
             '--output-root', support.WORKING/'sealed', '--descriptor-tolerance'], log='seal_op3.log', env=env)
support.write_receipt('k3_v4c_op3', dict(system=support.system_info(), analysis=plan['label'],
                                         descriptor_min_std=plan['change']['descriptor_min_std'], prepared=prepared,
                                         code_archive=archive[0], code_archive_sha256=support.file_sha256(archive[0])))
print('K3 v4c complete')
