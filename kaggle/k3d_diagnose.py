"""Kaggle kernel K3d (CPU): training/validation-only chemCPA effect-size diagnostic.

Mounts only the response-free K2 outputs, so no test truth is reachable.
"""
import glob
import sys
import tarfile

archive = glob.glob('/kaggle/input/**/drugperturb-src.archive', recursive=True)
assert len(archive) == 1, archive
with tarfile.open(archive[0]) as tar:
    tar.extractall('/tmp/dp', filter='data')
sys.path.insert(0, '/tmp/dp/kaggle')
import support  # noqa: E402

prepared = glob.glob('/kaggle/input/**/prepared/prepare_receipt.json', recursive=True)
assert len(prepared) == 1, prepared
prepared = prepared[0].rsplit('/', 1)[0]
python = support.environment('cellmodels', 'locks/cellmodels-cpu.txt', '3.11', 'cpu')
env = support.python_env()
for candidate in ['C1_repository_default', 'C2_biolord_reproducibility']:
    support.run([python, 'scripts/v3_diagnose_chemcpa.py', '--prepared', prepared, '--candidate', candidate, '--seed', '42',
                 '--epochs', '200', '--output', support.WORKING/'diagnostics'/f'chemcpa_Q1_{candidate}_s42_200ep.json'],
                log=f'k3d_{candidate}.log', env=env)
support.write_receipt('k3d', dict(system=support.system_info(), code_archive=archive[0]))
print('K3d complete')
