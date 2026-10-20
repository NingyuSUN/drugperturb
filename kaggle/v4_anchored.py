"""Kaggle kernel v4-A1 (CPU): seal the vehicle-anchored baselines B1a/M1a (amendment A1) for OP3 and sci-Plex3.

Mounts only response-free prepare outputs and the sealed context baselines; no test truth is reachable.
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


def one(pattern, depth=1):
    found = glob.glob(pattern, recursive=True)
    assert len(found) == 1, (pattern, found)
    return found[0].rsplit('/', depth)[0]


python = support.environment('core', 'locks/python-cpu.txt', '3.12.14', 'cpu')
env = support.python_env()
jobs = [('op3', one('/kaggle/input/**/prepared/frozen_genes.tsv'),
         one('/kaggle/input/**/sealed/context_baselines_op3_seal.json')),
        ('sciplex_Q2', one('/kaggle/input/**/prepared/count_qc/compound_qualification.tsv', depth=2),
         one('/kaggle/input/**/sealed/context_baselines_Q2_seal.json'))]
for setting, prepared, baselines in jobs:
    support.run([python, 'scripts/v4_anchored_baselines.py', '--setting', setting, '--prepared', prepared,
                 '--baselines-root', baselines, '--output-root', support.WORKING/'sealed'],
                log=f'a1_{setting}.log', env=env)
support.write_receipt('a1_anchored', dict(system=support.system_info(), jobs=jobs, code_archive=archive[0],
                                          code_archive_sha256=support.file_sha256(archive[0])))
print('A1 anchored baselines complete')
