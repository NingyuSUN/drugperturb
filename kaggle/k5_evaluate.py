"""Kaggle kernel K5 (CPU): score sealed predictions; anchors truth to published v1/v2 results."""
import glob
import sys
import tarfile

archive = glob.glob('/kaggle/input/**/drugperturb-src.archive', recursive=True)
assert len(archive) == 1, archive
with tarfile.open(archive[0]) as tar:
    tar.extractall('/tmp/dp', filter='data')
sys.path.insert(0, '/tmp/dp/kaggle')
import support  # noqa: E402


def one(pattern):
    found = glob.glob(pattern, recursive=True)
    assert len(found) == 1, (pattern, found)
    return found[0].rsplit('/', 1)[0]


extraction = one('/kaggle/input/**/extraction/output_hashes.tsv')
prepared = one('/kaggle/input/**/prepared/prepare_receipt.json')
sealed = sorted({p.rsplit('/', 1)[0] for p in glob.glob('/kaggle/input/**/sealed/*_seal.json', recursive=True)})
python = support.environment('core', 'locks/python-cpu.txt', '3.12.14', 'cpu')
command = [python, 'scripts/v3_evaluate.py', '--extraction-root', extraction, '--prepared', prepared,
           '--output-root', support.WORKING/'reports_v3']
for root in sealed:
    command += ['--sealed-root', root]
support.run(command, log='k5_evaluate.log', env=support.python_env())
support.write_receipt('k5', dict(system=support.system_info(), sealed_roots=sealed, code_archive=archive[0]))
print('K5 complete')
