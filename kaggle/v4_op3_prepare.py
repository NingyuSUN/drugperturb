"""Kaggle kernel v4-O2 (CPU): response-free OP3 model inputs and sealed per-context baselines."""
import glob
import sys
import tarfile

archive = glob.glob('/kaggle/input/**/drugperturb-src.archive', recursive=True)
assert len(archive) == 1, archive
with tarfile.open(archive[0]) as tar:
    tar.extractall('/tmp/dp', filter='data')
sys.path.insert(0, '/tmp/dp/kaggle')
import support  # noqa: E402

extraction = glob.glob('/kaggle/input/**/extraction/extract_receipt.json', recursive=True)
assert len(extraction) == 1, extraction
root = extraction[0].rsplit('/', 1)[0]
python = support.environment('core', 'locks/python-cpu.txt', '3.12.14', 'cpu')
env = support.python_env()
support.run([python, 'scripts/v4_op3_prepare.py', '--extraction-root', root,
             '--output-root', support.WORKING/'prepared'], log='o2_prepare.log', env=env)
support.run([python, 'scripts/v4_context_baselines.py', '--prepared', support.WORKING/'prepared',
             '--output-root', support.WORKING/'sealed'], log='o2_context_baselines.log', env=env)
support.write_receipt('o2_prepare', dict(system=support.system_info(), extraction_root=root, code_archive=archive[0], code_archive_sha256=support.file_sha256(archive[0])))
print('O2 complete')
