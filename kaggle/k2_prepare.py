"""Kaggle kernel K2 (CPU, internet): count QC, response-free model inputs, context baselines."""
import glob
import sys
import tarfile

archive = glob.glob('/kaggle/input/**/drugperturb-src.archive', recursive=True)
assert len(archive) == 1, archive
with tarfile.open(archive[0]) as tar:
    tar.extractall('/tmp/dp', filter='data')
sys.path.insert(0, '/tmp/dp/kaggle')
import support  # noqa: E402

extraction = glob.glob('/kaggle/input/**/extraction/output_hashes.tsv', recursive=True)
assert len(extraction) == 1, extraction
root = extraction[0].rsplit('/', 1)[0]
python = support.environment('core', 'locks/python-cpu.txt', '3.12.14', 'cpu')
env = support.python_env()
support.run([python, '-m', 'pytest', '-q', 'tests/test_v3_core.py'], log='k2_tests.log', env=env)
support.run([python, 'scripts/v3_prepare.py', '--extraction-root', root,
             '--output-root', support.WORKING/'prepared'], log='k2_prepare.log', env=env)
support.run([python, 'scripts/v3_context_baselines.py', '--prepared', support.WORKING/'prepared',
             '--output-root', support.WORKING/'sealed'], log='k2_context_baselines.log', env=env)
support.write_receipt('k2', dict(system=support.system_info(), extraction_root=root,
                                 code_archive=archive[0]))
print('K2 complete')
