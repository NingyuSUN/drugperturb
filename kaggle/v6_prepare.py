"""Kaggle kernel v6-P1 (CPU): merge extraction parts, prepare, seal the comparators, write the CPU-only report.

Everything here is CPU and response-free: the halves' truth, each key's DE set
and responder status, the sealed comparators and the report that decides gate G8
and decision D7. No arm is fitted and no evaluation-half test delta is computed.
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

roots = sorted({p.rsplit('/', 2)[0] for p in glob.glob('/kaggle/input/**/extraction/part_*/part_receipt.json',
                                                       recursive=True)})
assert roots, 'no extraction part is mounted'
python = support.environment('core', 'locks/python-cpu.txt', '3.12.14', 'cpu')
env = support.python_env()
support.run([python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_v6_pipeline.py',
             'tests/test_v6_endpoint.py', 'tests/test_v6_evaluate.py'], log='p1_tests.log', env=env)
support.run([python, 'scripts/v6_prepare.py', '--extraction-root', *roots,
             '--output-root', support.WORKING/'prepared'], log='p1_prepare.log', env=env)
support.run([python, 'scripts/v6_baselines.py', '--prepared', support.WORKING/'prepared',
             '--output-root', support.WORKING/'sealed'], log='p1_baselines.log', env=env)
support.run([python, 'scripts/v6_cheap_report.py', '--prepared', support.WORKING/'prepared',
             '--sealed', support.WORKING/'sealed',
             '--output', support.WORKING/'reports'/'cheap_report.md'], log='p1_cheap_report.log', env=env)
support.write_receipt('p1_prepare', dict(system=support.system_info(), extraction_roots=roots,
                                         code_archive=archive[0],
                                         code_archive_sha256=support.file_sha256(archive[0])))
print('V6 P1 complete')
