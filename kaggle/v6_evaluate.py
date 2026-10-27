"""Kaggle kernel v6-V (CPU): score the sealed v6 predictions; the evaluation-half test truth is first computed here.

Mounts the E1 extraction, the P1 prepared inputs and comparator seal, and the
v6-S output (the arm seals and the selection, committed with the diagnostics
before this kernel runs). The evaluator verifies every seal hash before reading
any test response, and its anchors require zero change to score exactly 0.5 and
the truth fed back as a prediction to score 1.0.
"""
import glob
import sys
import tarfile
from pathlib import Path

archive = glob.glob('/kaggle/input/**/drugperturb-src.archive', recursive=True)
assert len(archive) == 1, archive
with tarfile.open(archive[0]) as tar:
    tar.extractall('/tmp/dp', filter='data')
sys.path.insert(0, '/tmp/dp/kaggle')
import support  # noqa: E402


def one(pattern):
    found = glob.glob(pattern, recursive=True)
    assert len(found) == 1, (pattern, found)
    return Path(found[0]).parent


extraction = one('/kaggle/input/**/extraction/part_*/part_receipt.json').parent
prepared = one('/kaggle/input/**/prepared/prepare_receipt.json')
comparators = one('/kaggle/input/**/sealed/baselines_v6_seal.json')
arms = one('/kaggle/input/**/selection_v6.json')
python = support.environment('core', 'locks/python-cpu.txt', '3.12.14', 'cpu')
env = support.python_env()
support.run([python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_v6_evaluate.py',
             'tests/test_v6_endpoint.py'], log='evaluate_tests.log', env=env)
support.run([python, 'scripts/v6_evaluate.py', '--extraction-root', extraction, '--prepared', prepared,
             '--sealed', arms, comparators, '--output-root', support.WORKING/'evaluation',
             '--manifests', support.CODE/'manifests/v6',
             '--protocol', support.CODE/'configs/protocol_v6.yaml'], log='evaluate.log', env=env)
support.write_receipt('v6_evaluate', dict(system=support.system_info(), extraction=str(extraction),
                                          prepared=str(prepared), arm_seals=str(arms),
                                          comparator_seal=str(comparators), code_archive=archive[0],
                                          code_archive_sha256=support.file_sha256(archive[0])))
print('V6 evaluation complete')
