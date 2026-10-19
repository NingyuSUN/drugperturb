"""Kaggle kernel K4 (CPU): truth-anchor check in already-opened contexts, before any v3 model is scored.

Recomputes measured deltas for A549 1 uM and the five v2 target contexts and
requires zero-change and frozen v1 per-compound RMSE to reproduce the committed
tables. Never-scored contexts are skipped; no new prediction is read.
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


def one(pattern):
    found = glob.glob(pattern, recursive=True)
    assert len(found) == 1, (pattern, found)
    return found[0].rsplit('/', 1)[0]


python = support.environment('core', 'locks/python-cpu.txt', '3.12.14', 'cpu')
support.run([python, 'scripts/v3_evaluate.py', '--anchors-only',
             '--extraction-root', one('/kaggle/input/**/extraction/output_hashes.tsv'),
             '--prepared', one('/kaggle/input/**/prepared/prepare_receipt.json'),
             '--output-root', support.WORKING/'anchors'], log='k4_anchors.log', env=support.python_env())
support.write_receipt('k4', dict(system=support.system_info(), code_archive=archive[0]))
print('K4 complete')
