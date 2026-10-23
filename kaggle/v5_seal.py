"""Kaggle kernel v5-S (CPU): seal the v5 arms, select on validation, tabulate diagnostics; no test truth is read.

Mounts both fit kernels' outputs (full fits only; the one-epoch smoke fits under
smoke/ are ignored) and the P1 outputs (prepared inputs and the comparator seal).
The seals, the selection and the diagnostics are committed before the
evaluation kernel, which mounts this kernel's output, reads any test response.
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


prepared = one('/kaggle/input/**/prepared/prepare_receipt.json')
comparators = one('/kaggle/input/**/baselines_v5_seal.json')
fit_roots = sorted({Path(p).parents[1] for p in glob.glob('/kaggle/input/**/fits/*_v5_s*.json', recursive=True)
                    if '/smoke/' not in p})
assert len(fit_roots) == 2, fit_roots
python = support.environment('cellmodels', 'locks/cellmodels-cpu.txt', '3.11', 'cpu')
env = support.python_env()
support.run([python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_v5_cellmodels.py',
             'tests/test_v5_selection.py'], log='v5_seal_tests.log', env=env)
support.run([python, 'scripts/v5_seal.py', '--fit-root', *fit_roots, '--prepared', prepared,
             '--output-root', support.WORKING/'sealed'], log='v5_seal.log', env=env)
support.run([python, 'scripts/v5_diagnostics_report.py', '--sealed-root', support.WORKING/'sealed', comparators,
             '--prepared', prepared, '--output-root', support.WORKING/'diagnostics'], log='v5_diagnostics.log', env=env)
support.write_receipt('v5_seal', dict(system=support.system_info(), fit_roots=[str(r) for r in fit_roots],
                                      prepared=str(prepared), comparator_seal=str(comparators), code_archive=archive[0],
                                      code_archive_sha256=support.file_sha256(archive[0])))
print('v5 seal complete')
