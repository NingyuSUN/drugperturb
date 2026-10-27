"""Kaggle kernel v6-S (CPU): seal the v6 arms, select on validation, write the diagnostics; no test truth is read.

Mounts the fit kernel's output (full fits only; the one-epoch smoke fits live
under /tmp in that kernel and are never written out) and the P1 outputs. Under
decision D7 only M1 was fitted, at three seeds, so --arms and --seeds name the
reduced plan and the selection record states it (amendment A3).

The seals, the selection and the diagnostics are committed before the
evaluation kernel, which is the first thing to read a test response.
"""
import glob
import sys
import tarfile
from pathlib import Path

ARMS, SEEDS = '__ARMS__', '__SEEDS__'  # set by kaggle/push.py

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
comparators = one('/kaggle/input/**/sealed/baselines_v6_seal.json')
fit_roots = sorted({Path(p).parents[1] for p in glob.glob('/kaggle/input/**/fits/*_v6_s*.json', recursive=True)})
assert fit_roots, 'no v6 fit is mounted'
python = support.environment('cellmodels', 'locks/cellmodels-cpu.txt', '3.11', 'cpu')
env = support.python_env()
support.run([python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_v6_cellmodels.py'],
            log='seal_tests.log', env=env)
support.run([python, 'scripts/v6_seal.py', '--fit-root', *fit_roots, '--prepared', prepared,
             '--output-root', support.WORKING/'sealed', '--arms', *ARMS, '--seeds', *map(str, SEEDS),
             '--protocol', support.CODE/'configs/protocol_v6.yaml'], log='seal.log', env=env)
support.run([python, 'scripts/v6_diagnostics_report.py', '--prepared', prepared,
             '--sealed', support.WORKING/'sealed', comparators,
             '--output', support.WORKING/'reports'/'diagnostics.md',
             '--protocol', support.CODE/'configs/protocol_v6.yaml'], log='diagnostics.log', env=env)
support.write_receipt('v6_seal', dict(system=support.system_info(), fit_roots=[str(r) for r in fit_roots],
                                      prepared=str(prepared), comparator_seal=str(comparators), arms=list(ARMS),
                                      seeds=list(SEEDS), code_archive=archive[0],
                                      code_archive_sha256=support.file_sha256(archive[0])))
print('V6 seal complete')
