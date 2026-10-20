"""Kaggle kernel v4-K5 (CPU): score sealed v4 predictions (OP3 confirmatory, sci-Plex3 exploratory)."""
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


op3_extraction = one('/kaggle/input/**/extraction/extract_receipt.json')
op3_prepared = one('/kaggle/input/**/prepared/frozen_genes.tsv')
sciplex_extraction = one('/kaggle/input/**/extraction/output_hashes.tsv')
sciplex_prepared = one('/kaggle/input/**/prepared/count_qc/compound_qualification.tsv', depth=2)
sealed = sorted({p.rsplit('/', 1)[0] for p in glob.glob('/kaggle/input/**/sealed/*_seal.json', recursive=True)})
python = support.environment('core', 'locks/python-cpu.txt', '3.12.14', 'cpu')
env = support.python_env()
roots = [arg for root in sealed for arg in ['--sealed-root', root]]
support.run([python, 'scripts/v4_evaluate_op3.py', '--extraction-root', op3_extraction, '--prepared', op3_prepared,
             '--output-root', support.WORKING/'reports_v4/op3', *roots], log='k5_v4_op3.log', env=env)
support.run([python, 'scripts/v4_evaluate_sciplex.py', '--extraction-root', sciplex_extraction,
             '--prepared', sciplex_prepared, '--output-root', support.WORKING/'reports_v4/sciplex', *roots],
            log='k5_v4_sciplex.log', env=env)
support.write_receipt('k5_v4', dict(system=support.system_info(), sealed_roots=sealed, code_archive=archive[0], code_archive_sha256=support.file_sha256(archive[0])))
print('K5 v4 complete')
