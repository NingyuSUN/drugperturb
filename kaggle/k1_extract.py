"""Kaggle kernel K1 (CPU, internet): fetch sci-Plex3 RDS sources, verify SHA256, export counts.

Installs conda-forge R 4.3.1 (the v2 extraction runtime) with micromamba, runs
the R tests, then the all-gene pseudobulk and frozen-gene cell exporters for each
cell type. No treatment effect is computed here.
"""
import csv
import glob
import hashlib
import sys
import tarfile
from pathlib import Path

archive = glob.glob('/kaggle/input/**/drugperturb-src.archive', recursive=True)
assert len(archive) == 1, archive
with tarfile.open(archive[0]) as tar:
    tar.extractall('/tmp/dp', filter='data')
sys.path.insert(0, '/tmp/dp/kaggle')
import support  # noqa: E402

support.run(['bash', '-c', 'curl -sSfL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp bin/micromamba'],
            cwd='/tmp')
support.run(['/tmp/bin/micromamba', 'create', '-y', '-q', '-r', '/tmp/mamba', '-p', '/tmp/r431', '-c', 'conda-forge',
             'r-base=4.3.1'], cwd='/tmp', log='k1_r_env.log')
rscript = '/tmp/r431/bin/Rscript'
version = support.run([rscript, '--version'])
for test in ['tests/test_context_export.R', 'tests/test_v3_export_cells.R']:
    support.run([rscript, test], log=f'k1_{Path(test).stem}.log')
out = support.WORKING/'extraction'
downloads = []
for row in csv.DictReader(open('/tmp/dp/manifests/v3/source_files.tsv'), delimiter='\t'):
    target = support.fetch(row['url'], f'/tmp/raw/{row["filename"]}', row['sha256'])
    if target.stat().st_size != int(row['bytes']):
        raise SystemExit(f'size mismatch for {row["filename"]}')
    downloads.append(dict(filename=row['filename'], bytes=target.stat().st_size, sha256=row['sha256']))
    cell = row['cell_type']
    support.run([rscript, 'scripts/export_context_rds.R', target, 'manifests/v2/well_mapping.tsv',
                 out/'pseudobulk'/f'{cell}_24h', cell, '24'], log=f'k1_pseudobulk_{cell}.log')
    support.run([rscript, 'scripts/v3_export_cells.R', target, 'manifests/v2/well_mapping.tsv',
                 'manifests/selected_genes.tsv', out/'cells'/f'{cell}_24h', cell, '24'], log=f'k1_cells_{cell}.log')
    target.unlink()
with open(out/'source_downloads.tsv', 'w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['filename', 'bytes', 'sha256'], delimiter='\t')
    writer.writeheader()
    writer.writerows(downloads)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


rows = [dict(path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=digest(p))
        for p in sorted(out.rglob('*')) if p.is_file()]
with open(out/'output_hashes.tsv', 'w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['path', 'bytes', 'sha256'], delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)
support.write_receipt('k1', dict(system=support.system_info(), R=version.strip(), code_archive=archive[0],
                                 code_archive_sha256=digest(archive[0]), downloads=downloads))
print('K1 complete')
