"""Kaggle kernel v6-E1 (CPU, internet): Tahoe-100M metadata, design check and v6 extraction of one shard range.

Downloads the pinned metadata files and verifies each against manifests/v6,
replays scripts/v6_design_audit.py --check, then streams the expression shards
of the range (each verified against manifests/v6/expression_source.tsv and
deleted after use) through scripts/v6_extract.py, which records the selection
and evaluation halves of every unit. No response is computed here.
"""
import glob
import subprocess
import sys
import tarfile
from pathlib import Path

SHARD_START, SHARD_STOP = '__SHARDS__'  # set by kaggle/push.py (default: all 3,388 shards in one part)

archive = glob.glob('/kaggle/input/**/drugperturb-src.archive', recursive=True)
assert len(archive) == 1, archive
with tarfile.open(archive[0]) as tar:
    tar.extractall('/tmp/dp', filter='data')
sys.path.insert(0, '/tmp/dp/kaggle')
import pandas as pd  # noqa: E402
import support  # noqa: E402

python = support.environment('core', 'locks/python-cpu.txt', '3.12.14', 'cpu')
env = support.python_env()
support.run([python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_v6_design.py',
             'tests/test_v6_pipeline.py', 'tests/test_v6_endpoint.py'], log='e1_tests.log', env=env)
manifests = support.CODE/'manifests/v6'
sources = pd.concat([pd.read_csv(manifests/'source.tsv', sep='\t'),
                     pd.read_csv(manifests/'expression_source.tsv', sep='\t')])
meta = Path('/tmp/meta')
meta.mkdir(parents=True, exist_ok=True)
fetched = {}
for row in sources[~sources.role.eq('expression_shard')].to_dict('records'):
    target = meta/Path(row['path']).name
    for attempt in range(5):
        subprocess.run(['curl', '-sSfL', '--retry', '5', '-o', str(target), row['url']], check=False)
        digest = support.file_sha256(target) if target.exists() else ''
        if digest == row['sha256'] and target.stat().st_size == int(row['bytes']):
            break
    assert digest == row['sha256'], f'{row["path"]}: SHA256 mismatch after retries'
    fetched[row['role']] = digest
support.run([python, 'scripts/v6_design_audit.py', '--check',
             '--receipt', support.WORKING/'receipts'/'v6_design_check.json'],
            log='e1_design_check.log', env=env)
support.run([python, 'scripts/v6_extract.py', '--obs', meta/'obs_metadata.parquet',
             '--gene-metadata', meta/'gene_metadata.parquet', '--download', '/tmp/shards',
             '--output-root', support.WORKING/'extraction', '--shard-start', SHARD_START,
             '--shard-stop', SHARD_STOP], log='e1_extract.log', env=env)
support.write_receipt('e1_extract', dict(system=support.system_info(), code_archive=archive[0],
                                         code_archive_sha256=support.file_sha256(archive[0]),
                                         metadata_sha256=fetched, shard_start=SHARD_START,
                                         shard_stop=SHARD_STOP))
print('V6 E1 complete')
