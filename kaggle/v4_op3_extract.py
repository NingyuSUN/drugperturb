"""Kaggle kernel v4-O1 (CPU, internet): fetch OP3, record/verify SHA256 and size, check design, extract.

No validation or test response is computed here. The h5ad is deleted after use.
"""
import glob
import hashlib
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

archive = glob.glob('/kaggle/input/**/drugperturb-src.archive', recursive=True)
assert len(archive) == 1, archive
with tarfile.open(archive[0]) as tar:
    tar.extractall('/tmp/dp', filter='data')
sys.path.insert(0, '/tmp/dp/kaggle')
import support  # noqa: E402
import yaml  # noqa: E402

source = yaml.safe_load(open('/tmp/dp/configs/protocol_v4.yaml'))['datasets']['op3']['source']
python = support.environment('core', 'locks/python-cpu.txt', '3.12.14', 'cpu')
env = support.python_env()
support.run([python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_v4_core.py'], log='o1_tests.log', env=env)
target = Path(f'/tmp/raw/{source["file"]}')
target.parent.mkdir(parents=True, exist_ok=True)
attempts = []
for attempt in range(20):  # NCBI drops long TLS transfers; resume until the pinned size is reached
    result = subprocess.run(['curl', '-sSL', '--retry', '5', '-C', '-', '-o', str(target), source['url']],
                            capture_output=True, text=True)
    attempts.append(dict(exit=result.returncode, bytes=target.stat().st_size if target.exists() else 0,
                         stderr=result.stderr.strip()[-300:]))
    print(attempts[-1], flush=True)
    if target.exists() and target.stat().st_size >= int(source['bytes']):
        break
    time.sleep(10)
digest = hashlib.sha256()
with target.open('rb') as stream:
    for block in iter(lambda: stream.read(8 << 20), b''):
        digest.update(block)
sha256 = digest.hexdigest()
assert target.stat().st_size == int(source['bytes']), 'OP3 size differs from the pinned Content-Length'
if re.fullmatch(r'[0-9a-f]{64}', str(source['sha256'])):
    assert sha256 == source['sha256'], f'OP3 SHA256 mismatch: {sha256}'
print('OP3 sha256', sha256, flush=True)
support.run([python, 'scripts/v4_op3_design.py', '--h5ad', target, '--check',
             '--receipt', support.WORKING/'receipts'/'op3_design_check.json'], log='o1_design_check.log', env=env)
support.run([python, 'scripts/v4_op3_extract.py', '--h5ad', target, '--output-root', support.WORKING/'extraction'],
            log='o1_extract.log', env=env)
target.unlink()
support.write_receipt('o1_extract', dict(system=support.system_info(), code_archive=archive[0], code_archive_sha256=support.file_sha256(archive[0]),
                                         source=dict(url=source['url'], bytes=int(source['bytes']), sha256=sha256,
                                                     protocol_sha256=source['sha256'], download_attempts=attempts)))
print('O1 complete')
