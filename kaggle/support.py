"""Helpers shared by the v3 Kaggle kernels: pinned environments, logged commands, receipts."""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE = Path('/tmp/dp')
WORKING = Path('/kaggle/working')
INDEXES = {'cpu': 'https://download.pytorch.org/whl/cpu', 'cu121': 'https://download.pytorch.org/whl/cu121'}


def run(command, *, log=None, env=None, cwd=CODE):
    """Run a command, tee output to a log under /kaggle/working/logs, fail loudly."""
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print('$', ' '.join(map(str, command)), flush=True)
    merged = {**os.environ, **(env or {})}
    with subprocess.Popen(list(map(str, command)), cwd=cwd, env=merged, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as process:
        lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            lines.append(line)
        code = process.wait()
    if log:
        path = WORKING/'logs'/log
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'# started {started}\n# {" ".join(map(str, command))}\n'+''.join(lines))
    if code:
        raise SystemExit(f'command failed with exit code {code}: {command}')
    return ''.join(lines)


def environment(name, lock, python, index):
    """Create an isolated venv from a lock file with uv; return its python path."""
    run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.12.9'], cwd='/tmp')
    uv = shutil.which('uv') or str(Path(sys.executable).parent/'uv')
    root = Path('/tmp/venvs')/name
    if not (root/'bin/python').exists():
        run([uv, 'venv', '-q', '--python', python, root], cwd='/tmp')
        run([uv, 'pip', 'install', '-q', '--python', root/'bin/python', '-r', CODE/lock,
             '--extra-index-url', INDEXES[index], '--index-strategy', 'unsafe-best-match'],
            log=f'env_{name}.log', cwd='/tmp')
    return root/'bin/python'


def python_env():
    return {'PYTHONPATH': str(CODE/'src'), 'PYTHONDONTWRITEBYTECODE': '1'}


def write_receipt(name, payload):
    path = WORKING/'receipts'/f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload, written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(payload, indent=2, default=str)+'\n')


def system_info():
    info = dict(python=sys.version, cpu_count=os.cpu_count())
    for key, command in [('gpu', ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
                         ('disk_tmp', ['df', '-h', '/tmp']), ('memory', ['free', '-g'])]:
        try:
            info[key] = subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            info[key] = f'unavailable: {error}'
    return info


def fetch(url, target, sha256):
    """Download with curl and verify SHA256; a mismatch is fatal."""
    import hashlib
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    run(['curl', '-sSfL', '--retry', '3', '-o', target, url], cwd='/tmp')
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise SystemExit(f'hash mismatch for {url}: {digest.hexdigest()}')
    return target
