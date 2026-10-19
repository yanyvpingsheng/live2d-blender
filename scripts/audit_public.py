"""Conservative publication check; run on a clean source checkout before tests.

Checks the reviewed manifest, binary/asset boundary and common secret formats.
Heuristics are defense-in-depth, not proof that arbitrary new content is safe.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
names = (ROOT / 'PUBLIC_FILES.txt').read_text(encoding='utf-8').splitlines()
assert names == sorted(set(names)), 'Manifest must be sorted and unique'
actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
          if p.is_file() and '.git' not in p.relative_to(ROOT).parts}
assert actual == set(names), {'unlisted': sorted(actual - set(names)),
                              'missing': sorted(set(names) - actual)}
patterns = (
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    r'\bgh[pousr]_[A-Za-z0-9]{20,}\b',
    r'\bgithub_pat_[A-Za-z0-9_]{20,}\b',
    r'\bAKIA[0-9A-Z]{16}\b',
    r'\bxox[baprs]-[A-Za-z0-9-]{20,}\b',
    r'\bsk-[A-Za-z0-9_-]{30,}\b',
    r'/workspace/(?:scratch|shared)/',
    r'/' + r'Users/[^/\s]+/',
    r'/' + r'home/(?!example/)[^/\s]+/',
)
pngs = {f'addon/live2d_blender/demo_assets/{color}.png'
        for color in ('dark', 'leaf', 'mint', 'peach')}
records = []
for name in names:
    rel = PurePosixPath(name)
    path = ROOT / name
    assert not rel.is_absolute() and '..' not in rel.parts and not path.is_symlink(), name
    assert path.suffix in ('', '.py', '.md', '.txt', '.json', '.cpp', '.png', '.yml'), name
    if path.suffix == '.png':
        assert name in pngs, name
        assert path.stat().st_size < 4096, name
    else:
        content = path.read_text(encoding='utf-8')
        for pattern in patterns:
            assert not re.search(pattern, content), f'Potential private data: {name}'
    records.append({'path': name, 'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
assert {r['path'] for r in records if r['path'].endswith('.cpp')} == {'native/physics_bridge.cpp'}
print(json.dumps({'passed': True, 'file_count': len(records),
                  'bytes': sum(r['bytes'] for r in records), 'files': records}, indent=2))
