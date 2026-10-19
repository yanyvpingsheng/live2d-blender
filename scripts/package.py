"""Build an installable ZIP from the explicit reviewed public source manifest.

No filesystem walk of model inputs, SDK installs or generated data is used.
Run the public-tree audit before publishing source. Local dist/ is not source.
"""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.3.0'
names = (ROOT / 'PUBLIC_FILES.txt').read_text(encoding='utf-8').splitlines()
addon_names = [name for name in names if name.startswith('addon/live2d_blender/')]
assert 'addon/live2d_blender/__init__.py' in addon_names
assert 'addon/live2d_blender/LICENSE' in addon_names
assert len(addon_names) == len(set(addon_names))
dist = ROOT / 'dist'
dist.mkdir(exist_ok=True)
target = dist / f'live2d_offline_converter-{VERSION}.zip'
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for name in sorted(addon_names):
        path = ROOT / name
        assert not path.is_symlink() and path.is_file(), name
        assert path.suffix in ('.py', '.md', '.png', ''), name
        entry = zipfile.ZipInfo(str(path.relative_to(ROOT / 'addon')), (2026, 10, 19, 0, 0, 0))
        entry.create_system = 3
        entry.external_attr = 0o100644 << 16
        archive.writestr(entry, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
print(json.dumps({'file': target.name, 'bytes': target.stat().st_size,
                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                  'entries': len(addon_names)}, indent=2))
