"""Safe model3.json manifest loading; moc3 is always opaque binary."""
import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from .data import Live2DError, finite

MAX_JSON_BYTES = 32 * 1024 * 1024


def read_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise Live2DError(f'JSON exceeds {MAX_JSON_BYTES // 1024 // 1024} MiB: {path.name}')
    try:
        with path.open('r', encoding='utf-8-sig') as handle:
            data = json.load(handle, parse_float=lambda x: finite(float(x), 'JSON number'), parse_constant=lambda x: (_ for _ in ()).throw(Live2DError(f'Invalid JSON number: {x}')))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Live2DError(f'Invalid JSON {path.name}: {error}') from error
    if not isinstance(data, dict):
        raise Live2DError(f'{path.name}: expected JSON object')
    return data


def resolve_asset(base, name):
    if not isinstance(name, str) or not name or '\0' in name:
        raise Live2DError('Asset reference must be a nonempty string')
    normalized = name.replace('\\', '/')
    if Path(normalized).is_absolute() or PureWindowsPath(name).drive or '://' in name:
        raise Live2DError(f'Absolute or network asset references are not allowed: {name}')
    base = Path(base).resolve()
    path = (base / normalized).resolve()
    if not path.is_relative_to(base):
        raise Live2DError(f'Asset escapes model directory: {name}')
    return path


@dataclass
class ModelPackage:
    path: Path
    raw: dict
    moc: Path
    textures: list
    references: dict
    missing: list

    @property
    def name(self):
        return self.path.name.removesuffix('.model3.json')

    def require_model(self):
        if not self.moc.is_file():
            raise Live2DError(f'Missing moc3 binary: {self.moc.name}')
        for texture in self.textures:
            if not texture.is_file():
                raise Live2DError(f'Missing texture: {texture.name}')


def load_model3(path):
    path = Path(path).expanduser().resolve()
    raw = read_json(path)
    if raw.get('Version') != 3:
        raise Live2DError('Only Cubism model3.json Version 3 is supported; not Cubism 2 or editor cmo3')
    refs = raw.get('FileReferences')
    if not isinstance(refs, dict) or not isinstance(refs.get('Moc'), str):
        raise Live2DError('FileReferences.Moc is required')
    textures = refs.get('Textures', [])
    if not isinstance(textures, list):
        raise Live2DError('FileReferences.Textures must be an array')
    paths = {'Moc': resolve_asset(path.parent, refs['Moc'])}
    texture_paths = [resolve_asset(path.parent, p) for p in textures]
    for key in ('Physics', 'Pose', 'DisplayInfo', 'UserData'):
        if refs.get(key):
            paths[key] = resolve_asset(path.parent, refs[key])
    expressions = refs.get('Expressions', [])
    if not isinstance(expressions, list):
        raise Live2DError('Expressions must be an array')
    for index, entry in enumerate(expressions):
        if not isinstance(entry, dict) or 'File' not in entry:
            raise Live2DError('Expression entry must have File')
        paths[f'Expression:{index}'] = resolve_asset(path.parent, entry['File'])
    motions = refs.get('Motions', {})
    if not isinstance(motions, dict):
        raise Live2DError('Motions must be an object')
    for group, entries in motions.items():
        if not isinstance(entries, list):
            raise Live2DError(f'Motion group {group} must be an array')
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or 'File' not in entry:
                raise Live2DError('Motion entry must have File')
            paths[f'Motion:{group}:{index}'] = resolve_asset(path.parent, entry['File'])
            if entry.get('Sound'):
                paths[f'Sound:{group}:{index}'] = resolve_asset(path.parent, entry['Sound'])
    missing = [str(p) for p in [*paths.values(), *texture_paths] if not p.is_file()]
    return ModelPackage(path, raw, paths['Moc'], texture_paths, paths, missing)
