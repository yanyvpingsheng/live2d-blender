"""Optional official Framework physics/pose adapter, not an approximate solver.

The user explicitly selects a locally built native bridge. The add-on does not
download, accept licenses for, or redistribute SDK code/binaries. The bridge
owns an independent Core model, so the mesh backend's memory is never borrowed.

R5 late update: motion -> blink -> expression -> look -> breath -> physics ->
lipsync -> pose -> Core update. This module handles physics/pose only. Callers
must replay sampled animation in small steps when seeking or scrubbing.
"""
import ctypes as C
from decimal import Decimal
import json
import math
from pathlib import Path
import threading

from .data import Live2DError

P, I, U, F = C.c_void_p, C.c_int, C.c_uint, C.c_float
PF = C.POINTER(F)
MAX_JSON = 32 * 1024 * 1024
R5_UPDATE_ORDER = ('motion', 'eye_blink', 'expression', 'look', 'breath',
                   'physics', 'lip_sync', 'pose', 'core_update')


def _number(value, label, minimum=-1e12, maximum=1e12):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise Live2DError(f'{label} must be numeric')
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise Live2DError(f'{label} must be finite and inside {minimum}..{maximum}')
    return float(value)


def _id(value, label):
    if not isinstance(value, str) or not value or '\0' in value or len(value.encode('utf-8')) > 4096:
        raise Live2DError(f'{label} must be a nonempty UTF-8 ID without NUL')
    return value


def _object(value, label):
    if not isinstance(value, dict): raise Live2DError(f'{label} must be an object')
    return value


def _array(value, label, limit=100000):
    if not isinstance(value, list) or len(value) > limit:
        raise Live2DError(f'{label} must be an array with at most {limit} entries')
    return value


def _json_blob(blob, label):
    if isinstance(blob, (str, Path)):
        path = Path(blob)
        if not path.is_file() or path.stat().st_size > MAX_JSON:
            raise Live2DError(f'{label} missing or exceeds 32 MiB')
        blob = path.read_bytes()
    if isinstance(blob, dict):
        try: blob = json.dumps(blob, allow_nan=False).encode('utf-8')
        except (ValueError, TypeError) as error: raise Live2DError(f'Invalid {label}: {error}') from error
    if not isinstance(blob, bytes) or not 0 < len(blob) <= MAX_JSON:
        raise Live2DError(f'{label} must be nonempty JSON bytes, a path, or an object')
    try:
        value = json.loads(blob.decode('utf-8-sig'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise Live2DError(f'Invalid {label}: {error}') from error
    _object(value, label)
    # R5's SDK JSON reader needs a newline/comma after a number and does not
    # accept exponent notation or \u escapes. Emit equivalent plain decimals
    # and raw UTF-8, with a newline before closing objects/arrays.
    def sdk_json(item):
        if isinstance(item, dict):
            return '{\n' + ',\n'.join(sdk_json(k) + ':' + sdk_json(v) for k, v in item.items()) + '\n}'
        if isinstance(item, list): return '[\n' + ',\n'.join(sdk_json(v) for v in item) + '\n]'
        if isinstance(item, float):
            if not math.isfinite(item): raise ValueError('Non-finite JSON number')
            return format(Decimal(str(item)), 'f')
        encoded = json.dumps(item, ensure_ascii=False, allow_nan=False)
        if '\\u' in encoded: raise ValueError('SDK does not support Unicode escape sequences for control characters')
        return encoded
    try: canonical = sdk_json(value).encode('utf-8')
    except (ValueError, TypeError, RecursionError) as error: raise Live2DError(f'Invalid {label}: {error}') from error
    if len(canonical) > MAX_JSON: raise Live2DError(f'{label} canonical form exceeds 32 MiB')
    return value, canonical


def validate_physics(data):
    """Validate native array sizes/indices/types before entering SDK C++ code."""
    _object(data, 'physics3')
    if data.get('Version') != 3: raise Live2DError('Only physics3 Version 3 is supported')
    meta = _object(data.get('Meta'), 'Physics Meta')
    settings = _array(data.get('PhysicsSettings'), 'PhysicsSettings', 10000)
    fps = _number(meta.get('Fps', 0), 'Physics FPS', 0, 1000)
    forces = _object(meta.get('EffectiveForces'), 'EffectiveForces')
    for force in ('Gravity', 'Wind'):
        vector = _object(forces.get(force), force)
        for axis in ('X', 'Y'): _number(vector.get(axis), f'{force}.{axis}')
    counts = {'PhysicsSettingCount': len(settings), 'TotalInputCount': 0,
              'TotalOutputCount': 0, 'VertexCount': 0}
    source_ids, output_ids = set(), set()
    for group in settings:
        _object(group, 'Physics group')
        inputs = _array(group.get('Input'), 'Physics Input')
        outputs = _array(group.get('Output'), 'Physics Output')
        vertices = _array(group.get('Vertices'), 'Physics Vertices')
        if not vertices: raise Live2DError('Physics group needs at least its root particle')
        counts['TotalInputCount'] += len(inputs)
        counts['TotalOutputCount'] += len(outputs)
        counts['VertexCount'] += len(vertices)
        norm = _object(group.get('Normalization'), 'Physics Normalization')
        for kind in ('Position', 'Angle'):
            values = _object(norm.get(kind), f'Normalization.{kind}')
            lo, default, hi = [_number(values.get(k), f'{kind}.{k}') for k in ('Minimum', 'Default', 'Maximum')]
            # A zero-width normalization is valid (constant output). The SDK
            # also orders reversed min/max before normalization.
            if not min(lo, hi) <= default <= max(lo, hi):
                raise Live2DError(f'Invalid {kind} normalization bounds')
        for is_output, records in ((False, inputs), (True, outputs)):
            for item in records:
                _object(item, 'Physics Input/Output')
                target = _object(item.get('Destination' if is_output else 'Source'), 'Physics parameter reference')
                if target.get('Target') != 'Parameter': raise Live2DError('Physics supports Parameter targets only')
                pid = _id(target.get('Id'), 'Physics parameter ID')
                (output_ids if is_output else source_ids).add(pid)
                if item.get('Type') not in ('X', 'Y', 'Angle'): raise Live2DError('Unknown physics axis type')
                if not isinstance(item.get('Reflect'), bool): raise Live2DError('Physics Reflect must be boolean')
                _number(item.get('Weight'), 'Physics Weight', 0, 100)
                if is_output:
                    index = item.get('VertexIndex')
                    if type(index) is not int or not 1 <= index < len(vertices):
                        raise Live2DError('Physics output VertexIndex outside particle chain')
                    _number(item.get('Scale'), 'Physics Scale')
        for index, particle in enumerate(vertices):
            _object(particle, 'Physics particle')
            vector = _object(particle.get('Position'), 'Particle Position')
            for axis in ('X', 'Y'): _number(vector.get(axis), f'Particle Position.{axis}')
            for key in ('Mobility', 'Delay', 'Acceleration', 'Radius'):
                _number(particle.get(key), f'Particle {key}', 0, 1e6)
            if index and particle['Radius'] <= 0:
                raise Live2DError('Non-root particle radius must be positive')
    for key, count in counts.items():
        if type(meta.get(key)) is not int or meta[key] != count or count > 100000:
            raise Live2DError(f'Physics Meta.{key} does not match actual bounded data')
    return {**counts, 'fps': fps, 'source_ids': source_ids, 'output_ids': output_ids}


def validate_pose(data):
    _object(data, 'pose3')
    if data.get('Type', 'Live2D Pose') != 'Live2D Pose': raise Live2DError('Unsupported pose type')
    if 'FadeInTime' in data: _number(data['FadeInTime'], 'Pose FadeInTime', 0, 1e6)
    part_ids = set()
    for group in _array(data.get('Groups'), 'Pose Groups', 10000):
        for item in _array(group, 'Pose Group'):
            _object(item, 'Pose part')
            part_ids.add(_id(item.get('Id'), 'Pose part ID'))
            for linked in _array(item.get('Link', []), 'Pose Link'):
                part_ids.add(_id(linked, 'Pose linked part ID'))
    return {'part_ids': part_ids}


class OfficialPhysics:
    """Stateful official physics and pose, with explicit deterministic reset.

    evaluate(parameters, parts, dt) returns (all real parameters, all parts).
    Missing real parameters use model defaults each call. Only supplied part
    entries override current state, so callers should avoid feeding a full
    default parts map every frame when they want pose fading to accumulate.
    Unknown parameter IDs are supported as SDK virtual pose-switch parameters;
    physics references, however, must name real model parameters for safety.
    """
    def __init__(self, library_path, moc_bytes, physics_bytes=None, pose_bytes=None):
        self._handle = None
        self._lock = threading.RLock()
        self._lib = None
        self.physics_stats, self.pose_stats = None, None
        if physics_bytes is not None:
            data, physics_bytes = _json_blob(physics_bytes, 'physics3')
            self.physics_stats = validate_physics(data)
        if pose_bytes is not None:
            data, pose_bytes = _json_blob(pose_bytes, 'pose3')
            self.pose_stats = validate_pose(data)
        if not isinstance(moc_bytes, bytes) or not 4 <= len(moc_bytes) <= 512 * 1024 * 1024:
            raise Live2DError('MOC buffer outside supported 4 bytes..512 MiB range')
        path = Path(library_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in ('.so', '.dylib', '.dll'):
            raise Live2DError('Select a locally built, trusted official Framework bridge shared library')
        self.library_path = str(path)
        try: self._lib = C.CDLL(str(path))  # This adapter exports the C calling convention on every OS.
        except OSError as error: raise Live2DError(f'Cannot load Framework bridge: {error}') from error
        specs = {
            'abi_version': (U, []), 'core_version': (U, []), 'last_error': (C.c_char_p, []),
            'create': (P, [P, U, P, U, P, U]), 'destroy': (None, [P]), 'reset': (I, [P]),
            'parameter_count': (I, [P]), 'part_count': (I, [P]),
            'parameter_id': (C.c_char_p, [P, I]), 'part_id': (C.c_char_p, [P, I]),
            'defaults': (I, [P, PF, I]), 'set_parameters': (I, [P, PF, I]),
            'set_parameter_id': (I, [P, C.c_char_p, F]), 'set_part': (I, [P, I, F]),
            'stabilize': (I, [P]), 'step': (I, [P, F, I]),
            'read': (I, [P, PF, I, PF, I]),
        }
        self._api = {}
        for name, (restype, argtypes) in specs.items():
            try: fn = getattr(self._lib, 'l2dp_' + name)
            except AttributeError as error: raise Live2DError(f'Incompatible Framework bridge: missing {name}') from error
            fn.restype, fn.argtypes = restype, argtypes
            self._api[name] = fn
        if self._api['abi_version']() != 1: raise Live2DError('Unsupported Framework bridge ABI')
        self.core_version = self._api['core_version']()
        self._handle = self._api['create'](moc_bytes, len(moc_bytes), physics_bytes,
                                          len(physics_bytes or b''), pose_bytes, len(pose_bytes or b''))
        if not self._handle: self._fail()
        try:
            nc, np = self._api['parameter_count'](self._handle), self._api['part_count'](self._handle)
            if not 0 <= nc <= 1000000 or not 0 <= np <= 1000000: raise Live2DError('Invalid bridge model counts')
            self.parameter_ids = tuple(self._api['parameter_id'](self._handle, i).decode('utf-8') for i in range(nc))
            self.part_ids = tuple(self._api['part_id'](self._handle, i).decode('utf-8') for i in range(np))
            self._parameter_ids_set = set(self.parameter_ids)
            self._part_indices = {pid: i for i, pid in enumerate(self.part_ids)}
            self._values, self._parts = (F * nc)(), (F * np)()
            self._check('defaults', self._values, nc)
            self.defaults = dict(zip(self.parameter_ids, self._values))
            if self.physics_stats:
                missing = (self.physics_stats['source_ids'] | self.physics_stats['output_ids']) - self._parameter_ids_set
                if missing: raise Live2DError(f'Physics refers to missing model parameters: {sorted(missing)[:8]}')
            if self.pose_stats:
                missing = self.pose_stats['part_ids'] - set(self.part_ids)
                if missing: raise Live2DError(f'Pose refers to missing model parts: {sorted(missing)[:8]}')
            self.last_parameters, self.last_parts = self._read()
        except Exception:
            self.close()
            raise

    def _fail(self):
        message = self._api['last_error']()
        raise Live2DError(message.decode('utf-8', 'replace') if message else 'Official Framework bridge failed')

    def _check(self, name, *args):
        if not self._handle: raise Live2DError('Official physics instance is closed')
        if not self._api[name](self._handle, *args): self._fail()

    def _input(self, parameters, parts):
        parameters, parts = parameters or {}, parts or {}
        for i, pid in enumerate(self.parameter_ids):
            self._values[i] = _number(parameters.get(pid, self.defaults[pid]), f'Parameter {pid}')
        extras = [(_id(pid, 'Parameter ID'), _number(value, f'Parameter {pid}'))
                  for pid, value in parameters.items() if pid not in self._parameter_ids_set]
        overrides = []
        for pid, value in parts.items():
            if pid not in self._part_indices: raise Live2DError(f'Unknown part override: {pid}')
            overrides.append((self._part_indices[pid], _number(value, f'Part {pid}', 0, 1)))
        self._check('set_parameters', self._values, len(self.parameter_ids))
        for pid, value in extras: self._check('set_parameter_id', pid.encode('utf-8'), value)
        for index, value in overrides: self._check('set_part', index, value)

    def _read(self):
        self._check('read', self._values, len(self.parameter_ids), self._parts, len(self.part_ids))
        return dict(zip(self.parameter_ids, self._values)), dict(zip(self.part_ids, self._parts))

    def reset(self, parameters=None, parts=None, stabilize=True):
        with self._lock:
            self._check('reset')
            self._input(parameters, parts)
            if stabilize: self._check('stabilize')
            self.last_parameters, self.last_parts = self._read()
            return self.last_parameters.copy(), self.last_parts.copy()

    def evaluate(self, parameters=None, parts=None, dt=0.0, *, phase='both'):
        """phase='physics'/'pose' permits R5 lipsync between the two stages.

        Do not pass a fresh unprocessed input map to the pose phase; pass the
        preceding physics result after applying any lipsync updates.
        """
        delta = _number(dt, 'Physics delta', 0, 5)
        phases = {'both': 0, 'physics': 1, 'pose': 2}
        if phase not in phases: raise Live2DError('Physics phase must be both, physics or pose')
        with self._lock:
            self._input(parameters, parts)
            self._check('step', delta, phases[phase])
            self.last_parameters, self.last_parts = self._read()
            return self.last_parameters.copy(), self.last_parts.copy()

    def close(self):
        with self._lock:
            if self._handle:
                self._api['destroy'](self._handle)
                self._handle = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def __del__(self):
        try: self.close()
        except Exception: pass
