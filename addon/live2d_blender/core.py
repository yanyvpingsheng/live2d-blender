"""Optional ctypes adapter to a USER-INSTALLED official Cubism Native Core.

No SDK binaries/headers are distributed, downloaded, searched for or accepted.
The chosen library executes native code in Blender: only trusted official builds.
ABI reference: NativeCoreAPIReference_en_r15.pdf, public Live2D documentation.
Synthetic tests do not prove real-model fidelity. Separate private-model smoke tests use an authorized SDK.
"""
import ctypes as C
from array import array
from math import isfinite
import os
from pathlib import Path
from .assets import read_json
from .data import Drawable, Live2DError, Parameter, Part, Snapshot, validate_snapshot

P = C.c_void_p
I = C.c_int32
U = C.c_uint32
F = C.c_float
PI = C.POINTER(I)
PF = C.POINTER(F)
PP = C.POINTER(C.c_char_p)


class Vec2(C.Structure):
    _fields_ = [('x', F), ('y', F)]


class Vec4(C.Structure):
    _fields_ = [('x', F), ('y', F), ('z', F), ('w', F)]


class AlignedBuffer:
    def __init__(self, size, alignment, content=None):
        if not 0 < size <= 512 * 1024 * 1024:
            raise Live2DError('Core allocation outside 1 byte–512 MiB safety limit')
        if alignment <= 0 or alignment & (alignment - 1):
            raise Live2DError('Alignment must be a positive power of two')
        self.owner = C.create_string_buffer(size + alignment - 1)
        self.address = (C.addressof(self.owner) + alignment - 1) & ~(alignment - 1)
        self.size = size
        self.pointer = P(self.address)
        if content is not None:
            if len(content) != size:
                raise Live2DError('Buffer content length mismatch')
            C.memmove(self.address, content, size)


class CoreLibrary:
    def __init__(self, path):
        path = Path(path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in ('.so', '.dll', '.dylib'):
            raise Live2DError('Select an official SDK shared library (.so/.dll/.dylib), not a static .a/.lib')
        try:
            loader = C.WinDLL if os.name == 'nt' else C.CDLL
            self.lib = loader(str(path))  # Official header uses __stdcall for Windows DLL.
        except OSError as error:
            raise Live2DError(f'Cannot load Core (check OS/CPU architecture): {error}') from error
        self.path = str(path)
        self.api = {}
        self._bind('csmGetVersion', U, [])
        self._bind('csmGetLatestMocVersion', U, [])
        self._bind('csmGetMocVersion', U, [P, U])
        self._bind('csmHasMocConsistency', I, [P, U])
        self._bind('csmReviveMocInPlace', P, [P, U])
        self._bind('csmGetSizeofModel', U, [P])
        self._bind('csmInitializeModelInPlace', P, [P, P, U])
        self._bind('csmUpdateModel', None, [P])
        self._bind('csmResetDrawableDynamicFlags', None, [P])
        self._bind('csmReadCanvasInfo', None, [P, C.POINTER(Vec2), C.POINTER(Vec2), PF])
        for prefix in ('Parameter', 'Part', 'Drawable'):
            self._bind('csmGet' + prefix + 'Count', I, [P])
            self._bind('csmGet' + prefix + 'Ids', PP, [P])
        for name in ('ParameterMinimumValues', 'ParameterMaximumValues', 'ParameterDefaultValues',
                     'ParameterValues', 'PartOpacities', 'DrawableOpacities'):
            self._bind('csmGet' + name, PF, [P])
        for name in ('DrawableTextureIndices', 'DrawableVertexCounts', 'DrawableIndexCounts', 'DrawableMaskCounts'):
            self._bind('csmGet' + name, PI, [P])
        for name in ('DrawableConstantFlags', 'DrawableDynamicFlags'):
            self._bind('csmGet' + name, C.POINTER(C.c_uint8), [P])
        for name in ('DrawableVertexPositions', 'DrawableVertexUvs'):
            self._bind('csmGet' + name, C.POINTER(C.POINTER(Vec2)), [P])
        self._bind('csmGetDrawableIndices', C.POINTER(C.POINTER(C.c_uint16)), [P])
        self._bind('csmGetDrawableMasks', C.POINTER(PI), [P])
        for name in ('DrawableMultiplyColors', 'DrawableScreenColors'):
            self._bind('csmGet' + name, C.POINTER(Vec4), [P], optional=True)
        for name in ('ParameterTypes', 'ParameterRepeats', 'PartParentPartIndices',
                     'DrawableParentPartIndices', 'DrawableBlendModes', 'DrawableRenderOrders', 'RenderOrders'):
            self._bind('csmGet' + name, PI, [P], optional=True)
        self._bind('csmGetOffscreenCount', I, [P], optional=True)
        if not any(n in self.api for n in ('csmGetRenderOrders', 'csmGetDrawableRenderOrders')):
            raise Live2DError('Core does not provide a supported render-order API')
        self.version = self.api['csmGetVersion']()

    def _bind(self, name, result, args, optional=False):
        try:
            fn = getattr(self.lib, name)
        except AttributeError as error:
            if optional:
                return
            raise Live2DError(f'Core lacks required {name}; use a current official Native SDK') from error
        fn.restype = result
        fn.argtypes = args
        self.api[name] = fn


class NativeBackend:
    kind = 'NATIVE'
    # Initial topology and every dynamic snapshot are validated by this backend.
    trusted_validated_snapshot = True

    def __init__(self, package, library_path):
        package.require_model()
        self.package = package
        self.name = package.name
        self.textures = package.textures
        self.metadata = package.raw
        display_path = package.references.get('DisplayInfo')
        self.display_info = read_json(display_path) if display_path and display_path.is_file() else {}
        self.core = CoreLibrary(library_path)
        self.api = self.core.api
        size = package.moc.stat().st_size
        if not 4 <= size <= 512 * 1024 * 1024:
            raise Live2DError('moc3 file size outside supported bounds')
        blob = package.moc.read_bytes()
        self._moc_memory = AlignedBuffer(len(blob), 64, blob)
        moc_version = self.api['csmGetMocVersion'](self._moc_memory.pointer, len(blob))
        if moc_version == 0 or moc_version > self.api['csmGetLatestMocVersion']():
            raise Live2DError('moc3 is unknown or newer than this installed Core')
        if not self.api['csmHasMocConsistency'](self._moc_memory.pointer, len(blob)):
            raise Live2DError('Official Core rejected moc3 consistency; import stopped')
        self._moc = self.api['csmReviveMocInPlace'](self._moc_memory.pointer, len(blob))
        if not self._moc:
            raise Live2DError('Official Core could not revive moc3')
        self._model_memory = AlignedBuffer(self.api['csmGetSizeofModel'](self._moc), 16)
        self._model = self.api['csmInitializeModelInPlace'](self._moc, self._model_memory.pointer, self._model_memory.size)
        if not self._model:
            raise Live2DError('Official Core could not initialize model')
        self.offscreen_count = self._call('OffscreenCount', fallback=0)
        # A flattened z-stack is NOT an implementation of Cubism 5.3 offscreen groups.
        if self.offscreen_count:
            raise Live2DError(f'Model requires {self.offscreen_count} Cubism 5.3 offscreen groups; this renderer cannot import them faithfully yet')
        self.parameters = self._parameters()
        self.parts = self._parts()
        self.warnings = ['Native Core mesh evaluation tested on one model; complete visual/animation parity is unvalidated.']
        for key in ('Physics', 'Pose'):
            if key in package.references:
                self.warnings.append(f'{key} file retained in manifest; {key.lower()} evaluation is not implemented.')
        if package.missing:
            self.warnings.append('Some optional referenced files are missing; see manifest report.')
        size, origin, ppu = Vec2(), Vec2(), F()
        self.api['csmReadCanvasInfo'](self._model, C.byref(size), C.byref(origin), C.byref(ppu))
        self.canvas = {'size': [size.x, size.y], 'origin': [origin.x, origin.y], 'pixels_per_unit': ppu.value}
        self._static_drawables = None
        self.api['csmUpdateModel'](self._model)

    def _call(self, suffix, fallback=None):
        fn = self.api.get('csmGet' + suffix)
        return fn(self._model) if fn else fallback

    def _count(self, suffix):
        count = self._call(suffix)
        if not 0 <= count <= 1000000:
            raise Live2DError(f'Invalid Core {suffix}: {count}')
        return count

    def _parameters(self):
        n = self._count('ParameterCount')
        ids, lo, hi, default = [self._call('Parameter' + s) for s in ('Ids', 'MinimumValues', 'MaximumValues', 'DefaultValues')]
        types, repeats = self._call('ParameterTypes'), self._call('ParameterRepeats')
        return [Parameter(ids[i].decode('utf-8'), lo[i], hi[i], default[i], types[i] if types else 0, bool(repeats[i]) if repeats else False) for i in range(n)]

    def _parts(self):
        n = self._count('PartCount')
        ids, opacity, parents = self._call('PartIds'), self._call('PartOpacities'), self._call('PartParentPartIndices')
        return [Part(ids[i].decode('utf-8'), parents[i] if parents else -1, opacity[i]) for i in range(n)]

    def evaluate(self, parameters, parts=None):
        if self._model is None:
            raise Live2DError('Core model is closed; reconnect it before evaluating')
        values = self._call('ParameterValues')
        for i, parameter in enumerate(self.parameters):
            values[i] = parameter.constrain(parameters.get(parameter.id, parameter.default))
        opacity = self._call('PartOpacities')
        for i, part in enumerate(self.parts):
            opacity[i] = max(0.0, min(1.0, (parts or {}).get(part.id, part.opacity)))
        self.api['csmUpdateModel'](self._model)
        return self.snapshot()

    def snapshot(self):
        if self._model is None:
            raise Live2DError('Core model is closed; reconnect it before taking a snapshot')
        n = self._count('DrawableCount')
        static = self._static_drawables
        first_snapshot = static is None
        if not first_snapshot and n != len(static):
            raise Live2DError('Core Drawable count changed; reconnect the model')
        # These static attributes never change during Core parameter evaluation.
        # Keep an owned cache; returned snapshots never borrow native pointers.
        names = ['VertexCounts', 'IndexCounts', 'MaskCounts', 'VertexPositions',
                 'DynamicFlags', 'Opacities', 'MultiplyColors', 'ScreenColors']
        if first_snapshot:
            names += ['Ids', 'TextureIndices', 'VertexUvs', 'Indices', 'Masks',
                      'ConstantFlags', 'ParentPartIndices', 'BlendModes']
            static = []
        fields = {s: self._call('Drawable' + s) for s in names}
        orders = self._call('RenderOrders')
        if orders is None:
            orders = self._call('DrawableRenderOrders')

        def pairs(pointer, count, label):
            if not count:
                return []
            if not pointer:
                raise Live2DError(f'Core returned a null {label} pointer')
            # Batch-copy float32 coordinates, instead of accessing ctypes Vec2
            # structures twice per vertex. No NumPy dependency is required.
            values = array('f')
            values.frombytes(C.string_at(pointer, count * C.sizeof(Vec2)))
            if not all(map(isfinite, values)):
                raise Live2DError(f'Core returned a non-finite {label}')
            return list(zip(values[::2], values[1::2]))

        drawables = []
        for i in range(n):
            vc, ic, mc = (fields[s][i] for s in ('VertexCounts', 'IndexCounts', 'MaskCounts'))
            if not (0 <= vc <= 65536 and 0 <= ic <= 10000000 and 0 <= mc <= n):
                raise Live2DError('Invalid Core geometry count')
            if first_snapshot:
                flags = fields['ConstantFlags'][i]
                entry = {
                    'id': fields['Ids'][i].decode('utf-8'), 'texture': fields['TextureIndices'][i],
                    'uvs': tuple(pairs(fields['VertexUvs'][i], vc, 'UV coordinate')),
                    'indices': tuple(fields['Indices'][i][j] for j in range(ic)),
                    'masks': tuple(fields['Masks'][i][j] for j in range(mc)),
                    'inverted_mask': bool(flags & 8), 'double_sided': bool(flags & 4),
                    'blend_mode': fields['BlendModes'][i] if fields['BlendModes'] else (1 if flags & 1 else 2 if flags & 2 else 0),
                    'parent_part': fields['ParentPartIndices'][i] if fields['ParentPartIndices'] else -1,
                    'counts': (vc, ic, mc),
                }
                static.append(entry)
            else:
                entry = static[i]
                if (vc, ic, mc) != entry['counts']:
                    raise Live2DError(f"{entry['id']}: Core geometry counts changed; reconnect the model")

            def color(name, default):
                ptr = fields[name]
                return (ptr[i].x, ptr[i].y, ptr[i].z, ptr[i].w) if ptr else default

            opacity = fields['Opacities'][i]
            multiply = color('MultiplyColors', (1, 1, 1, 1))
            screen = color('ScreenColors', (0, 0, 0, 0))
            if not all(map(isfinite, (opacity, *multiply, *screen))):
                raise Live2DError(f"{entry['id']}: Core returned non-finite opacity/color")
            # Never dereference jagged pointers when the corresponding count is zero.
            vertices = pairs(fields['VertexPositions'][i], vc, 'vertex coordinate')
            drawables.append(Drawable(
                id=entry['id'], texture=entry['texture'], vertices=vertices, uvs=list(entry['uvs']),
                indices=list(entry['indices']), render_order=orders[i],
                opacity=opacity, visible=bool(fields['DynamicFlags'][i] & 1),
                masks=list(entry['masks']), inverted_mask=entry['inverted_mask'], double_sided=entry['double_sided'],
                blend_mode=entry['blend_mode'], multiply_color=multiply, screen_color=screen,
                parent_part=entry['parent_part'],
            ))
        snapshot = Snapshot(drawables, list(self.warnings))
        if first_snapshot:
            validate_snapshot(snapshot)
            # Commit only valid topology. Copy its lists into each result so
            # editing an earlier snapshot cannot poison later evaluations.
            self._static_drawables = static
        self.api['csmResetDrawableDynamicFlags'](self._model)
        return snapshot

    def close(self):
        self._model = self._moc = None
        self._model_memory = self._moc_memory = None
        self._static_drawables = None
