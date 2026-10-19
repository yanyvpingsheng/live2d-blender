"""Original synthetic geometric mascot: NOT a Live2D model or Core emulator."""
import math
from pathlib import Path
from .data import Drawable, Parameter, Part, Snapshot


def oval(cx, cy, rx, ry, texture, ident, order, angle=0.0, segments=32):
    vertices = [(cx, cy)]
    uvs = [(0.5, 0.5)]
    for i in range(segments):
        t = i * 2 * math.pi / segments
        dx, dy = math.cos(t) * rx, math.sin(t) * ry
        vertices.append((cx + dx * math.cos(angle) - dy * math.sin(angle), cy + dx * math.sin(angle) + dy * math.cos(angle)))
        uvs.append((0.5 + 0.49 * math.cos(t), 0.5 + 0.49 * math.sin(t)))
    indices = []
    for i in range(segments):
        indices.extend((0, i + 1, (i + 1) % segments + 1))
    return Drawable(ident, texture, vertices, uvs, indices, render_order=order)


class SyntheticBackend:
    kind = 'SYNTHETIC'
    name = 'Synthetic Sprout'
    canvas = {'size': [1000, 1000], 'origin': [500, 500], 'pixels_per_unit': 500}
    metadata = {'Synthetic': True, 'Notice': 'Original test fixture. No Live2D model, Core or SDK assets.'}
    parameters = [Parameter('ParamAngleX', -30, 30, 0), Parameter('ParamEyeOpen', 0, 1, 1),
                  Parameter('ParamMouthOpenY', 0, 1, 0), Parameter('ParamBreath', 0, 1, 0),
                  Parameter('ParamLeafWave', -1, 1, 0), Parameter('ParamOpacity', 0, 1, 1)]
    parts = [Part('PartBody'), Part('PartFace', 0)]

    def __init__(self):
        self.textures = [Path(__file__).parent / 'demo_assets' / (name + '.png') for name in ('mint', 'dark', 'peach', 'leaf')]
        self.warnings = ['SYNTHETIC FIXTURE: verifies Blender plumbing only; not a real Live2D import.']

    def evaluate(self, parameters, parts=None):
        p = {v.id: v.constrain(parameters.get(v.id, v.default)) for v in self.parameters}
        x = p['ParamAngleX'] / 30 * 0.15
        breath = p['ParamBreath'] * 0.05
        eye = max(0.025, p['ParamEyeOpen'] * 0.075)
        ds = [oval(0, -0.1, 0.62 + breath, 0.66 + breath, 0, 'Body', 0),
              oval(-0.16, 0.57, 0.11, 0.32, 3, 'LeafLeft', 1, -0.6 + p['ParamLeafWave'] * 0.4),
              oval(0.13, 0.59, 0.12, 0.30, 3, 'LeafRight', 2, 0.6 + p['ParamLeafWave'] * 0.4),
              oval(-0.21 + x, 0.0, 0.045, eye, 1, 'EyeLeft', 3),
              oval(0.21 + x, 0.0, 0.045, eye, 1, 'EyeRight', 4),
              oval(x, -0.22, 0.09, 0.018 + p['ParamMouthOpenY'] * 0.11, 1, 'Mouth', 5),
              oval(-0.33 + x, -0.12, 0.1, 0.055, 2, 'CheekLeft', 6),
              oval(0.33 + x, -0.12, 0.1, 0.055, 2, 'CheekRight', 7)]
        for i, drawable in enumerate(ds):
            drawable.parent_part = 0 if i < 3 else 1
            part_id = self.parts[drawable.parent_part].id
            drawable.opacity = p['ParamOpacity'] * (parts or {}).get(part_id, 1.0)
        return Snapshot(ds, list(self.warnings))

    def close(self):
        pass
