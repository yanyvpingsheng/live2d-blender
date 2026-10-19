"""Deterministic single-clip motion3 curves and expression values.

This is not the Cubism Framework motion manager: transitions, per-curve fades,
eye-blink/lip-sync effects, event callbacks and physics are intentionally absent.
"""
from dataclasses import dataclass
from .data import Live2DError, finite


def bezier(a, b, c, d, u):
    v = 1 - u
    return v*v*v*a + 3*v*v*u*b + 3*v*u*u*c + u*u*u*d


@dataclass(frozen=True)
class Segment:
    kind: int
    points: tuple

    def value(self, t):
        a, z = self.points[0], self.points[-1]
        if self.kind == 2:
            return a[1]
        if self.kind == 3:
            return z[1]
        if z[0] <= a[0]:
            return z[1]
        if self.kind == 0:
            u = max(0.0, min(1.0, (t - a[0]) / (z[0] - a[0])))
            return a[1] + u * (z[1] - a[1])
        b, c = self.points[1:3]
        # Solve the Bezier TIME axis instead of incorrectly treating time as u.
        lo, hi = 0.0, 1.0
        for _ in range(45):
            u = (lo + hi) / 2
            if bezier(a[0], b[0], c[0], z[0], u) < t:
                lo = u
            else:
                hi = u
        return bezier(a[1], b[1], c[1], z[1], (lo + hi) / 2)


@dataclass
class Curve:
    target: str
    id: str
    initial: tuple
    segments: list

    def value(self, t):
        if t <= self.initial[0] or not self.segments:
            return self.initial[1]
        # At a segment boundary, the next segment takes precedence (stepped curves).
        for segment in self.segments:
            if t < segment.points[-1][0]:
                return segment.value(t)
        return self.segments[-1].points[-1][1]


def parse_curve(data):
    if not isinstance(data, dict) or not isinstance(data.get('Id'), str):
        raise Live2DError('Curve must have string Id')
    stream = data.get('Segments')
    if not isinstance(stream, list) or len(stream) < 2:
        raise Live2DError('Curve Segments must begin with time and value')
    stream = [finite(v, 'curve value') for v in stream]
    first = previous = tuple(stream[:2])
    segments = []
    index = 2
    while index < len(stream):
        code = stream[index]
        if code not in (0, 1, 2, 3):
            raise Live2DError(f'Unknown motion segment type {code}')
        count = 6 if code == 1 else 2
        if index + count >= len(stream):
            raise Live2DError('Truncated motion segment')
        values = stream[index + 1:index + 1 + count]
        points = (previous,) + tuple(tuple(values[i:i+2]) for i in range(0, count, 2))
        if points[-1][0] <= previous[0]:
            raise Live2DError('Motion segment time must increase')
        if code == 1 and not (points[0][0] <= points[1][0] <= points[2][0] <= points[3][0]):
            raise Live2DError('Non-monotonic Bezier time handles are unsupported')
        segments.append(Segment(int(code), points))
        previous = points[-1]
        index += count + 1
    target = data.get('Target', 'Parameter')
    if target not in ('Parameter', 'PartOpacity', 'Model'):
        raise Live2DError(f'Unsupported curve target {target}')
    return Curve(target, data['Id'], first, segments)


@dataclass
class Motion:
    duration: float
    loop: bool
    curves: list
    raw: dict

    @classmethod
    def from_json(cls, raw):
        if raw.get('Version') != 3 or not isinstance(raw.get('Meta'), dict):
            raise Live2DError('Expected motion3.json Version 3 with Meta')
        duration = finite(raw['Meta'].get('Duration'), 'Duration')
        if duration <= 0:
            raise Live2DError('Motion duration must be positive')
        curves = raw.get('Curves', [])
        if not isinstance(curves, list):
            raise Live2DError('Curves must be an array')
        return cls(duration, bool(raw['Meta'].get('Loop', False)), [parse_curve(c) for c in curves], raw)

    def sample(self, seconds):
        t = max(0.0, finite(seconds, 'time'))
        t = t % self.duration if self.loop else min(t, self.duration)
        result = {'Parameter': {}, 'PartOpacity': {}, 'Model': {}}
        for curve in self.curves:
            result[curve.target][curve.id] = curve.value(t)
        return result


def apply_expression(parameters, raw, weight=1.0):
    if raw.get('Type') != 'Live2D Expression':
        raise Live2DError('Expected exp3.json Type Live2D Expression')
    out = dict(parameters)
    weight = max(0.0, min(1.0, finite(weight, 'expression weight')))
    for entry in raw.get('Parameters', []):
        key = entry['Id']
        if key not in out:
            continue
        value = finite(entry['Value'], 'expression value')
        mode = entry.get('Blend', 'Add')
        if mode == 'Add':
            out[key] += value * weight
        elif mode == 'Multiply':
            out[key] *= 1 + (value - 1) * weight
        elif mode == 'Overwrite':
            out[key] = out[key] * (1 - weight) + value * weight
        else:
            raise Live2DError(f'Unsupported expression blend {mode}')
    return out
