"""Backend-neutral data. No Blender or proprietary SDK dependency."""
from dataclasses import dataclass, field
from math import isfinite


class Live2DError(ValueError):
    pass


def finite(value, label='number'):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise Live2DError(f'{label} must be a finite number')
    return float(value)


@dataclass(frozen=True)
class Parameter:
    id: str
    minimum: float
    maximum: float
    default: float
    type: int = 0
    repeat: bool = False

    def constrain(self, value):
        value = finite(value, self.id)
        if self.repeat and self.maximum > self.minimum:
            if value < self.minimum or value > self.maximum:
                return self.minimum + (value - self.minimum) % (self.maximum - self.minimum)
        return max(self.minimum, min(self.maximum, value))


@dataclass(frozen=True)
class Part:
    id: str
    parent: int = -1
    opacity: float = 1.0


@dataclass
class Drawable:
    id: str
    texture: int
    vertices: list
    uvs: list
    indices: list
    render_order: int = 0
    opacity: float = 1.0
    visible: bool = True
    masks: list = field(default_factory=list)
    inverted_mask: bool = False
    double_sided: bool = True
    blend_mode: int = 0
    multiply_color: tuple = (1.0, 1.0, 1.0, 1.0)
    screen_color: tuple = (0.0, 0.0, 0.0, 0.0)
    parent_part: int = -1

    def validate(self):
        if len(self.vertices) != len(self.uvs):
            raise Live2DError(f'{self.id}: vertex/UV count mismatch')
        if len(self.indices) % 3:
            raise Live2DError(f'{self.id}: triangle index count must be divisible by 3')
        if any(not isinstance(i, int) or i < 0 or i >= len(self.vertices) for i in self.indices):
            raise Live2DError(f'{self.id}: index outside vertex range')
        for values in (self.vertices, self.uvs):
            for pair in values:
                if len(pair) != 2:
                    raise Live2DError(f'{self.id}: expected 2D coordinates')
                for value in pair:
                    finite(value, f'{self.id} coordinate')
        finite(self.opacity, f'{self.id} opacity')
        return self


@dataclass
class Snapshot:
    drawables: list
    warnings: list = field(default_factory=list)


def validate_snapshot(snapshot):
    ids = set()
    for drawable in snapshot.drawables:
        drawable.validate()
        if drawable.id in ids:
            raise Live2DError(f'Duplicate Drawable ID: {drawable.id}')
        ids.add(drawable.id)
        if any(i < 0 or i >= len(snapshot.drawables) for i in drawable.masks):
            raise Live2DError(f'{drawable.id}: invalid mask index')
    return snapshot
