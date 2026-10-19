"""Deterministic bounded replay of a stateful official solver (no Blender API)."""
import math
from .data import Live2DError, finite


def replay_physics(engine, sample_inputs, seconds, fps=60.0, max_steps=12000):
    """Reset to the chosen timeline start and replay sampled controls in order.

    sample_inputs(t) returns (parameter_inputs, explicit_part_overrides, opacity).
    Each call deliberately rebuilds the solver history: scrubbing forward,
    backward and repeatedly to one time yields the same result. No warm-up
    beyond the official stabilization call is assumed.
    """
    seconds = max(0.0, finite(seconds, 'timeline time'))
    fps = finite(fps, 'simulation FPS')
    if fps <= 0 or fps > 1000:
        raise Live2DError('Simulation FPS must be in (0, 1000]')
    steps = int(math.ceil(seconds * fps - 1e-9))
    if steps > max_steps:
        raise Live2DError(f'Physics replay requires {steps} steps (limit {max_steps}); shorten the timeline or move its start')
    parameters, parts, opacity = sample_inputs(0.0)
    result_p, result_parts = engine.reset(parameters, parts, stabilize=True)
    previous = 0.0
    for index in range(1, steps + 1):
        now = min(seconds, index / fps)
        parameters, parts, opacity = sample_inputs(now)
        result_p, result_parts = engine.evaluate(parameters, parts, now - previous)
        previous = now
    return result_p, result_parts, opacity
