# Official physics/pose sampling bridge

This is original integration code. It does **not** reimplement Cubism's physics
equations and does not include any SDK source, binary, header, or model.

The bridge is a **conversion-time sampling dependency**. A final fitted Blender
armature/skin/Action must not load it to play. Bone fitting is a separate stage;
these samples alone are not a successful or lossless bone conversion.

## Build privately

Use an already installed, appropriately licensed **Cubism Native SDK 5-r.5** and
the matching official shared Core library. The following Linux build was tested:

```sh
python native/build_physics_bridge.py \
  --sdk /private/CubismSdkForNative-5-r.5 \
  --core /private/CubismSdkForNative-5-r.5/Core/dll/linux/x86_64/libLive2DCubismCore.so \
  --output /private/build/liblive2d_physics_bridge.so
```

The compiler only reads the chosen SDK. No SDK download, installation, license
acceptance, or source modification occurs. CMake is also provided for local
builds. macOS/Windows have **not** been built or tested; their calling convention,
dependencies and toolchain must be verified before support is claimed.

The resulting shared library contains linked Framework code. Keep it outside
the distributed add-on and published archives. Supplying source integration does
not resolve the SDK, expandable-application, Blender/GPL, or model licensing
questions. Review those separately before distribution.

## Python API

```python
from live2d_blender.physics import OfficialPhysics
from live2d_blender.physics_sampling import sample_sequence

with OfficialPhysics(shim_path, moc_bytes, physics_bytes, pose_bytes=None) as rig:
    parameters, parts = rig.reset(initial_inputs, {}, stabilize=True)
    parameters, parts = rig.evaluate(next_inputs, {}, 1 / 60)
    snapshot = native_backend.evaluate(parameters, parts)
```

`OfficialPhysics` owns a separate Framework model. It never takes ownership of
the mesh backend's `csmModel*` or Python-aligned allocation. `reset` reconstructs
the model and solver, removing all input/output, residual-time, particle, pose,
and virtual-parameter state. Stabilization is explicit and on by default.

The dictionaries passed to `evaluate` are **pre-physics inputs**. Missing real
parameters use model defaults, while SDK-owned particle and interpolation state
continues across calls. Do not feed last frame's physics output back as ordinary
input unless an intentional feedback system requires it. Group-to-group
propagation within a solver step remains entirely inside the official solver.
Only explicitly supplied part opacities are overridden: passing a complete
default-part map every frame prevents pose fading from accumulating normally.

For R5, place the stages in this order:

1. Motion/base animation
2. Eye blink (200)
3. Expression (300)
4. Look (400)
5. Breath (500)
6. Physics (600)
7. Lip sync (700)
8. Pose (800)
9. Core mesh update

The bridge handles stages 6 and 8 only. `phase='physics'` and `phase='pose'` allow
lip sync to be applied between them; pass the physics result plus lip-sync edits
into the pose phase. The bridge does not call Core's mesh update itself. The
mesh backend should receive the resulting parameter/part dictionaries and update
Core afterward. Check that the two Core versions match.

`sample_sequence` resets and yields a regular-clock sequence of parameter,
part, and optional mesh snapshots. Sample-frame zero is the stabilized initial
state. For a seek/rewind, replay the same pre-physics input samples from the
start; a single large delta is not a deterministic replacement. Internal fixed
FPS and input/output interpolation are handled by the selected official SDK.

The `sample_sequence` API can be used for local downstream bone-fitting work.
Any generated geometry or cache is model-derived data and remains subject to
the model's permissions. Do not commit it to this public source repository.

## Validation and boundaries

The Python adapter validates JSON shape, aggregate counts, array indices,
numeric finiteness, supported axes and parameter/part references before
evaluation. Physics references to absent real model parameters are rejected
because the SDK physics code indexes Core parameter arrays directly.

R5's bundled JSON reader requires a newline/comma after numbers and does not
accept exponent notation or Unicode escape sequences. The adapter serializes
equivalent plain decimal values, newlines and raw UTF-8. Original files are not
rewritten. Unsupported control-character escape strings are rejected.

Tests include synthetic configuration validation, reflected and weighted angle
output, multi-segment chains, inter-group propagation, X/Y/Angle finiteness,
24/30/60-Hz repeatability, model isolation, close/reset behavior, and synthetic
pose fading/linking on explicitly chosen non-watermark parts. Native tests need
an explicit trusted library and authorized model; ordinary test runs skip them.

These results do **not** establish full visual parity, equivalence of different
sampling rates, arbitrary-model compatibility, external gravity/wind controls,
or complete motion/expression/controller behavior. X/Y axis tests check finite
official outputs, not an independently derived numerical oracle. The bridge
uses the installed R5 algorithm including any version-specific quirks; it does
not silently patch the SDK. Malformed native models remain the official Core's
responsibility after consistency checking; this is not a native-code sandbox.

Official API references are the supplied SDK headers `Physics/CubismPhysics.hpp`,
`Effect/CubismPose.hpp`, and `Motion/ICubismUpdater.hpp`, plus the public
[physics manual](https://docs.live2d.com/en/cubism-sdk-manual/physics/) and
[R5 migration guide](https://docs.live2d.com/en/cubism-sdk-manual/compatibility-with-cubism-5-3-official/).
