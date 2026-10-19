# v0.3.0 validation

Date: 2026-10-19. These are local checks on Linux x86_64 and Blender 4.3.2;
GitHub CI, Windows/macOS and other Blender versions are not implied.

## SDK-free Python

Python 3.12.14, NumPy 2.3.5, Pillow 12.3.0:

- `python -m unittest discover -s tests -p 'test_*.py' -v`: **164 discovered,
  147 passed, 17 skipped**, no failures.
- Skips: 10 optional external SDK/model tests and 7 tests that require Blender.
- The mask module and its pure tests use NumPy and Python's standard library;
  image writing does not introduce a Pillow dependency inside Blender.

## New appearance path

All fixtures are original geometry/textures, unrelated to any third-party model.
Run commands are in [TESTING.md](TESTING.md).

- Mask suite in Blender: **16/16 passed** (9 numerical + 7 Blender cases).
  Covers canonical groups, raw alpha union, inversion, texture orientation,
  grayscale PNGs, full integer frames, real Armature evaluation, model-local
  projection under parent transforms, missing UVs, unsaved files, write/load/
  frame-restore failures and rollback.
- Registered Convert UI appearance fixture: **57 checks passed**. Retains all
  nine Drawables; step-2 fitting still writes five integer-frame masks starting
  at scene frame 5. Tests normal/add/inverted/multiply, black additive pixels,
  hidden and zero-opacity mask sources, source-own-mask independence, exact
  original packed color-file hashes, explicit multiply restrictions, failed
  writes and post-bake material-attachment rollback.
- Relocated `.blend` + mask-folder replay: **19 checks passed**, without copying
  original color files. A fresh `--factory-startup --disable-autoexec` process
  imports no add-on/Core, has no scripts/handlers/drivers, and resolves all mask
  files inside the relocated bundle.
- 27 interior RGBA checks across three blend modes, three frames, original
  placement, transformed root and fresh relocated replay: maximum absolute
  error **1.61e-7** against the stated scene-linear equations.
- Float-texture shader checks: **9 passed**, maximum RGBA error **2.03e-7**;
  texture alpha × opacity × mask is applied once, add preserves fractional
  destination alpha, multiply uses an opaque backdrop, culling is preserved,
  and invisible targets leave the backdrop unchanged.

Pixel comparisons deliberately use flat interior points away from triangle
seams. They do not certify boundary filtering, antialiasing or SDK image parity.
Opaque color PNGs in the full-scene fixture avoid Blender's existing 8-bit
semitransparent-color association quantization; the float-image shader fixture
separately exercises texture alpha. Multiply on a fractional-alpha backdrop
remains unsupported, not a passing case hidden by a wider tolerance.

## Existing regression coverage retained

The following were rerun against v0.3 implementation (test filenames retain
v0.2 feature labels where applicable):

- ZIP installation and all three grouping modes: **33 checks passed**.
- Fresh saved-scene replay at **33 integer/half-frame times**: maximum vertex
  and root-world-matrix saved/reopened differences both **0**; 16 ordinary bones,
  no add-on/Core imports, drivers, shape keys or embedded scripts.
- Base import/material/parameter/lifecycle smoke: **19 passed**.
- Conversion failures and rollback: **4 passed**.
- Affine fitting: **37 cases passed**; maximum vertex component error `8.124e-7`,
  maximum skinning matrix component error `1.275e-6`.
- Root placement/animation: **7 passed**, including shear rejection.
- Unsupported child Action/driver/NLA rejection: **3 passed**.
- Existing material-effect creation: **16 passed**; fresh replay **53 passed**,
  maximum synthetic Cycles RGBA error `4.054e-7`.

## Publication boundary and limitations

- Public tree: **74 explicitly listed files**, including only four original
  64×64 PNG fixtures. No SDK/Core/Framework source or binary, user model,
  generated mask/model/cache, `.blend`, compiled bridge, logs or credentials.
- Existing MIT notices and original C++ bridge remain unchanged. The separate
  frozen v0.2 directory's **145 file hashes** remain unchanged; v0.3 changes only
  the new public working tree.
- The source audit and installer use an explicit reviewed manifest. Generated
  validation files remain outside the published source tree.
- Public validation does not distribute or depend on any private model test.
  It establishes the selected native implementation's numerical behavior and
  persistence, not general visual parity with real Live2D models.
- Scene-linear blend arithmetic, no per-Drawable saturation, finite/discrete
  masks, skeletal approximation, transparent-ray limits, new offscreen/group
  blends, unbaked actions and other platforms remain limitations. See
  [the workflow and limits](MASKS_BLENDS_zh.md).
