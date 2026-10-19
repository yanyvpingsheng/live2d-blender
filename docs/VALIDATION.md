# Public-source validation

Date: 2026-10-19. Scope: source-publication preparation for version 0.2.0.

- SDK-free standalone tests: Python 3.12.14, NumPy 2.3.5, Pillow 12.3.0.
- `python -m unittest discover -s tests -p 'test_*.py' -v`: 148 discovered;
  **138 passed, 10 optional native SDK/model tests skipped**, no failures.
- Optional native tests were deliberately not given external SDK/model paths.
  These results do not constitute native SDK validation.

## Blender 4.3.2 / Linux x86_64

Executed in a disposable copy with isolated Blender user scripts, configuration
and data directories, `--factory-startup`, and `--disable-autoexec`:

- ZIP installation and three conversion modes: **33 checks passed**.
- Fresh-process v0.2 replay: **33 integer/half-frame times**, maximum saved/reopened
  vertex and root-world-matrix differences both **0**. Ordinary 16-bone playback;
  no add-on/Core imports, drivers, shape keys, embedded scripts or auto-execution.
- Base synthetic import/material/parameter/lifecycle smoke test: **19 passed**.
- Conversion failure and rollback tests: **4 passed**.
- Synthetic affine regression: **37 cases passed**; maximum vertex component
  error `8.124e-7`, maximum skinning matrix component error `1.275e-6`.
- Root placement/animation regression: **7 cases passed**, including explicit
  rejection of unrepresentable parent shear.
- Unsupported child Action/driver/NLA rejection: **3 passed**.
- Native material-effect creation: **16 passed**; fresh-process verification:
  **53 passed**. Maximum synthetic Cycles RGBA absolute error `4.054e-7`.

Here “native material effects” means ordinary Blender nodes/Actions. These
checks do not use the external C++ physics bridge or any Live2D model.

## Publication boundary

- Source tree audit: **69 explicitly listed files**, only four original 64×64
  PNG fixtures, no SDK/Core/Framework source or binary, user model, generated
  model/cache, `.blend`, compiled bridge, logs or credentials found.
- All 20 add-on implementation modules and the original C++ bridge are
  byte-identical to frozen v0.2. The existing MIT LICENSE notices are unchanged.
- Documentation, public packaging/audit tools and a read-only GitHub Actions
  workflow were prepared separately. GitHub CI is not claimed as run by these
  local results.
- The frozen release was checked before and after preparation and was unchanged.

This validation does not establish real-model rendered parity, full masking,
advanced blends, later private fixes, native bridge compatibility on other
platforms, or support for Blender versions other than 4.3.2.
