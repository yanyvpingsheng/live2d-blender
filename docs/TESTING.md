# Testing / 测试说明

This is a source-only publication of the frozen 0.2.0 add-on. Publication
preparation changes packaging, documentation and source hygiene, not converter
implementation. It does not incorporate later private face-mask changes.

## SDK-free Python tests

Use Python 3.12 and install the test dependencies from the normal Python package
index. The add-on itself uses Blender's Python/NumPy installation.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-test.txt
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_*.py' -v
```

The native-bridge cases are opt-in and skip when `L2D_PHYSICS_BRIDGE` and
`L2D_TEST_MOC` are unset. SDK-free success does not test real `.moc3` files or
prove visual parity. Never provide SDK/model secrets or files to public CI.

## Source audit and installer

Run the audit on a clean source tree before creating generated files:

```sh
PYTHONDONTWRITEBYTECODE=1 python scripts/audit_public.py
PYTHONDONTWRITEBYTECODE=1 python scripts/package.py
```

The audit requires an exact match with `PUBLIC_FILES.txt` (apart from `.git/`)
and scans common secret/path patterns. It is a safety aid, not a substitute for
reviewing every new file or permission to publish external assets. A normal
local build creates ignored `dist/` output; use a clean checkout or temporary
copy for subsequent publication audits.

`scripts/package.py` builds only the installable add-on from explicitly listed
files. It never recursively includes local models, SDK files or generated demos.

## Blender integration tests

Validated platform: **Linux x86_64 / Blender 4.3.2**. Other Blender versions,
Windows and macOS have not passed this validation. The declared 4.2 API floor is
not evidence of a completed 4.2 test run.

Use a disposable checkout and isolated Blender user directories. Installation
tests install the ZIP into that temporary user add-ons directory; tests also
write original synthetic demonstrations and result JSON into the checkout.

```sh
# Linux shell; run from a disposable copy of this repository.
export BLENDER_USER_SCRIPTS="$(mktemp -d)"
export BLENDER_USER_CONFIG="$(mktemp -d)"
export BLENDER_USER_DATAFILES="$(mktemp -d)"
export PYTHONDONTWRITEBYTECODE=1
python scripts/package.py
blender -b --factory-startup --disable-autoexec --python tests/blender_v02_install_convert.py
blender -b --factory-startup --disable-autoexec --python tests/blender_v02_reopen.py
blender -b --factory-startup --disable-autoexec --python tests/blender_smoke.py
blender -b --factory-startup --disable-autoexec --python tests/blender_conversion_failures.py
blender -b --factory-startup --disable-autoexec --python tests/blender_skinning_affine_cases.py
blender -b --factory-startup --disable-autoexec --python tests/blender_v02_review_root_transform.py
blender -b --factory-startup --disable-autoexec --python tests/blender_v02_child_animation_rejection.py
blender -b --factory-startup --disable-autoexec --python tests/blender_native_effects_smoke.py
blender -b --factory-startup --disable-autoexec demo/native_effects_smoke.blend --python tests/blender_native_effects_verify.py
```

The install test regenerates the original SDK-free demo before the reopen test
opens it. The reopen verifier deliberately imports no add-on, loads no Core,
disables automatic script execution, and checks saved ordinary Blender data.

The legacy `blender_skinning_smoke.py` / `blender_skinning_reopen.py` pair uses
its own generated expectations; run that pair together if needed. It overwrites
the same demonstration filename, so do not mix its output with the v0.2 verifier.

## Interpretation

- Passing reopen tests proves persistence and independence from the conversion
  runtime, not exact agreement with previously unseen source poses.
- Training vertex error in model units or normalized pixels is not rendered
  image error. A low training error does not certify holdout frames.
- Synthetic fixtures are project-owned geometry and tiny procedural textures.
  No real model or model-derived expectation arrays are committed.
- Full clipping masks, advanced blends, offscreen rendering and cross-pass
  ordering remain outside the integrated v0.2 conversion path.
- Native SDK tests are not part of public CI. The optional local C++ bridge is
  source only; its built library contains external Framework code and must stay
  outside this public source tree and distributed add-on.

Publication validation results are recorded in [VALIDATION.md](VALIDATION.md).
