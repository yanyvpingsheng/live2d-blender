# Dependency and asset boundary

- Add-on Python and native shim source: written for this project, MIT.
- Synthetic Sprout geometry, four tiny procedural textures, and generated demo: original project fixtures, MIT; not Live2D sample assets.
- Blender: external application, GPL. https://www.blender.org/about/license/
- NumPy: external package normally included with Blender, BSD-3-Clause; not bundled.
- Pillow: optional external PNG loader for standalone tests/reference rendering; not bundled.
- Live2D Cubism Core: proprietary, user-provided shared library; **not bundled**. https://www.live2d.com/eula/live2d-proprietary-software-license-agreement_en.html
- Cubism Native Framework: independent SDK installation, Live2D Open Software License; **not copied into the add-on**. https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html
- Live2D publication review / Expandable Application: separate release gate. https://www.live2d.com/en/sdk/license/
- User-provided models and trial assets: not included. Their original owners retain all rights; use and redistribution of originals or converted assets require the applicable permission. Preserve any required trial watermark.

No proprietary SDK binary/header, official Framework source, official sample model, user model, reverse-engineered parser, or alternative Core binary is part of this package. Builds of an optional native shim are local to the user's SDK installation. Selecting a native library executes native code; use only an official/trusted build. The project does not make a legal conclusion about combined distribution.
