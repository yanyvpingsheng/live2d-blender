"""Build this original adapter against a user-supplied Cubism 5-r.5 SDK.

No download, license acceptance, SDK copying, or package installation occurs.
The output contains licensed Framework code. Keep it OUTSIDE distributable ZIPs.
Linux/macOS compiler flow; Windows requires a separately validated MSVC build.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

SOURCES = (
    'CubismFramework.cpp',
    'Id/CubismId.cpp', 'Id/CubismIdManager.cpp',
    'Math/CubismMath.cpp', 'Math/CubismVector2.cpp',
    'Model/CubismMoc.cpp', 'Model/CubismModel.cpp',
    'Model/CubismModelMultiplyAndScreenColor.cpp',
    'Physics/CubismPhysics.cpp', 'Physics/CubismPhysicsJson.cpp',
    'Effect/CubismPose.cpp', 'Rendering/csmBlendMode.cpp',
    'Type/csmString.cpp', 'Utils/CubismDebug.cpp',
    'Utils/CubismJson.cpp', 'Utils/CubismString.cpp',
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', required=True, type=Path)
    parser.add_argument('--core', required=True, type=Path, help='Official shared Core library')
    parser.add_argument('--output', required=True, type=Path, help='Private output .so/.dylib outside addon')
    parser.add_argument('--compiler', default='c++')
    args = parser.parse_args()
    if sys.platform == 'win32':
        parser.error('This build helper has not been validated with MSVC. Use CMakeLists.txt.')
    compiler = shutil.which(args.compiler)
    if not compiler: parser.error('C++ compiler not found; install from a trusted official toolchain')
    sdk, core, output = args.sdk.resolve(), args.core.resolve(), args.output.resolve()
    framework = sdk / 'Framework' / 'src'
    sources = [framework / name for name in SOURCES]
    if not core.is_file() or any(not p.is_file() for p in sources):
        parser.error('SDK 5-r.5 source files or shared Core library missing')
    addon = Path(__file__).resolve().parents[1] / 'addon'
    if output.is_relative_to(addon): parser.error('Do not place compiled SDK code inside distributable addon')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, '-std=c++17', '-O2', '-fPIC', '-fvisibility=hidden',
               '-dynamiclib' if sys.platform == 'darwin' else '-shared',
               '-I' + str(framework), '-I' + str(sdk / 'Core' / 'include'),
               str(Path(__file__).with_name('physics_bridge.cpp')),
               *(str(p) for p in sources), str(core), '-Wl,-rpath,' + str(core.parent),
               '-o', str(output)]
    if sys.platform != 'darwin': command.append('-Wl,--no-undefined')
    subprocess.run(command, check=True)
    print(output)

if __name__ == '__main__':
    main()
