"""Create an isolated SAPIEN startup fix; never edit the installed package.

SAPIEN 3.0.0b1 probes every NVIDIA GPU before checking explicit Vulkan/EGL
configuration. Honor these overrides first, without changing simulation code.
Only set them to existing driver ICD files on the host running evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def patch_startup(source):
    if 'def _ensure_vulkan_icd():' not in source or 'def _ensure_egl_icd():' not in source:
        raise ValueError('Unrecognized SAPIEN startup module')
    for name, condition in (
        ('_ensure_vulkan_icd', 'os.environ.get("VK_ICD_FILENAMES")'),
        ('_ensure_egl_icd', 'os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") or os.environ.get("__EGL_VENDOR_LIBRARY_DIRS")'),
    ):
        anchor = f'def {name}():\n    if os.system("nvidia-smi > /dev/null 2>&1") != 0:'
        if source.count(anchor) != 1:
            raise ValueError(f'Unexpected startup implementation: {name}')
        replacement = f'def {name}():\n    if {condition}:\n        return\n    if os.system("nvidia-smi > /dev/null 2>&1") != 0:'
        source = source.replace(anchor, replacement)
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-dir', type=Path, required=True, help='Existing site-packages/sapien directory')
    p.add_argument('--out', type=Path, required=True, help='New isolated PYTHONPATH directory')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new overlay directory; existing environments are never modified')
    source = args.package_dir / '_vulkan_tricks.py'
    original = source.read_bytes()
    modified = patch_startup(original.decode()).encode()
    shutil.copytree(args.package_dir, args.out / 'sapien', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    libraries = args.package_dir.parent / 'sapien.libs'
    if libraries.is_dir():
        shutil.copytree(libraries, args.out / 'sapien.libs')
    (args.out / 'sapien/_vulkan_tricks.py').write_bytes(modified)
    (args.out / 'startup_patch.json').write_text(json.dumps({
        'scope': 'Honor explicit ICD settings before optional nvidia-smi autodetection',
        'original_sha256': hashlib.sha256(original).hexdigest(),
        'patched_sha256': hashlib.sha256(modified).hexdigest(),
        'physics_and_renderer_binaries_modified': False,
    }, indent=2) + '\n')
    print(args.out.resolve())


if __name__ == '__main__':
    main()
