"""The startup overlay must not alter simulator code or run driver autodetection
when the caller has explicitly configured ICD discovery.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location(
    "sapien_overlay", Path(__file__).resolve().parents[1] / "scripts/prepare_sapien_overlay.py"
)
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)

SOURCE = '''def _ensure_vulkan_icd():
    if os.system("nvidia-smi > /dev/null 2>&1") != 0:
        return
    calls.append("vulkan")

def _ensure_egl_icd():
    if os.system("nvidia-smi > /dev/null 2>&1") != 0:
        return
    calls.append("egl")
'''


class StartupTests(unittest.TestCase):
    def execute(self, environ):
        probes, calls = [], []
        def system(command):
            probes.append(command)
            return 0
        scope = {"os": SimpleNamespace(environ=environ, system=system), "calls": calls}
        exec(overlay.patch_startup(SOURCE), scope)
        scope["_ensure_vulkan_icd"]()
        scope["_ensure_egl_icd"]()
        return probes, calls

    def test_explicit_icds_skip_autodetection(self):
        self.assertEqual(self.execute({"VK_ICD_FILENAMES": "/driver/vk.json",
            "__EGL_VENDOR_LIBRARY_FILENAMES": "/driver/egl.json"}), ([], []))

    def test_egl_directory_override(self):
        probes, calls = self.execute({"__EGL_VENDOR_LIBRARY_DIRS": "/driver"})
        self.assertEqual(len(probes), 1)
        self.assertEqual(calls, ["vulkan"])

    def test_default_behavior_preserved(self):
        probes, calls = self.execute({})
        self.assertEqual(len(probes), 2)
        self.assertEqual(calls, ["vulkan", "egl"])

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            overlay.patch_startup("def changed(): pass")

    def test_double_patch_rejected(self):
        with self.assertRaises(ValueError):
            overlay.patch_startup(overlay.patch_startup(SOURCE))


if __name__ == "__main__":
    unittest.main()
