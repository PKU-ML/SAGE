import hashlib
import json
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from sage import robotwin_runtime as MODULE


class RuntimeInstallTests(unittest.TestCase):
    def test_split_archive_assembly_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mirror = root / 'mirror'
            mirror.mkdir()
            parts = []
            for i, data in enumerate((b'first', b'second')):
                name = f'archive.part{i}'
                (mirror / name).write_bytes(data)
                parts.append({'filename': name, 'bytes': len(data),
                              'sha256': hashlib.sha256(data).hexdigest()})
            record = {'filename': 'archive.tar.gz', 'bytes': 11, 'parts': parts,
                      'sha256': hashlib.sha256(b'firstsecond').hexdigest()}
            for _ in range(2):
                MODULE.download_archive({'archive': record}, root / 'downloads', mirror)
                self.assertEqual((root / 'downloads/archive.tar.gz').read_bytes(), b'firstsecond')

    def test_bundled_startup_catalogs_match_inventory(self):
        folder = MODULE.ROOT / 'runtime/robotwin'
        inventory = json.loads((folder / 'bootstrap_assets.json').read_text())
        self.assertEqual(set(inventory), {'assets/objects/objaverse/list.json', 'assets/objects/same.json'})
        for record in inventory.values():
            content = (MODULE.ROOT / record['bundled_file']).read_bytes()
            self.assertEqual(len(content), record['bytes'])
            self.assertEqual(hashlib.sha256(content).hexdigest(), record['sha256'])

    def pack(self, path, names, content=b'fixture'):
        with tarfile.open(path, 'w:gz') as archive:
            for name in names:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

    def test_install_resume_and_verify(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = {'assets/item.bin': {'bytes': 7, 'sha256': hashlib.sha256(b'fixture').hexdigest()}}
            self.pack(root / 'test.tar.gz', expected)
            for _ in range(2):
                self.assertEqual(MODULE.extract_verified(root / 'test.tar.gz', root / 'out', expected), 1)
            self.assertEqual(MODULE.verify_tree(root / 'out', expected), 1)

    def test_reject_unregistered_and_duplicate_members(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = {'ok': {'bytes': 7, 'sha256': hashlib.sha256(b'fixture').hexdigest()}}
            for names in (['../escape'], ['ok', 'ok'], ['unexpected']):
                self.pack(root / 'test.tar.gz', names)
                with self.assertRaises(ValueError):
                    MODULE.extract_verified(root / 'test.tar.gz', root / 'out', expected)

    def test_checksum_and_missing_member(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = {'ok': {'bytes': 7, 'sha256': '0' * 64}}
            self.pack(root / 'test.tar.gz', ['ok'])
            with self.assertRaises(ValueError):
                MODULE.extract_verified(root / 'test.tar.gz', root / 'out', expected)
            self.pack(root / 'test.tar.gz', [])
            with self.assertRaises(ValueError):
                MODULE.extract_verified(root / 'test.tar.gz', root / 'out', expected)

    def test_symlink_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with tarfile.open(root / 'test.tar.gz', 'w:gz') as archive:
                member = tarfile.TarInfo('ok')
                member.type = tarfile.SYMTYPE
                member.linkname = '/outside'
                archive.addfile(member)
            expected = {'ok': {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}}
            with self.assertRaises(ValueError):
                MODULE.extract_verified(root / 'test.tar.gz', root / 'out', expected)


if __name__ == '__main__':
    unittest.main()
