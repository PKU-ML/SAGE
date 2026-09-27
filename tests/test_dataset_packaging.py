import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tarfile

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/package_native_dataset.py'
spec = importlib.util.spec_from_file_location('package_native_dataset', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_hardlinked_data_is_embedded_and_verified(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'original').write_bytes(b'episode payload')
    os.link(data / 'original', data / 'linked')
    files = {name: dict(bytes=15, sha256=hashlib.sha256(b'episode payload').hexdigest())
             for name in ('original', 'linked')}
    out = tmp_path / 'data.tar'
    result = module.package(files, data, out, 'example', 'train')
    assert result['files'] == 2
    with tarfile.open(out) as archive:
        for name in files:
            assert archive.getmember(name).isfile()
            assert archive.extractfile(name).read() == b'episode payload'
        assert json.load(archive.extractfile('DATASET_MANIFEST.json'))['files'] == files
    with pytest.raises(FileExistsError):
        module.package(files, data, out, 'example', 'train')


def test_corrupt_data_is_not_finalized(tmp_path):
    (tmp_path / 'data').write_bytes(b'bad')
    out = tmp_path / 'bad.tar'
    with pytest.raises(ValueError, match='checksum'):
        module.package({'data': dict(bytes=3, sha256='0' * 64)}, tmp_path, out, 'example', 'train')
    assert not out.exists()


def test_install_round_trip_and_reject_wrong_existing_file(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(path.parent))
    import install_native_dataset as installer
    payload = b'portable episode'
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'episode.h5').write_bytes(payload)
    files = {'episode.h5': dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())}
    monkeypatch.setattr(installer, 'registered_files', lambda suite, split: files)
    archive = tmp_path / 'data.tar'
    module.package(files, source, archive, 'example', 'train')
    target = tmp_path / 'installed'
    target.mkdir()
    stale = target / 'episode.h5.partial'
    stale.write_bytes(b'interrupted earlier attempt')
    assert installer.install(archive, target, 'example', 'train')['verified']
    assert (target / 'episode.h5').read_bytes() == payload
    assert stale.read_bytes() == b'interrupted earlier attempt'
    assert list(target.glob('*.partial')) == [stale]
    assert installer.install(archive, target, 'example', 'train')['verified']
    (target / 'episode.h5').write_bytes(b'incorrect')
    with pytest.raises(ValueError, match='Existing data'):
        installer.install(archive, target, 'example', 'train')


def test_install_rejects_archive_links(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(path.parent))
    import install_native_dataset as installer
    monkeypatch.setattr(installer, 'registered_files', lambda suite, split: {'episode.h5': {}})
    archive = tmp_path / 'linked.tar'
    with tarfile.open(archive, 'w') as handle:
        info = tarfile.TarInfo('episode.h5')
        info.type = tarfile.SYMTYPE
        info.linkname = '/private/data'
        handle.addfile(info)
        handle.addfile(tarfile.TarInfo('DATASET_MANIFEST.json'))
    with pytest.raises(ValueError, match='regular files'):
        installer.install(archive, tmp_path / 'installed', 'example', 'train')
