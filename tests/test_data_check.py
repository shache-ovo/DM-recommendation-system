import hashlib
import json

from dmrec.data_check import check_files, sha256_of


def write_manifest(tmp_path, name, content):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"name": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}]}),
        encoding="utf-8",
    )
    return manifest


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc" * 1000)
    assert sha256_of(path, chunk_size=7) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_matching_file_has_no_problems(tmp_path):
    (tmp_path / "a.csv").write_bytes(b"x,y\n1,2\n")
    manifest = write_manifest(tmp_path, "a.csv", b"x,y\n1,2\n")
    assert check_files(manifest, tmp_path) == []


def test_missing_size_and_hash_mismatches_are_reported(tmp_path):
    manifest = write_manifest(tmp_path, "a.csv", b"x,y\n1,2\n")
    assert "파일 없음" in check_files(manifest, tmp_path)[0]
    (tmp_path / "a.csv").write_bytes(b"x,y\n1,2\n3,4\n")
    assert "크기" in check_files(manifest, tmp_path)[0]
    (tmp_path / "a.csv").write_bytes(b"x,y\n9,9\n")
    assert "sha256" in check_files(manifest, tmp_path)[0]
