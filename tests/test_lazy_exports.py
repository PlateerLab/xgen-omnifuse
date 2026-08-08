from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_sqlite_exports_do_not_eagerly_import_unrelated_backends():
    source = Path(__file__).resolve().parents[1] / "src"
    code = f"""
import sys
sys.path.insert(0, {str(source)!r})
import omnifuse
assert 'omnifuse.backends.memory' not in sys.modules
assert 'omnifuse.backends.fuseki' not in sys.modules
assert callable(omnifuse.build_sqlite_index)
assert 'omnifuse.backends.sqlite_snapshot' in sys.modules
assert 'omnifuse.backends.memory' not in sys.modules
assert 'omnifuse.backends.fuseki' not in sys.modules
assert callable(omnifuse.build_inmemory)
assert 'omnifuse.backends.memory' in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
