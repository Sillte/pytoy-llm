import pytest
from pathlib import Path
from pytoy_llm.materials.text_files import TextFile, TextFilesCollector
import time

def test_textfile_from_path(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("Hello\nWorld", encoding="utf8")

    tf = TextFile.from_path(file_path, workspace=tmp_path)
    assert tf.path == file_path
    assert tf.location == ("example.txt",)
    assert abs(tf.timestamp - file_path.stat().st_mtime) < 1e-3
    assert tf.text == "Hello\nWorld"

    structure = tf.structured_text
    assert "<file>" in structure
    assert "Hello" in structure
    assert "World" in structure


