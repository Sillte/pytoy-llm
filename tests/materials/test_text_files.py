from pathlib import Path

from pytoy_llm.materials.text_files.models import TextFileLocator, TextFileContent, TextFileCollection

def test_structured_text():
    locator1 = TextFileLocator(
        id="file1",
        path=Path("docs/file1.txt"),
        timestamp=1675200000.0
    )

    locator2 = TextFileLocator(
        id="file2",
        path=Path("docs/file2.txt"),
        timestamp=1675300000.0
    )

    locators = {
        locator1.id: locator1,
        locator2.id: locator2,
    }

    # --- テスト用 contents ---
    content1 = TextFileContent(
        id="file1",
        entry="All",
        body="これはファイル1の全文テキストです。"
    )

    content2 = TextFileContent(
        id="file2",
        entry="Summary",
        body="これはファイル2のサマリです。"
    )

    contents = {
        content1.id: content1,
        content2.id: content2,
    }

    collection = TextFileCollection(locators=locators, contents=contents)
    print(collection.structured_text)
    assert "テキストです。" in collection.structured_text