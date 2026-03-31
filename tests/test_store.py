from __future__ import annotations

import json

from src.store import ArticleStore


class TestArticleStore:
    def test_read_tolerates_invalid_utf8_in_json_strings(self, tmp_path):
        path = tmp_path / "articles.json"
        path.write_bytes(b'[{"headline":"Good\xfeHeadline","body":"Body"}]')

        store = ArticleStore(path=str(path))

        entries = store.list_all()

        assert entries[0]["headline"] == "Good\ufffdHeadline"

    def test_write_uses_utf8_encoding(self, tmp_path):
        path = tmp_path / "articles.json"
        store = ArticleStore(path=str(path))

        store.add({"headline": "Cafe", "body": "Body"})

        raw = path.read_bytes()
        assert json.loads(raw.decode("utf-8"))[0]["headline"] == "Cafe"
