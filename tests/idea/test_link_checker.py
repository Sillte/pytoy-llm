from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError

from pytoy_llm.idea import IdeaLink, LinkReachabilityChecker, TextPosition, TextRange


class FakeResponse:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        pass


def make_link(uri: str) -> IdeaLink:
    return IdeaLink(
        source_path=Path("source.md"),
        source_text_range=TextRange(TextPosition(0, 0), TextPosition(0, 1)),
        uri=uri,
    )


def test_http_success_is_true(monkeypatch) -> None:
    monkeypatch.setattr(
        "pytoy_llm.idea.link_checker.urlopen",
        lambda request, timeout: FakeResponse(200),
    )

    assert LinkReachabilityChecker().check(make_link("https://example.com")) is True


def test_http_not_found_is_false(monkeypatch) -> None:
    def raise_not_found(request, timeout):
        raise HTTPError(request.full_url, 404, "missing", Message(), None)

    monkeypatch.setattr("pytoy_llm.idea.link_checker.urlopen", raise_not_found)

    assert LinkReachabilityChecker().check(make_link("https://example.com")) is False


def test_http_head_method_falls_back_to_get(monkeypatch) -> None:
    methods: list[str] = []

    def respond(request, timeout):
        methods.append(request.method)
        if request.method == "HEAD":
            raise HTTPError(request.full_url, 405, "method not allowed", Message(), None)
        return FakeResponse(200)

    monkeypatch.setattr("pytoy_llm.idea.link_checker.urlopen", respond)

    assert LinkReachabilityChecker().check(make_link("https://example.com")) is True
    assert methods == ["HEAD", "GET"]


def test_http_connection_failure_is_unknown(monkeypatch) -> None:
    monkeypatch.setattr(
        "pytoy_llm.idea.link_checker.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(URLError("offline")),
    )

    assert LinkReachabilityChecker().check(make_link("https://example.com")) is None
