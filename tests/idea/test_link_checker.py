from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from pytoy_llm.idea import IdeaLink, LinkReachabilityChecker, TextPosition, TextRange
from pytoy_llm.idea.domain.uri import Uri
from pytoy_llm.idea.link_resolvers.resolvers import UriLocalPathResolver


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
        uri=Uri.from_any(uri),
    )


def test_registered_scheme_decodes_path_under_its_root(tmp_path: Path) -> None:
    root = tmp_path / "idea"
    root.mkdir()
    resolver = UriLocalPathResolver({"idea": root})

    assert resolver.resolve(Uri.from_any("idea:///notes/a%20b.md")) == (root / "notes" / "a b.md")


def test_registered_scheme_rejects_targets_outside_its_root(tmp_path: Path) -> None:
    root = tmp_path / "idea"
    root.mkdir()
    resolver = UriLocalPathResolver({"idea": root})

    with pytest.raises(ValueError, match="outside its registered root"):
        resolver.resolve(Uri.from_any("idea:///%2e%2e/outside.md"))


def test_relative_uri_requires_absolute_base_and_decodes_path(tmp_path: Path) -> None:
    resolver = UriLocalPathResolver({})
    uri = Uri.from_any("./relative/a%20b.md")

    with pytest.raises(ValueError, match="must be an absolute path"):
        resolver.resolve(uri, Path("relative"))

    assert resolver.resolve(uri, tmp_path) == tmp_path / "relative" / "a b.md"


def test_file_scheme_is_not_supported_yet() -> None:
    resolver = UriLocalPathResolver({})
    uri = Uri.from_any("file:///tmp/example.md")

    assert not resolver.is_target_scheme(uri.scheme)
    with pytest.raises(ValueError, match="not registered"):
        resolver.resolve(uri)
    assert LinkReachabilityChecker().check(make_link(str(uri))) is None


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
