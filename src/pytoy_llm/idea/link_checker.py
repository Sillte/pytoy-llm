from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .domain.links import IdeaLink
from .domain.uri import Uri


class HttpExistenceTester:
    def __init__(self, *, timeout: float = 1.0):
        self._timeout = timeout

    def test(self, uri: Uri) -> bool | None:
        """Return whether a URI is reachable.

        Returns ``True`` when the target responds successfully, ``False`` when
        it is known not to exist, and ``None`` when the URI is unsupported or
        reachability cannot be determined.
        """
        if uri.scheme in {"http", "https"}:
            return self._test_http(str(uri))

        return None

    def _test_http(self, uri: str) -> bool | None:
        try:
            request = Request(
                uri,
                method="HEAD",
                headers={"User-Agent": "pytoy-llm/1.0"},
            )

            with urlopen(request, timeout=self._timeout) as response:
                return 200 <= response.status < 300

        except HTTPError as exc:
            if exc.code == 404:
                return False

            if exc.code != 405:
                return None

        except (URLError, TimeoutError, OSError):
            return None

        try:
            request = Request(
                uri,
                method="GET",
                headers={"User-Agent": "pytoy-llm/1.0"},
            )

            with urlopen(request, timeout=self._timeout) as response:
                return 200 <= response.status < 300

        except HTTPError as exc:
            if exc.code == 404:
                return False
            return None

        except (URLError, TimeoutError, OSError):
            return None


class LinkReachabilityChecker:
    def __init__(self, *, timeout: float = 1.0):
        self._tester = HttpExistenceTester(timeout=timeout)

    def check(self, link: IdeaLink) -> bool | None:
        """Check whether an idea link target is reachable.

        The result is ``True`` for a reachable target, ``False`` when the
        target is known to be unavailable, and ``None`` when reachability
        cannot be determined. HTTP checks may perform synchronous I/O and are
        bounded by the configured timeout.
        """
        if link.target_path:
            return link.target_path.exists()
        else:
            return self._tester.test(link.uri)
