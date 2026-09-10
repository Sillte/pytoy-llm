import os
from pathlib import Path
from urllib.parse import unquote, urlsplit


def path_from_file_uri(uri: str) -> Path | None:
    parsed = urlsplit(uri)
    if parsed.scheme.lower() != "file":
        return None

    path = unquote(parsed.path)
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        path = f"//{parsed.netloc}{path}"

    if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]

    return Path(path)
