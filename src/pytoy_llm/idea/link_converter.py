from typing import assert_never

from .domain.links import (
    IdeaLink,
    ResolvedLink,
    ResolvedLocalLink,
    ResolvedRemoteLink,
    UnresolvedIdeaLink,
    UnresolvedLink,
)
from .domain.uri import Uri


class IdeaLinkConverter:
    def __init__(self):
        pass

    def convert(
        self,
        inner_link: ResolvedLink | UnresolvedLink,
    ) -> IdeaLink | UnresolvedIdeaLink:
        match inner_link:
            case ResolvedLocalLink():
                uri = inner_link.target_uri
                link = IdeaLink(
                    source_path=inner_link.source_path,
                    source_text_range=inner_link.link_source.text_range,
                    uri=Uri.from_any(uri),
                    target_location=inner_link.target_location,
                    target_path=inner_link.target_path,
                )
                return link
            case ResolvedRemoteLink():
                link = IdeaLink(
                    source_path=inner_link.source_path,
                    source_text_range=inner_link.link_source.text_range,
                    uri=Uri.from_any(inner_link.url),
                    target_path=None,
                )
            case UnresolvedLink():
                link = UnresolvedIdeaLink(
                    source_path=inner_link.source_path,
                    reason=inner_link.reason,
                    source_text_range=inner_link.link_source.text_range,
                )
            case _:
                assert_never(inner_link)
        return link
