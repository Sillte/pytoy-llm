import re
from typing import Final, Iterator

from pytoy_llm.idea.domain.links import (
    MarkdownLinkSource,
    TextPosition,
    TextRange,
)
from pytoy_llm.idea.domain.uri import Uri


def position_at(text: str, offset: int) -> TextPosition:
    line = text.count("\n", 0, offset)

    last_newline = text.rfind("\n", 0, offset)

    if last_newline == -1:
        col = offset
    else:
        col = offset - last_newline - 1

    return TextPosition(
        line=line,
        col=col,
    )


def range_at(
    text: str,
    start: int,
    end: int,
) -> TextRange:
    return TextRange(
        start=position_at(text, start),
        end=position_at(text, end),
    )


class NaiveLinkSourceExtractor:
    LINK_RE: Final[re.Pattern] = re.compile(
        r"""
    (?<!!)\[
        (?P<markdown_caption>[^\]]*)
    \]\(
        (?:<(?P<markdown_angle_target>[^>]+)>
        |
        (?P<markdown_target>[^\s)]+))
    \)
    """,
        re.VERBOSE,
    )

    def extract(self, text: str) -> Iterator[MarkdownLinkSource]:
        for match in self.LINK_RE.finditer(text):
            markdown_target = match.group("markdown_angle_target") or match.group("markdown_target")
            assert markdown_target is not None
            markdown_caption = match.group("markdown_caption")

            target, seq, fragment = markdown_target.strip().partition("#")

            yield MarkdownLinkSource(
                text_range=range_at(
                    text,
                    match.start(),
                    match.end(),
                ),
                target=Uri.from_any(target),
                caption=(markdown_caption if markdown_caption is not None else None),
                fragment=(fragment if seq else None),
            )
