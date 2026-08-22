from __future__ import annotations

from collections.abc import Sequence
from os.path import commonpath
from pathlib import Path
from typing import Self

from pydantic import BaseModel


class PathNode(BaseModel, frozen=True):
    path: Path
    children: tuple[PathNode, ...]


class PathTree:
    def __init__(self, root: PathNode) -> None:
        self._root = root

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[Path],
        root_path: Path | None = None,
    ) -> Self:
        if not paths:
            raise ValueError("Empty `paths`")

        paths = [path.resolve() for path in paths]
        if root_path is not None:
            root_path = root_path.resolve()
        else:
            root_path = Path(commonpath(paths)).resolve()
            if root_path.is_file():
                root_path = root_path.parent

        class MutableNode:
            def __init__(self, path: Path) -> None:
                self.path = path
                self.children: dict[str, MutableNode] = {}

        root = MutableNode(root_path)

        for path in paths:
            try:
                relative = path.relative_to(root_path)
            except ValueError as e:
                raise ValueError(f"Path `{path}` is outside root `{root_path}`.") from e

            current = root

            for part in relative.parts:
                child = current.children.get(part)

                if child is None:
                    child = MutableNode(current.path / part)
                    current.children[part] = child

                current = child

        def freeze(node: MutableNode) -> PathNode:
            return PathNode(
                path=node.path,
                children=tuple(
                    freeze(child)
                    for child in sorted(
                        node.children.values(),
                        key=lambda child: child.path.name,
                    )
                ),
            )

        return cls(root=freeze(root))

    def render(self, include_root: bool = True) -> str:
        lines: list[str] = []

        def visit(node: PathNode, prefix: str = "", is_last: bool = True) -> None:
            branch = "└── " if is_last else "├── "
            lines.append(f"{prefix}{branch}{node.path.name}")

            child_prefix = prefix + ("    " if is_last else "│   ")

            for index, child in enumerate(node.children):
                visit(
                    child,
                    prefix=child_prefix,
                    is_last=index == len(node.children) - 1,
                )

        if include_root:
            lines.append(self._root.path.name)
        for index, child in enumerate(self._root.children):
            visit(child, prefix="", is_last=index == len(self._root.children) - 1)

        return "\n".join(lines)


if __name__ == "__main__":
    tree = PathTree.from_paths([Path("b.py"), Path("./fa.py")])
    print(tree.render(include_root=True))
