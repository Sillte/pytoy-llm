from functools import wraps
from typing import Callable

from pytoy_llm.idea import MetadataDeserializationError, OutsidePathError
from pytoy_llm.tools.errors import ToolError, ToolErrorKind


def tool_discovery_boundary[R](
    func: Callable[..., R],
) -> Callable[..., R | ToolError]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> R | ToolError:
        try:
            return func(*args, **kwargs)
        except OutsidePathError as exc:
            return ToolError(
                kind=ToolErrorKind.PERMISSION_DENIED,
                msg=str(exc),
                retry=False,
            )
        except FileNotFoundError as exc:
            return ToolError(
                kind=ToolErrorKind.NOT_FOUND,
                msg=str(exc),
                retry=False,
            )
        except OSError as exc:
            return ToolError(
                kind=ToolErrorKind.IO_ERROR,
                msg=str(exc),
                retry=False,
            )

    return wrapper


def tool_inspection_boundary[R](
    func: Callable[..., R],
) -> Callable[..., R | ToolError]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> R | ToolError:
        try:
            return func(*args, **kwargs)
        except MetadataDeserializationError:
            return ToolError(
                kind=ToolErrorKind.PARSE_ERROR,
                msg="The specified IdeaNote is broken.",
                retry=False,
            )
        except ValueError as exc:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=str(exc))
        except FileNotFoundError as exc:
            return ToolError(
                kind=ToolErrorKind.NOT_FOUND,
                msg=str(exc),
                retry=False,
            )
        except OutsidePathError as exc:
            return ToolError(
                kind=ToolErrorKind.PERMISSION_DENIED,
                msg=str(exc),
                retry=False,
            )
        except OSError as exc:
            return ToolError(
                kind=ToolErrorKind.IO_ERROR,
                msg=str(exc),
                retry=False,
            )

    return wrapper
