import hashlib
import os
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location, spec_from_loader
from inspect import isabstract, isclass
from pathlib import Path
from sys import meta_path, modules
from types import ModuleType

from pytoy_llm.tool.plugin.contract import LLMToolsPlugin
from pytoy_llm.tool.plugin.models import LLMPluginCollection

_PLUGIN_PREFIX = "_pytoy_llm_plugin_"
_DIRECTORY_PACKAGE_PREFIX = "_pytoy_llm_package_"
_LEGACY_COLLECTION_ATTRIBUTE = "__pytoy_llm_plugin_collection__"


class PluginLoadError(RuntimeError):
    """The requested path does not satisfy the plugin loading contract."""


class _NoPluginFound(PluginLoadError):
    """A valid Python module contains no concrete plugin implementation."""


class _PluginSourceLoader(Loader):
    def __init__(self, path: Path, source: bytes) -> None:
        self._path = path
        self._source = source

    def exec_module(self, module: ModuleType) -> None:
        code = compile(self._source, str(self._path), "exec")
        exec(code, module.__dict__)


class _PluginSourceFinder(MetaPathFinder):
    def __init__(self, package_name: str, sources: dict[str, tuple[Path, bytes]]) -> None:
        self._module_prefix = f"{package_name}."
        self._sources = sources

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._module_prefix):
            return None
        module_name = fullname.removeprefix(self._module_prefix)
        if "." in module_name:
            return None
        source_file = self._sources.get(module_name)
        if source_file is None:
            return None
        path, source = source_file
        loader = _PluginSourceLoader(path, source)
        return spec_from_loader(fullname, loader, origin=str(path))


def _module_prefix(path: Path) -> str:
    path_digest = hashlib.sha256(os.fsencode(os.path.normcase(str(path)))).hexdigest()
    return f"{_PLUGIN_PREFIX}{path_digest}_"


def _module_name(path: Path, source: bytes) -> str:
    source_digest = hashlib.sha256(source).hexdigest()
    return f"{_module_prefix(path)}{source_digest}"


def _directory_identity(
    directory: Path,
    sources: dict[str, tuple[Path, bytes]],
) -> tuple[str, str]:
    path_digest = hashlib.sha256(os.fsencode(os.path.normcase(str(directory)))).hexdigest()
    prefix = f"{_DIRECTORY_PACKAGE_PREFIX}{path_digest}_"
    content_digest = hashlib.sha256()
    for module_name, (path, source) in sorted(sources.items()):
        encoded_name = os.fsencode(path.name)
        content_digest.update(len(encoded_name).to_bytes(4, "big"))
        content_digest.update(encoded_name)
        content_digest.update(len(source).to_bytes(8, "big"))
        content_digest.update(source)
    return f"{prefix}{content_digest.hexdigest()}", prefix


def _directory_package(package_name: str, directory: Path) -> ModuleType:
    package = modules.get(package_name)
    if package is not None:
        if not isinstance(package, ModuleType):
            raise PluginLoadError(f"Invalid plugin package module cached as {package_name}")
        return package

    spec = ModuleSpec(package_name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(directory)]
    package = module_from_spec(spec)
    modules[package_name] = package
    return package


def _remove_directory_package(package_name: str) -> None:
    module_prefix = f"{package_name}."
    for module_name in tuple(modules):
        if module_name == package_name or module_name.startswith(module_prefix):
            modules.pop(module_name, None)


def _remove_previous_directory_versions(path_prefix: str, package_name: str) -> None:
    current_prefix = f"{package_name}."
    for module_name in tuple(modules):
        if not module_name.startswith(path_prefix):
            continue
        if module_name != package_name and not module_name.startswith(current_prefix):
            modules.pop(module_name, None)


def _load_module(path: Path, source: bytes):
    module_name = _module_name(path, source)
    cached_module = modules.get(module_name)
    if cached_module is not None:
        return cached_module

    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PluginLoadError(f"Could not create a module loader for {path}")

    module = module_from_spec(spec)
    modules[module_name] = module
    try:
        code = compile(source, str(path), "exec")
        exec(code, module.__dict__)
    except Exception:
        modules.pop(module_name, None)
        raise
    return module


def _remove_previous_versions(path: Path, current_module_name: str) -> None:
    path_prefix = _module_prefix(path)
    for module_name in tuple(modules):
        if module_name.startswith(path_prefix) and module_name != current_module_name:
            modules.pop(module_name, None)


def _plugin_implementations(
    module,
    *,
    require_plugin: bool = True,
) -> tuple[LLMToolsPlugin, ...]:
    implementations = []
    seen_classes: set[type[LLMToolsPlugin]] = set()
    for value in vars(module).values():
        if not isclass(value) or value is LLMToolsPlugin:
            continue
        if value.__module__ != module.__name__ or not issubclass(value, LLMToolsPlugin):
            continue
        if isabstract(value) or value in seen_classes:
            continue
        seen_classes.add(value)
        implementations.append(value())

    if require_plugin and not implementations:
        raise _NoPluginFound(f"No concrete LLMToolsPlugin subclasses found in {module.__file__}")
    return tuple(implementations)


class FileLoader:
    """Load one Python file and collect its concrete LLMToolsPlugin subclasses."""

    def load(self, path: Path | str) -> LLMPluginCollection:
        plugin_path = Path(path).expanduser().resolve(strict=True)
        if not plugin_path.is_file() or plugin_path.suffix.lower() != ".py":
            raise PluginLoadError(f"Plugin path must be a Python file: {plugin_path}")

        source = plugin_path.read_bytes()
        module = _load_module(plugin_path, source)
        vars(module).pop(_LEGACY_COLLECTION_ATTRIBUTE, None)
        try:
            collection = LLMPluginCollection.from_plugins(_plugin_implementations(module))
            _remove_previous_versions(plugin_path, module.__name__)
            return collection
        except Exception:
            modules.pop(module.__name__, None)
            raise


class DirectoryLoader:
    """Load direct Python modules as a package and collect their plugin classes."""

    def load(self, path: Path | str) -> LLMPluginCollection:
        plugin_dir = Path(path).expanduser().resolve(strict=True)
        if not plugin_dir.is_dir():
            raise PluginLoadError(f"Plugin path must be a directory: {plugin_dir}")

        plugin_files = sorted(
            (
                child
                for child in plugin_dir.iterdir()
                if child.is_file() and child.suffix.lower() == ".py" and child.name != "__init__.py"
            ),
            key=lambda child: child.name,
        )
        if not plugin_files:
            raise PluginLoadError(f"No Python plugin files found directly in {plugin_dir}")

        sources: dict[str, tuple[Path, bytes]] = {}
        for plugin_file in plugin_files:
            if not plugin_file.stem.isidentifier():
                raise PluginLoadError(
                    f"Plugin module name must be a valid Python identifier: {plugin_file.name}"
                )
            sources[plugin_file.stem] = (plugin_file, plugin_file.read_bytes())

        package_name, path_prefix = _directory_identity(plugin_dir, sources)
        package = _directory_package(package_name, plugin_dir)
        vars(package).pop(_LEGACY_COLLECTION_ATTRIBUTE, None)

        finder = _PluginSourceFinder(package_name, sources)
        try:
            meta_path.insert(0, finder)
            try:
                loaded_modules = tuple(
                    import_module(f"{package_name}.{module_name}") for module_name in sources
                )
            finally:
                meta_path.remove(finder)

            implementations = tuple(
                plugin
                for module in loaded_modules
                for plugin in _plugin_implementations(module, require_plugin=False)
            )
            if not implementations:
                raise PluginLoadError(
                    f"No concrete LLMToolsPlugin subclasses found directly in {plugin_dir}"
                )

            collection = LLMPluginCollection.from_plugins(implementations)
            _remove_previous_directory_versions(path_prefix, package_name)
            return collection
        except Exception:
            if finder in meta_path:
                meta_path.remove(finder)
            _remove_directory_package(package_name)
            raise
