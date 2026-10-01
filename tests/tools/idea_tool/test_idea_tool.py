from pathlib import Path

from pydantic import TypeAdapter

from pytoy_llm.idea import IdeaSpace, SchemeDirectory, UriLocalPathResolver
from pytoy_llm.idea.domain.uri import Uri
from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.idea_tool import IdeaTool
from pytoy_llm.tools.idea_tool.models import IdeaNoteModel, RemoteLinkModel
from pytoy_llm.tools.idea_tool.semantic_types import IdeaNoteReference, Namespace


def test_remote_link_model_serializes_uri_dataclass() -> None:
    uri = Uri.from_any("https://example.com/reference?q=python#install")
    model = RemoteLinkModel(
        source_idea_note_reference=IdeaNoteReference(
            idea_note_path="knowledge/python.md", namespace="research"
        ),
        uri=uri,
    )

    assert model.model_dump()["uri"] == {
        "scheme": "https",
        "authority": "example.com",
        "path": "/reference",
        "query": "q=python",
        "fragment": "install",
    }
    assert model.model_dump()["source_idea_note_reference"] == {
        "namespace": "research",
        "idea_note_path": "knowledge/python.md",
    }
    assert RemoteLinkModel.model_validate(model.model_dump()).uri == uri


def test_optional_namespace_schema_uses_null_for_default() -> None:
    schema = TypeAdapter(Namespace | None).json_schema()
    namespace_schema = next(item for item in schema["anyOf"] if item.get("type") == "string")

    assert "null" in namespace_schema["description"]
    assert "" not in namespace_schema["examples"]


def test_idea_note_operations_use_selected_namespace(tmp_path: Path) -> None:
    drafts_root = tmp_path / "drafts"
    archive_root = tmp_path / "archive"
    drafts_root.mkdir()
    archive_root.mkdir()
    (drafts_root / "shared.md").write_text("draft note\n", encoding="utf-8")
    (archive_root / "shared.md").write_text("archived note\n", encoding="utf-8")
    tool = IdeaTool.from_any(
        [
            drafts_root,
            archive_root,
        ],
        default_namespace="drafts",
    )

    default_note = tool.inspection.get_idea_note("shared.md")
    archive_note = tool.inspection.get_idea_note("shared.md", idea_namespace="archive")
    written_path = tool.mutation.write_idea_note(
        "new.md", "new archived note", {}, idea_namespace="archive"
    )

    assert isinstance(default_note, IdeaNoteModel)
    assert default_note.body == "draft note\n"
    assert isinstance(archive_note, IdeaNoteModel)
    assert archive_note.body == "archived note\n"
    assert written_path == "new.md"
    assert (archive_root / "new.md").read_text(encoding="utf-8") == "new archived note"
    assert not (drafts_root / "new.md").exists()


def test_local_link_path_is_decoded_and_uri_string_round_trips(tmp_path: Path) -> None:
    idea_root = tmp_path / "ideas"
    workspace_root = tmp_path / "workspace"
    idea_root.mkdir()
    workspace_root.mkdir()
    (idea_root / "source.md").write_text(
        "[target](workspace:///src/a%20b%23c.md)", encoding="utf-8"
    )
    tool = IdeaTool.from_any(idea_root, workspace_root=workspace_root)

    note = tool.inspection.get_idea_note("source.md")

    assert isinstance(note, IdeaNoteModel)
    assert len(note.local_links) == 1
    local_link = note.local_links[0]
    assert local_link.path == "src/a b#c.md"

    resolver = UriLocalPathResolver([SchemeDirectory(workspace_root, "workspace")])
    round_trip_uri = Uri.from_any(local_link.uri_string)
    assert resolver.resolve(round_trip_uri) == workspace_root / "src" / "a b#c.md"


def test_discovery_tools_are_registered_with_documentation(tmp_path: Path) -> None:
    tool = IdeaTool.from_any(IdeaSpace(tmp_path))
    registered_tools = {
        getattr(registered_tool, "__name__", ""): registered_tool for registered_tool in tool.tools
    }

    for name in (
        "get_idea_space_paths_with_conventions",
        "get_sub_idea_spaces",
        "get_idea_note_paths",
    ):
        assert name in registered_tools
        assert registered_tools[name].__doc__


def test_inspection_tools_are_registered_with_documentation(tmp_path: Path) -> None:
    tool = IdeaTool.from_any(IdeaSpace(tmp_path))
    registered_tools = {
        getattr(registered_tool, "__name__", ""): registered_tool for registered_tool in tool.tools
    }

    for name in (
        "get_idea_space_working_context",
        "get_idea_space_convention",
        "get_effective_conventions",
        "get_updated_time_of_idea_notes",
        "get_metadata_of_idea_notes",
        "get_idea_note",
    ):
        assert name in registered_tools
        assert registered_tools[name].__doc__


def test_mutation_tools_are_registered_with_documentation(tmp_path: Path) -> None:
    tool = IdeaTool.from_any(IdeaSpace(tmp_path))
    registered_tools = {
        getattr(registered_tool, "__name__", ""): registered_tool for registered_tool in tool.tools
    }

    for name in (
        "create_sub_idea_space",
        "delete_sub_idea_space",
        "update_metadata_of_idea_note",
        "write_idea_note",
        "delete_idea_note",
    ):
        assert name in registered_tools
        assert registered_tools[name].__doc__


def test_get_metadata_of_idea_notes_returns_metadata_and_none_for_invalid_paths(
    tmp_path: Path,
) -> None:
    note_path = tmp_path / "note.md"
    note_path.write_text("---\nstatus: active\ntags: [one, two]\n---\nBody\n")
    (tmp_path / "folder").mkdir()

    tool = IdeaTool.from_any(tmp_path)

    result = tool.inspection.get_metadata_of_idea_notes(["note.md", "missing.md", "folder"])

    assert result == {
        "note.md": {"status": "active", "tags": ["one", "two"]},
        "missing.md": None,
        "folder": None,
    }


def test_create_sub_idea_space_creates_a_new_directory(tmp_path: Path) -> None:
    tool = IdeaTool.from_any(tmp_path)
    result = tool.mutation.create_sub_idea_space("knowledge")

    assert result == "knowledge"
    assert (tmp_path / "knowledge").is_dir()


def test_create_sub_idea_space_rejects_existing_directory(tmp_path: Path) -> None:
    (tmp_path / "knowledge").mkdir()
    tool = IdeaTool.from_any(tmp_path)

    result = tool.mutation.create_sub_idea_space("knowledge")

    assert isinstance(result, ToolError)
    assert result.kind == ToolErrorKind.INVALID_ARGUMENT


def test_delete_subspace_deletes_only_empty_directories(tmp_path: Path) -> None:
    (tmp_path / "knowledge").mkdir()
    tool = IdeaTool.from_any(tmp_path)

    result = tool.mutation.delete_sub_idea_space("knowledge")

    assert result == "knowledge"
    assert not (tmp_path / "knowledge").exists()


def test_delete_subspace_rejects_non_empty_directories(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "note.md").write_text("note", encoding="utf-8")
    tool = IdeaTool.from_any(tmp_path)

    result = tool.mutation.delete_sub_idea_space("knowledge")

    assert isinstance(result, ToolError)
    assert result.kind == ToolErrorKind.INVALID_ARGUMENT
    assert knowledge.exists()


def test_mutation_tools_cannot_access_reserved_space_metadata(tmp_path: Path) -> None:
    tool = IdeaTool.from_any(tmp_path)
    tool.mark_llm_finished()
    meta_name = IdeaSpace.SPACE_META_NAME
    context_path = tmp_path / meta_name / "tool_context.json"
    original_context = context_path.read_text(encoding="utf-8")

    results = [
        tool.mutation.create_sub_idea_space(f"{meta_name}/new-space"),
        tool.mutation.delete_sub_idea_space(meta_name),
        tool.mutation.write_idea_note(f"{meta_name}/tool_context.json", "overwritten", {}),
        tool.mutation.update_metadata_of_idea_note(
            f"{meta_name}/tool_context.json", {"status": "overwritten"}
        ),
        tool.mutation.delete_idea_note(f"{meta_name}/tool_context.json"),
    ]

    assert all(
        isinstance(result, ToolError) and result.kind == ToolErrorKind.PERMISSION_DENIED
        for result in results
    )
    assert context_path.read_text(encoding="utf-8") == original_context
    assert not (context_path.parent / "new-space").exists()


def test_idea_space_paths_with_conventions_returns_root_and_nested_spaces(
    tmp_path: Path,
) -> None:
    (tmp_path / ".convention.md").write_text("root convention\n")
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / ".idea_space_convention.md").write_text("knowledge convention\n")
    (tmp_path / "knowledge" / "python").mkdir()
    (tmp_path / "knowledge" / "python" / ".convention.md").write_text("python convention\n")

    tool = IdeaTool.from_any(tmp_path)

    result = tool.discovery.get_idea_space_paths_with_conventions()

    assert result == [".", "knowledge", "knowledge/python"]


def test_idea_space_paths_with_conventions_represents_root_as_dot(tmp_path: Path) -> None:
    (tmp_path / ".convention.md").write_text("root convention\n")
    tool = IdeaTool.from_any(tmp_path)

    result = tool.discovery.get_idea_space_paths_with_conventions()

    assert result == ["."]


def test_idea_space_paths_with_conventions_returns_empty_without_conventions(
    tmp_path: Path,
) -> None:
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "note.md").write_text("note\n")
    tool = IdeaTool.from_any(tmp_path)

    result = tool.discovery.get_idea_space_paths_with_conventions()

    assert result == []


def test_get_effective_conventions_returns_ancestor_conventions_root_first(
    tmp_path: Path,
) -> None:
    (tmp_path / ".convention.md").write_text("root convention\n", encoding="utf-8")
    knowledge_path = tmp_path / "knowledge"
    knowledge_path.mkdir()
    (knowledge_path / ".idea_space_convention.md").write_text(
        "knowledge convention\n", encoding="utf-8"
    )
    python_path = knowledge_path / "python"
    python_path.mkdir()
    (python_path / ".convention.md").write_text("python convention\n", encoding="utf-8")
    note_path = python_path / "note.md"
    note_path.write_text("note\n", encoding="utf-8")
    tool = IdeaTool.from_any(tmp_path)

    result = tool.inspection.get_effective_conventions("knowledge/python/note.md")

    assert not isinstance(result, ToolError)
    assert list(result) == [".", "knowledge", "knowledge/python"]
    assert [model.idea_note.body for model in result.values()] == [
        "root convention\n",
        "knowledge convention\n",
        "python convention\n",
    ]


def test_get_effective_conventions_for_space_excludes_descendant_conventions(
    tmp_path: Path,
) -> None:
    (tmp_path / ".convention.md").write_text("root convention\n", encoding="utf-8")
    knowledge_path = tmp_path / "knowledge"
    knowledge_path.mkdir()
    (knowledge_path / ".convention.md").write_text("knowledge convention\n", encoding="utf-8")
    (knowledge_path / "python").mkdir()
    (knowledge_path / "python" / ".convention.md").write_text(
        "python convention\n", encoding="utf-8"
    )
    tool = IdeaTool.from_any(tmp_path)

    result = tool.inspection.get_effective_conventions("knowledge")

    assert list(result) == [".", "knowledge"]


def test_get_effective_conventions_returns_empty_mapping_without_conventions(
    tmp_path: Path,
) -> None:
    (tmp_path / "knowledge").mkdir()
    tool = IdeaTool.from_any(tmp_path)

    result = tool.inspection.get_effective_conventions("knowledge")

    assert result == {}


def test_get_effective_conventions_rejects_missing_paths_and_md_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "folder.md").mkdir()
    tool = IdeaTool.from_any(tmp_path)

    missing_note = tool.inspection.get_effective_conventions("missing.md")
    missing_space = tool.inspection.get_effective_conventions("missing-space")
    md_directory = tool.inspection.get_effective_conventions("folder.md")

    assert isinstance(missing_note, ToolError)
    assert missing_note.kind == ToolErrorKind.NOT_FOUND
    assert isinstance(missing_space, ToolError)
    assert missing_space.kind == ToolErrorKind.NOT_FOUND
    assert isinstance(md_directory, ToolError)
    assert md_directory.kind == ToolErrorKind.INVALID_ARGUMENT


def test_update_metadata_of_idea_note_preserves_body_and_merges_metadata(
    tmp_path: Path,
) -> None:
    note_path = tmp_path / "note.md"
    note_path.write_text("---\nstatus: draft\nowner: alice\n---\n# Body\n")

    tool = IdeaTool.from_any(tmp_path)

    result = tool.mutation.update_metadata_of_idea_note(
        "note.md", {"status": "published", "tags": ["one"]}
    )

    assert result == "note.md"
    assert (
        note_path.read_text() == "---\nstatus: published\nowner: alice\ntags:\n- one\n---\n# Body\n"
    )


def test_update_metadata_of_idea_note_returns_error_for_missing_note(tmp_path: Path) -> None:
    tool = IdeaTool.from_any(tmp_path)

    result = tool.mutation.update_metadata_of_idea_note("missing.md", {"status": "published"})

    assert isinstance(result, ToolError)


def test_update_metadata_of_idea_note_clear_replaces_existing_metadata(tmp_path: Path) -> None:
    note_path = tmp_path / "note.md"
    note_path.write_text("---\nstatus: draft\nowner: alice\n---\n# Body\n")

    tool = IdeaTool.from_any(tmp_path)

    result = tool.mutation.update_metadata_of_idea_note(
        "note.md", {"status": "published"}, clear=True
    )

    assert result == "note.md"
    assert note_path.read_text() == "---\nstatus: published\n---\n# Body\n"


def test_update_metadata_of_idea_note_clear_with_empty_metadata_removes_frontmatter(
    tmp_path: Path,
) -> None:
    note_path = tmp_path / "note.md"
    note_path.write_text("---\nstatus: draft\n---\n# Body\n")

    tool = IdeaTool.from_any(tmp_path)

    result = tool.mutation.update_metadata_of_idea_note("note.md", {}, clear=True)

    assert result == "note.md"
    assert note_path.read_text() == "# Body\n"
