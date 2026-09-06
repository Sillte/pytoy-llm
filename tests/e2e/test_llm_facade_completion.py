import os
from pathlib import Path

import pytest

from pytoy_llm.connection_configuration import DEFAULT_NAME, ConnectionConfiguration
from pytoy_llm.llm_facade import LLMFacade


@pytest.mark.skipif(
    os.getenv("PYTOY_LLM_RUN_E2E") != "1",
    reason="Set PYTOY_LLM_RUN_E2E=1 to run the provider-backed E2E test.",
)
def test_llm_facade_completion_returns_text():
    connection_name = os.getenv("PYTOY_LLM_CONNECTION", DEFAULT_NAME)
    connection_path = ConnectionConfiguration().get_connection_path(connection_name)
    if not Path(connection_path).exists():
        pytest.skip(f"Connection file does not exist: {connection_path}")

    result = LLMFacade(connection=connection_name).completion(
        "Reply with a short greeting.",
        output_type=str,
    )

    assert isinstance(result, str)
    assert result.strip()
