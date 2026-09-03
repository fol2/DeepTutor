from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from deeptutor.services.llm.provider_core.cursor_sdk_provider import CursorSDKProvider


class FakeLocalAgentOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeAgentOptions:
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)


@dataclass(frozen=True)
class FakeModelParameterValue:
    id: str
    value: str


@dataclass(frozen=True)
class FakeModelSelection:
    id: str
    params: tuple[FakeModelParameterValue, ...] | list[FakeModelParameterValue] = ()


class FakeRun:
    def __init__(self, *, result: str = "hello", chunks: tuple[str, ...] = ("hel", "lo")):
        self.result = result
        self.chunks = chunks
        self.status = "running"
        self.cancelled = False
        self.usage = SimpleNamespace(input_tokens=7, output_tokens=2, total_tokens=9)

    async def wait(self):
        self.status = "finished"
        return SimpleNamespace(status="finished", result=self.result, usage=self.usage)

    async def iter_text(self):
        for chunk in self.chunks:
            yield chunk
        self.status = "finished"

    async def cancel(self):
        self.cancelled = True
        self.status = "cancelled"


class FakeAgent:
    def __init__(self, captured: dict[str, object], run: FakeRun):
        self.captured = captured
        self.run = run

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def send(self, prompt: str):
        self.captured["prompt"] = prompt
        workspace = Path(str(self.captured["workspace"]))
        self.captured["rule"] = (workspace / ".cursor" / "rules" / "deeptutor.mdc").read_text(
            encoding="utf-8"
        )
        return self.run


class FakeAgents:
    def __init__(self, captured: dict[str, object], run: FakeRun):
        self.captured = captured
        self.run = run

    async def create(self, options):
        self.captured["create"] = options
        return FakeAgent(self.captured, self.run)


class FakeClient:
    captured: dict[str, object]
    model_resource: type
    run: FakeRun

    def __init__(self):
        self.agents = FakeAgents(self.captured, self.run)
        self.models = self.model_resource(self.captured)

    @classmethod
    async def launch_bridge(cls, *, workspace: str, state_root: str):
        cls.captured["workspace"] = workspace
        cls.captured["state_root"] = state_root
        return cls()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


def install_fake_sdk(
    monkeypatch,
    *,
    run: FakeRun | None = None,
    models: list[object] | None = None,
) -> tuple[dict[str, object], FakeRun]:
    captured: dict[str, object] = {}
    fake_run = run or FakeRun()
    FakeClient.captured = captured
    FakeClient.run = fake_run

    default_models = [
        SimpleNamespace(
            id="grok-4.6",
            display_name="Grok 4.6",
            variants=[
                SimpleNamespace(params=(FakeModelParameterValue(id="effort", value="high"),))
            ],
            parameters=(),
        )
    ]

    class FakeModels:
        def __init__(self, target: dict[str, object]):
            self.target = target

        async def list(self, *, api_key):
            self.target["models_api_key"] = api_key
            return default_models if models is None else models

    FakeClient.model_resource = FakeModels

    fake_sdk = SimpleNamespace(
        AsyncClient=FakeClient,
        AgentOptions=FakeAgentOptions,
        LocalAgentOptions=FakeLocalAgentOptions,
        ModelParameterValue=FakeModelParameterValue,
        ModelSelection=FakeModelSelection,
    )
    monkeypatch.setitem(sys.modules, "cursor_sdk", fake_sdk)
    return captured, fake_run


@pytest.mark.asyncio
async def test_cursor_sdk_provider_is_text_only_and_uses_subscription_key(monkeypatch) -> None:
    captured, _run = install_fake_sdk(monkeypatch)
    provider = CursorSDKProvider(api_key="cursor-secret")

    response = await provider.chat(
        [
            {"role": "system", "content": "Use UK English."},
            {"role": "user", "content": "Say hello."},
        ],
        tools=[{"type": "function", "function": {"name": "dangerous"}}],
    )

    assert response.content == "hello"
    assert response.finish_reason == "stop"
    assert response.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }
    create = captured["create"]
    assert create.model.id == "grok-4.6"
    assert [(item.id, item.value) for item in create.model.params] == [("effort", "high")]
    assert create.api_key == "cursor-secret"
    assert create.mode == "agent"
    assert create.tools == []
    assert create.mcp_servers == {}
    assert create.agents == {}
    assert create.local.kwargs["setting_sources"] == ["project"]
    assert captured["models_api_key"] == "cursor-secret"
    assert Path(str(captured["state_root"])).parent == Path(str(captured["workspace"]))
    assert not Path(str(captured["workspace"])).exists()

    rule = str(captured["rule"])
    prompt = str(captured["prompt"])
    assert "Use UK English." in rule
    assert "Use UK English." not in prompt
    assert '"role":"user","content":"Say hello."' in prompt


@pytest.mark.asyncio
async def test_cursor_sdk_provider_streams_text_deltas(monkeypatch) -> None:
    install_fake_sdk(monkeypatch)
    deltas: list[str] = []

    async def append(text: str) -> None:
        deltas.append(text)

    response = await CursorSDKProvider(api_key="key").chat_stream(
        [{"role": "user", "content": "hello"}],
        on_content_delta=append,
    )

    assert response.content == "hello"
    assert deltas == ["hello"]


@pytest.mark.asyncio
async def test_cursor_sdk_provider_hard_limits_buffered_and_streamed_output(monkeypatch) -> None:
    run = FakeRun(result="one two three", chunks=("one ", "two ", "three"))
    install_fake_sdk(monkeypatch, run=run)
    deltas: list[str] = []

    async def append(text: str) -> None:
        deltas.append(text)

    response = await CursorSDKProvider(api_key="key").chat_stream(
        [{"role": "user", "content": "count"}],
        max_tokens=1,
        on_content_delta=append,
    )

    assert response.finish_reason == "length"
    assert response.content != "one two three"
    assert deltas == [response.content]


@pytest.mark.asyncio
async def test_cursor_sdk_provider_cancels_timed_out_run(monkeypatch) -> None:
    class SlowRun(FakeRun):
        async def wait(self):
            await asyncio.Event().wait()

    run = SlowRun()
    install_fake_sdk(monkeypatch, run=run)

    response = await CursorSDKProvider(api_key="key", timeout_seconds=0.01).chat(
        [{"role": "user", "content": "hello"}]
    )

    assert response.finish_reason == "error"
    assert response.content == "Cursor SDK request timed out"
    assert run.cancelled is True


@pytest.mark.asyncio
async def test_cursor_sdk_provider_propagates_cancellation_after_stopping_run(monkeypatch) -> None:
    class SlowRun(FakeRun):
        async def wait(self):
            await asyncio.Event().wait()

    run = SlowRun()
    install_fake_sdk(monkeypatch, run=run)
    task = asyncio.create_task(
        CursorSDKProvider(api_key="key").chat([{"role": "user", "content": "hello"}])
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert run.cancelled is True


@pytest.mark.asyncio
async def test_cursor_sdk_provider_redacts_provider_errors(monkeypatch) -> None:
    class FailingRun(FakeRun):
        async def wait(self):
            raise RuntimeError("request failed with cursor-secret")

    install_fake_sdk(monkeypatch, run=FailingRun())

    response = await CursorSDKProvider(api_key="cursor-secret").chat(
        [{"role": "user", "content": "hello"}]
    )

    assert response.finish_reason == "error"
    assert response.content == "Cursor SDK request failed (RuntimeError)"
    assert "cursor-secret" not in response.content


@pytest.mark.asyncio
async def test_cursor_sdk_provider_rejects_non_text_messages(monkeypatch) -> None:
    captured, _run = install_fake_sdk(monkeypatch)

    response = await CursorSDKProvider(api_key="key").chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            }
        ]
    )

    assert response.finish_reason == "error"
    assert response.content == "Cursor subscription provider accepts text-only messages"
    assert "workspace" not in captured


@pytest.mark.asyncio
async def test_cursor_sdk_provider_reports_unavailable_model(monkeypatch) -> None:
    install_fake_sdk(
        monkeypatch,
        models=[SimpleNamespace(id="claude-4", display_name="Claude 4")],
    )

    response = await CursorSDKProvider(api_key="key").chat([{"role": "user", "content": "hello"}])

    assert response.finish_reason == "error"
    assert response.content == ("Cursor Grok 4.6 High is not available for this Cursor API key")


@pytest.mark.asyncio
async def test_cursor_sdk_provider_reports_missing_optional_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cursor_sdk", None)

    response = await CursorSDKProvider(api_key="key").chat([{"role": "user", "content": "hello"}])

    assert response.finish_reason == "error"
    assert response.content == "Cursor SDK is not installed; install the cursor-sdk package"
