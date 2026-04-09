from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import subprocess
import threading
from typing import Any, Mapping
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


READONLY_KEYWORDS = {
    "read",
    "get",
    "list",
    "search",
    "lookup",
    "query",
    "forecast",
    "route",
    "route",
    "travel",
    "geocode",
    "place",
    "location",
    "weather",
    "map",
}

WRITE_KEYWORDS = {
    "create",
    "write",
    "update",
    "delete",
    "remove",
    "set",
    "patch",
    "post",
    "put",
    "send",
    "commit",
    "sync",
}


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str
    url: str | None = None
    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class MCPToolSpec:
    server_name: str
    transport: str
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any]
    read_only: bool
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class MCPCallResult:
    ok: bool
    status: str
    source: str
    server_name: str
    transport: str
    tool_name: str | None
    capability: str
    request: dict[str, Any]
    response: dict[str, Any]
    error: str | None = None


class MCPProtocolError(RuntimeError):
    pass


def _clean_text(value: Any, max_len: int = 128) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:max_len]


def _normalize_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): value[key] for key in value}
    return {}


def _read_json_file(path: Path) -> list[dict[str, Any]]:
    try:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        servers = data.get("servers")
        if isinstance(servers, list):
            return [item for item in servers if isinstance(item, dict)]
    return []


def _maybe_parse_server_config(raw: dict[str, Any]) -> MCPServerConfig | None:
    name = _clean_text(raw.get("name"), 80)
    transport = _clean_text(raw.get("transport"), 16).lower() or "http"
    enabled = bool(raw.get("enabled", True))
    timeout_seconds = float(raw.get("timeout_seconds") or raw.get("timeoutSeconds") or 10.0)
    headers = _normalize_json_object(raw.get("headers"))
    env = {str(key): str(value) for key, value in _normalize_json_object(raw.get("env")).items()}

    if not name:
        return None
    if transport == "http":
        url = _clean_text(raw.get("url"), 512)
        if not url:
            return None
        return MCPServerConfig(
            name=name,
            transport=transport,
            url=url,
            headers={str(key): str(value) for key, value in headers.items()},
            enabled=enabled,
            timeout_seconds=timeout_seconds,
        )

    if transport == "stdio":
        command = raw.get("command")
        if not isinstance(command, list):
            command = []
        command_list = [str(item) for item in command if str(item).strip()]
        if not command_list:
            return None
        return MCPServerConfig(
            name=name,
            transport=transport,
            command=command_list,
            env=env,
            headers={str(key): str(value) for key, value in headers.items()},
            enabled=enabled,
            timeout_seconds=timeout_seconds,
        )

    return None


def load_mcp_server_configs(*, data_root: str | Path, env: Mapping[str, str] | None = None) -> list[MCPServerConfig]:
    environment = dict(os.environ if env is None else env)
    configs: dict[str, MCPServerConfig] = {}

    default_path = Path(data_root) / "mcp_servers.json"
    config_paths: list[Path] = [default_path]
    env_config = environment.get("PLANNER_MCP_CONFIG") or environment.get("MCP_CONFIG")
    if env_config:
        config_paths.insert(0, Path(env_config))

    inline_servers = environment.get("PLANNER_MCP_SERVERS") or environment.get("MCP_SERVERS")
    if inline_servers:
        try:
            parsed = json.loads(inline_servers)
        except Exception:
            parsed = []
        if isinstance(parsed, dict):
            parsed = parsed.get("servers", parsed)
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                config = _maybe_parse_server_config(item)
                if config:
                    configs[config.name] = config

    for path in config_paths:
        for raw in _read_json_file(path):
            config = _maybe_parse_server_config(raw)
            if config:
                configs.setdefault(config.name, config)

    return [configs[name] for name in sorted(configs)]


def _looks_read_only(name: str, description: str, annotations: dict[str, Any]) -> bool:
    lowered = f"{name} {description}".lower()
    if any(keyword in lowered for keyword in WRITE_KEYWORDS):
        return False
    if annotations.get("readOnlyHint") is False:
        return False
    if annotations.get("readOnlyHint") is True:
        return True
    return any(keyword in lowered for keyword in READONLY_KEYWORDS)


def _infer_capabilities(name: str, description: str, schema: dict[str, Any]) -> tuple[str, ...]:
    text = f"{name} {description}".lower()
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    prop_text = " ".join(str(key).lower() for key in properties)
    text = f"{text} {prop_text}"
    capabilities: list[str] = []

    if any(keyword in text for keyword in ["weather", "forecast", "temperature", "humidity", "precip", "rain", "wind"]):
        capabilities.append("weather")
    if any(keyword in text for keyword in ["route", "directions", "travel", "distance", "duration", "eta", "commute"]):
        capabilities.append("route")
    if any(keyword in text for keyword in ["geocode", "geo-code", "place", "location", "address", "lookup", "search"]):
        capabilities.append("place")
    if any(keyword in text for keyword in ["time zone", "timezone", "timezone", "region", "context"]):
        capabilities.append("context")
    if not capabilities:
        capabilities.append("generic")
    return tuple(dict.fromkeys(capabilities))


def _extract_tool_list(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        tools = result.get("tools")
        if isinstance(tools, list):
            return [tool for tool in tools if isinstance(tool, dict)]
        if isinstance(result.get("result"), dict):
            nested_tools = result["result"].get("tools")
            if isinstance(nested_tools, list):
                return [tool for tool in nested_tools if isinstance(tool, dict)]
    if isinstance(result, list):
        return [tool for tool in result if isinstance(tool, dict)]
    return []


def _normalize_tool_spec(server_name: str, transport: str, raw: dict[str, Any]) -> MCPToolSpec | None:
    name = _clean_text(raw.get("name"), 120)
    description = _clean_text(raw.get("description"), 500)
    annotations = _normalize_json_object(raw.get("annotations"))
    schema = _normalize_json_object(raw.get("inputSchema") or raw.get("input_schema"))
    if not name:
        return None
    read_only = _looks_read_only(name, description, annotations)
    if not read_only:
        return None
    return MCPToolSpec(
        server_name=server_name,
        transport=transport,
        name=name,
        description=description,
        input_schema=schema,
        annotations=annotations,
        read_only=read_only,
        capabilities=_infer_capabilities(name, description, schema),
    )


def _json_dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sanitize_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if not schema:
        return {key: value for key, value in arguments.items() if value is not None}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {key: value for key, value in arguments.items() if value is not None}
    sanitized: dict[str, Any] = {}
    for key in properties:
        if key in arguments and arguments[key] is not None:
            sanitized[key] = arguments[key]
    if not sanitized:
        sanitized = {key: value for key, value in arguments.items() if value is not None}
    return sanitized


class BaseMCPClient:
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config

    def initialize(self) -> dict[str, Any]:
        return self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "interactive-calendar-planner", "version": "0.1.0"},
                "capabilities": {},
            },
        ) or {}

    def list_tools(self) -> list[dict[str, Any]]:
        result = self.request("tools/list", {}) or {}
        return _extract_tool_list(result)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"items": result}
        return {"result": result}

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return


class HttpMCPClient(BaseMCPClient):
    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        assert self.config.url is not None
        payload = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            payload["params"] = params
        req = urlrequest.Request(
            self.config.url,
            data=_json_dumps(payload),
            headers={
                "Content-Type": "application/json",
                **{str(key): str(value) for key, value in self.config.headers.items()},
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise PermissionError(f"MCP server {self.config.name} denied access ({exc.code})") from exc
            raise MCPProtocolError(f"MCP HTTP request failed for {self.config.name}: {exc}") from exc
        except URLError as exc:
            raise MCPProtocolError(f"MCP HTTP transport unavailable for {self.config.name}: {exc}") from exc

        try:
            parsed = json.loads(body)
        except Exception as exc:
            raise MCPProtocolError(f"Invalid MCP HTTP response from {self.config.name}") from exc
        if isinstance(parsed, dict) and parsed.get("error"):
            error = parsed["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise MCPProtocolError(f"MCP HTTP error from {self.config.name}: {message}")
        return parsed.get("result") if isinstance(parsed, dict) else parsed


class StdioMCPClient(BaseMCPClient):
    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self._process: subprocess.Popen[bytes] | None = None
        self._request_id = 0
        self._lock = threading.Lock()

    def _ensure_process(self) -> subprocess.Popen[bytes]:
        if self._process and self._process.poll() is None:
            return self._process
        env = os.environ.copy()
        env.update(self.config.env)
        self._process = subprocess.Popen(  # noqa: S603,S607 - controlled by local config
            self.config.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        return self._process

    def _read_message(self, process: subprocess.Popen[bytes]) -> dict[str, Any]:
        assert process.stdout is not None
        header_bytes = bytearray()
        while b"\r\n\r\n" not in header_bytes:
            chunk = process.stdout.read(1)
            if not chunk:
                raise MCPProtocolError(f"MCP stdio server {self.config.name} closed stdout")
            header_bytes.extend(chunk)
        header_text, _, remainder = header_bytes.partition(b"\r\n\r\n")
        headers: dict[str, str] = {}
        for raw_line in header_text.decode("utf-8").split("\r\n"):
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        body = bytes(remainder)
        while len(body) < length:
            chunk = process.stdout.read(length - len(body))
            if not chunk:
                raise MCPProtocolError(f"MCP stdio server {self.config.name} returned truncated body")
            body += chunk
        parsed = json.loads(body[:length].decode("utf-8"))
        if isinstance(parsed, dict) and parsed.get("error"):
            error = parsed["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise MCPProtocolError(f"MCP stdio error from {self.config.name}: {message}")
        return parsed if isinstance(parsed, dict) else {"result": parsed}

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        with self._lock:
            process = self._ensure_process()
            assert process.stdin is not None
            self._request_id += 1
            payload = {"jsonrpc": "2.0", "id": self._request_id, "method": method}
            if params is not None:
                payload["params"] = params
            message = _json_dumps(payload)
            framed = f"Content-Length: {len(message)}\r\n\r\n".encode("utf-8") + message
            try:
                process.stdin.write(framed)
                process.stdin.flush()
            except Exception as exc:
                raise MCPProtocolError(f"MCP stdio write failed for {self.config.name}: {exc}") from exc

            parsed = self._read_message(process)
            return parsed.get("result") if isinstance(parsed, dict) else parsed

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except Exception:
                    process.kill()
        except Exception:
            pass
        finally:
            for handle_name in ("stdin", "stdout", "stderr"):
                handle = getattr(process, handle_name, None)
                if handle is None:
                    continue
                try:
                    handle.close()
                except Exception:
                    continue


def _build_client(config: MCPServerConfig) -> BaseMCPClient:
    if config.transport == "stdio":
        return StdioMCPClient(config)
    return HttpMCPClient(config)


class MCPRegistry:
    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self.configs = [config for config in configs if config.enabled]
        self.tools: list[MCPToolSpec] = []
        self.discovery_errors: list[dict[str, Any]] = []
        self._clients: dict[str, BaseMCPClient] = {}
        self._discovered = False

    @classmethod
    def from_environment(
        cls,
        *,
        data_root: str | Path,
        env: Mapping[str, str] | None = None,
    ) -> "MCPRegistry":
        return cls(load_mcp_server_configs(data_root=data_root, env=env))

    def _client_for(self, config: MCPServerConfig) -> BaseMCPClient:
        client = self._clients.get(config.name)
        if client is None:
            client = _build_client(config)
            self._clients[config.name] = client
        return client

    def discover(self) -> dict[str, Any]:
        if self._discovered:
            return self.discovery_summary()

        discovered: list[MCPToolSpec] = []
        for config in self.configs:
            client = self._client_for(config)
            server_summary = {
                "name": config.name,
                "transport": config.transport,
                "status": "unavailable",
                "tool_count": 0,
                "read_only_tool_count": 0,
            }
            try:
                client.initialize()
                tools = client.list_tools()
            except PermissionError as exc:
                server_summary["status"] = "permission_denied"
                server_summary["error"] = str(exc)
                self.discovery_errors.append(server_summary)
                continue
            except Exception as exc:
                server_summary["status"] = "unavailable"
                server_summary["error"] = str(exc)
                self.discovery_errors.append(server_summary)
                continue

            server_tools: list[MCPToolSpec] = []
            for raw_tool in tools:
                spec = _normalize_tool_spec(config.name, config.transport, raw_tool)
                if spec:
                    server_tools.append(spec)

            server_summary["status"] = "available"
            server_summary["tool_count"] = len(tools)
            server_summary["read_only_tool_count"] = len(server_tools)
            self.discovery_errors.append(server_summary)
            discovered.extend(server_tools)

        self.tools = discovered
        self._discovered = True
        return self.discovery_summary()

    def discovery_summary(self) -> dict[str, Any]:
        return {
            "servers": list(self.discovery_errors),
            "tools": [
                {
                    "server_name": tool.server_name,
                    "transport": tool.transport,
                    "name": tool.name,
                    "description": tool.description,
                    "capabilities": list(tool.capabilities),
                    "read_only": tool.read_only,
                }
                for tool in self.tools
            ],
            "available_capabilities": sorted({cap for tool in self.tools for cap in tool.capabilities}),
        }

    def _select_tool(self, capability: str) -> MCPToolSpec | None:
        if not self._discovered:
            self.discover()
        candidates = [tool for tool in self.tools if capability in tool.capabilities]
        if not candidates:
            candidates = [tool for tool in self.tools if "generic" in tool.capabilities]
        return candidates[0] if candidates else None

    def call(
        self,
        capability: str,
        *,
        arguments: dict[str, Any],
        purpose: str,
    ) -> MCPCallResult:
        spec = self._select_tool(capability)
        if spec is None:
            return MCPCallResult(
                ok=False,
                status="unavailable",
                source="mcp",
                server_name="",
                transport="",
                tool_name=None,
                capability=capability,
                request={"purpose": purpose, "arguments": arguments},
                response={},
                error="no matching MCP tool available",
            )

        client = self._client_for(next(config for config in self.configs if config.name == spec.server_name))
        sanitized = _sanitize_arguments(arguments, spec.input_schema)
        request_payload = {"purpose": purpose, "arguments": sanitized}
        try:
            response = client.call_tool(spec.name, sanitized)
        except PermissionError as exc:
            return MCPCallResult(
                ok=False,
                status="permission_denied",
                source="mcp",
                server_name=spec.server_name,
                transport=spec.transport,
                tool_name=spec.name,
                capability=capability,
                request=request_payload,
                response={},
                error=str(exc),
            )
        except Exception as exc:
            return MCPCallResult(
                ok=False,
                status="unavailable",
                source="mcp",
                server_name=spec.server_name,
                transport=spec.transport,
                tool_name=spec.name,
                capability=capability,
                request=request_payload,
                response={},
                error=str(exc),
            )

        if not isinstance(response, dict):
            response = {"value": response}

        return MCPCallResult(
            ok=True,
            status="ok",
            source="mcp",
            server_name=spec.server_name,
            transport=spec.transport,
            tool_name=spec.name,
            capability=capability,
            request=request_payload,
            response=response,
        )

    def close(self) -> None:
        for client in self._clients.values():
            try:
                client.close()
            except Exception:
                continue


def build_mcp_registry(*, data_root: str | Path, env: Mapping[str, str] | None = None) -> MCPRegistry:
    registry = MCPRegistry.from_environment(data_root=data_root, env=env)
    registry.discover()
    return registry
