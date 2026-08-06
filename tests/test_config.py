"""Configuration loading, validation and precedence."""

from __future__ import annotations

import pytest

from inference_service.config import (
    BatchingConfig,
    Config,
    ConfigError,
    ServerConfig,
    find_config_file,
    load_config,
)


def write(tmp_path, content: str, name: str = "service.toml") -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestDefaults:
    def test_defaults_are_valid(self) -> None:
        Config().validate()

    def test_production_safe_defaults(self) -> None:
        config = Config()
        # Auth off is a deliberate default for a service behind a gateway, but
        # everything that could silently degrade production is safe.
        assert config.batching.max_queue_size > 0, "queue must be bounded"
        assert config.batching.request_timeout_s > 0, "requests must have a budget"
        assert config.logging.log_request_body is False, "user text must not be logged"
        assert config.server.keep_alive_timeout >= 65, "must exceed typical LB idle timeout"

    def test_missing_file_uses_defaults(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        config = load_config(None, env={})
        assert config.source_path is None
        assert config.batching.max_batch_size == 32


class TestFileLoading:
    def test_values_are_read(self, tmp_path) -> None:
        path = write(
            tmp_path,
            """
            service_name = "custom"
            environment = "staging"

            [batching]
            max_batch_size = 64
            max_wait_ms = 25.0

            [api]
            api_keys = ["a", "b"]
            """,
        )
        config = load_config(path, env={})
        assert config.service_name == "custom"
        assert config.environment == "staging"
        assert config.batching.max_batch_size == 64
        assert config.batching.max_wait_ms == 25.0
        assert config.api.api_keys == ("a", "b")
        assert config.api.auth_enabled is True

    def test_shipped_config_is_valid(self) -> None:
        """The config we ship must actually load."""
        from tests.conftest import REPO_ROOT

        config = load_config(str(REPO_ROOT / "config" / "service.toml"), env={})
        assert config.service_name == "inference-service"

    def test_explicit_missing_path_is_fatal(self, tmp_path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(str(tmp_path / "nope.toml"), env={})

    def test_invalid_toml_names_the_file(self, tmp_path) -> None:
        path = write(tmp_path, "[batching\nmax = ")
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_config(path, env={})


class TestTypoRejection:
    """Unknown keys are fatal, deliberately.

    A silently-ignored `max_batch_sze` means an operator believes they tuned
    throughput when they did not, and the misconfiguration is invisible until
    someone reads the source.
    """

    def test_unknown_key_in_section(self, tmp_path) -> None:
        path = write(tmp_path, "[batching]\nmax_batch_sze = 64\n")
        with pytest.raises(ConfigError) as exc:
            load_config(path, env={})
        assert "max_batch_sze" in str(exc.value)
        assert "allowed" in str(exc.value)

    def test_unknown_top_level_section(self, tmp_path) -> None:
        path = write(tmp_path, "[batchng]\nmax_batch_size = 4\n")
        with pytest.raises(ConfigError, match="batchng"):
            load_config(path, env={})

    def test_wrong_type(self, tmp_path) -> None:
        path = write(tmp_path, '[server]\nport = "8000"\n')
        with pytest.raises(ConfigError, match="port must be an integer"):
            load_config(path, env={})

    def test_bare_string_where_array_expected(self, tmp_path) -> None:
        path = write(tmp_path, '[api]\napi_keys = "single-key"\n')
        with pytest.raises(ConfigError, match="array of strings"):
            load_config(path, env={})


class TestValidation:
    @pytest.mark.parametrize("port", [0, -1, 70000])
    def test_port_range(self, port: int) -> None:
        with pytest.raises(ConfigError, match="1-65535"):
            ServerConfig(port=port).validate()

    def test_prestop_must_leave_time_to_drain(self) -> None:
        """A prestop delay longer than the shutdown budget means the queue never
        drains -- the process is killed still holding accepted work."""
        with pytest.raises(ConfigError, match="below"):
            ServerConfig(prestop_delay_s=40.0, graceful_shutdown_timeout=30.0).validate()

    def test_batch_window_must_be_below_request_timeout(self) -> None:
        """Otherwise every request times out while waiting to be batched."""
        with pytest.raises(ConfigError, match="must be well below"):
            BatchingConfig(max_wait_ms=20000.0, request_timeout_s=10.0).validate()

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"max_batch_size": 0}, "max_batch_size"),
            ({"max_wait_ms": -1}, "max_wait_ms"),
            ({"max_queue_size": 0}, "max_queue_size"),
            ({"request_timeout_s": 0}, "request_timeout_s"),
        ],
    )
    def test_batching_bounds(self, kwargs, match) -> None:
        with pytest.raises(ConfigError, match=match):
            BatchingConfig(**kwargs).validate()

    def test_unsorted_latency_buckets_rejected(self, tmp_path) -> None:
        """Prometheus histogram buckets must be ascending or quantiles are wrong."""
        path = write(tmp_path, "[observability]\nlatency_buckets = [1.0, 0.5, 2.0]\n")
        with pytest.raises(ConfigError, match="ascending"):
            load_config(path, env={})

    def test_cors_wildcard_cannot_be_combined(self, tmp_path) -> None:
        path = write(tmp_path, '[api]\ncors_origins = ["*", "https://x.test"]\n')
        with pytest.raises(ConfigError, match="cannot combine"):
            load_config(path, env={})

    def test_unknown_backend_rejected(self, tmp_path) -> None:
        path = write(tmp_path, '[model]\nbackend = "pytorch"\n')
        with pytest.raises(ConfigError, match=r"model\.backend"):
            load_config(path, env={})


class TestEnvironmentOverrides:
    def test_env_beats_file(self, tmp_path) -> None:
        path = write(tmp_path, "[batching]\nmax_batch_size = 8\n")
        config = load_config(path, env={"INFERENCE_MAX_BATCH_SIZE": "64"})
        assert config.batching.max_batch_size == 64

    def test_comma_separated_lists(self, tmp_path) -> None:
        path = write(tmp_path, "")
        config = load_config(path, env={"INFERENCE_API_KEYS": "a, b ,c"})
        assert config.api.api_keys == ("a", "b", "c")

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_boolean_forms(self, tmp_path, value, expected) -> None:
        path = write(tmp_path, "")
        config = load_config(path, env={"INFERENCE_BATCHING_ENABLED": value})
        assert config.batching.enabled is expected

    def test_invalid_boolean_rejected(self, tmp_path) -> None:
        path = write(tmp_path, "")
        with pytest.raises(ConfigError, match="boolean-like"):
            load_config(path, env={"INFERENCE_BATCHING_ENABLED": "perhaps"})

    def test_invalid_int_rejected(self, tmp_path) -> None:
        path = write(tmp_path, "")
        with pytest.raises(ConfigError, match="must be an integer"):
            load_config(path, env={"INFERENCE_PORT": "eighty"})

    def test_config_path_from_env(self, tmp_path) -> None:
        path = write(tmp_path, 'service_name = "from-env-path"\n')
        config = load_config(None, env={"INFERENCE_CONFIG": path})
        assert config.service_name == "from-env-path"


class TestDiscovery:
    def test_returns_none_when_nothing_exists(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert find_config_file(None) is None

    def test_finds_local_config_directory(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "service.toml").write_text("", encoding="utf-8")
        assert find_config_file(None) == "./config/service.toml"
