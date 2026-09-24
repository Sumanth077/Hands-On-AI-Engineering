import socket
from contextlib import contextmanager

from util.network_status import is_online


@contextmanager
def _fake_socket(open_succeeds: bool):
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    def fake_create_connection(address, timeout=None):
        if open_succeeds:
            return FakeConnection()
        raise OSError("connection refused")

    yield fake_create_connection


def test_is_online_returns_true_when_connection_succeeds(monkeypatch):
    with _fake_socket(open_succeeds=True) as fake_create_connection:
        monkeypatch.setattr(socket, "create_connection", fake_create_connection)
        assert is_online(timeout=1.0) is True


def test_is_online_returns_false_when_connection_fails(monkeypatch):
    with _fake_socket(open_succeeds=False) as fake_create_connection:
        monkeypatch.setattr(socket, "create_connection", fake_create_connection)
        assert is_online(timeout=1.0) is False


def test_is_online_returns_false_on_timeout(monkeypatch):
    def fake_create_connection(address, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    assert is_online(timeout=1.0) is False
