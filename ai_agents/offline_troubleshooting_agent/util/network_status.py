import socket

NETWORK_CHECK_HOST = "1.1.1.1"
NETWORK_CHECK_PORT = 53


def is_online(timeout: float = 1.0) -> bool:
    """Best-effort check for internet connectivity.

    Attempts a short-timeout TCP connection to a public address. Returns
    True if it succeeds (online), False if it fails or times out (offline).
    """
    try:
        with socket.create_connection((NETWORK_CHECK_HOST, NETWORK_CHECK_PORT), timeout=timeout):
            return True
    except OSError:
        return False
