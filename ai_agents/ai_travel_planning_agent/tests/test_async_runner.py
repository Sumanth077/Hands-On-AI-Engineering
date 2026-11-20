import unittest
from unittest.mock import Mock, patch

from async_runner import run_coroutine


class RunCoroutineTests(unittest.TestCase):
    def test_uses_available_loop(self):
        coroutine = object()
        loop = Mock()
        loop.run_until_complete.return_value = "response"

        with patch("async_runner.asyncio.get_event_loop", return_value=loop):
            with patch("async_runner.asyncio.run") as asyncio_run:
                self.assertEqual(run_coroutine(coroutine), "response")

        loop.run_until_complete.assert_called_once_with(coroutine)
        asyncio_run.assert_not_called()

    def test_uses_asyncio_run_when_loop_lookup_fails(self):
        coroutine = object()

        with patch(
            "async_runner.asyncio.get_event_loop",
            side_effect=RuntimeError("no current event loop"),
        ):
            with patch("async_runner.asyncio.run", return_value="response") as asyncio_run:
                self.assertEqual(run_coroutine(coroutine), "response")

        asyncio_run.assert_called_once_with(coroutine)

    def test_does_not_retry_runtime_error_from_execution(self):
        coroutine = object()
        loop = Mock()
        loop.run_until_complete.side_effect = RuntimeError("agent stream failed")

        with patch("async_runner.asyncio.get_event_loop", return_value=loop):
            with patch("async_runner.asyncio.run") as asyncio_run:
                with self.assertRaisesRegex(RuntimeError, "agent stream failed"):
                    run_coroutine(coroutine)

        loop.run_until_complete.assert_called_once_with(coroutine)
        asyncio_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
