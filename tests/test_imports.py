"""Smoke tests for package imports."""

from blograg import app


def test_package_exports_app() -> None:
    assert app is not None
