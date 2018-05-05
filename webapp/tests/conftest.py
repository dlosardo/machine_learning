"""
docker-compose exec website py.test webapp/tests
docker-compose exec website py.test --cov-report term-missing --cov webapp
docker-compose exec website flake8 . --exclude __init__.py
"""
import pytest
from webapp.app import create_app


@pytest.yield_fixture(scope='session')
def app():
    """
    Setup flask test app, only gets executed once at
        beginning of test suite (session scoped)

    :return: Flask app instance
    """
    settings_override = "testing"

    _app = create_app(settings_override=settings_override)

    # Establish an application context before running the tests.
    ctx = _app.app_context()
    ctx.push()

    yield _app

    ctx.pop()


@pytest.yield_fixture(scope='function')
def client(app):
    """
    Setup an app client, this gets executes for each test function.
    Test client -- Watered down head-less browser

    :param app: Pytest fixture
    :return: Flask app client
    """
    yield app.test_client()
