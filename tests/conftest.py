"""The public release leaves out the held-out and sealed task sets and the
deception study's task generator. A test that needs one of them raises
WithheldTaskSet, which is reported here as a skip with its reason."""

import pytest

from opacity.harvest.loader import WithheldTaskSet


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(WithheldTaskSet):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {call.excinfo.value}")
