"""Smoke test: every page opens from the app entry point without errors."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ENTRY = Path(__file__).resolve().parents[1] / "src" / "hello.py"
PAGES = [
    "pages/1_descriptive.py",
    "pages/2_diagnostic.py",
    "pages/3_predictive.py",
    "pages/4_prescriptive.py",
]


def test_home_page_renders():
    app = AppTest.from_file(str(ENTRY), default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == "Sepsis survival decision support"


@pytest.mark.parametrize("page", PAGES)
def test_each_page_renders(page):
    app = AppTest.from_file(str(ENTRY), default_timeout=300).run()
    app.switch_page(page).run()
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.title) >= 1
