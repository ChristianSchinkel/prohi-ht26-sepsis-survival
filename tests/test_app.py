from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_both_frozen_models_predict():
    page=Path(__file__).resolve().parents[1]/'src/pages/3_predictive.py'
    app=AppTest.from_file(str(page),default_timeout=30).run()
    assert not app.exception
    for cohort in ['primary','study']:
        app.selectbox[0].select(cohort).run()
        app.button[0].click().run()
        assert not app.exception
        assert len(app.metric)==2
        death=float(app.metric[0].value.rstrip('%'))
        survival=float(app.metric[1].value.rstrip('%'))
        assert 0<=death<=100
        assert abs(death+survival-100)<0.11
