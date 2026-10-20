import numpy as np
import pandas as pd
import pytest
from predictive.core import FEATURES, Features, load_data, make_model, predict_death, validate_inputs
from predictive.evaluation import metrics

def test_outcome_direction():
    df,y=load_data('primary')
    assert y.sum()==8105
    assert len(y)==110204
    assert (y==1-df.iloc[:,-1]).all()

def test_baseline_is_training_death_rate():
    x=pd.DataFrame([[20,0,1],[30,1,1],[40,0,2],[60,1,1]],columns=FEATURES)
    m=make_model('prevalence').fit(x,[0,0,0,1])
    np.testing.assert_allclose(predict_death(m,x),.25)

@pytest.mark.parametrize('row', [[-1,0,1],[101,0,1],[30,2,1],[30,0,0],[30,0,6],[30.5,0,1],[np.nan,0,1]])
def test_bad_input(row):
    with pytest.raises(ValueError):
        validate_inputs(pd.DataFrame([row],columns=FEATURES))

def test_spline_has_linear_tails_and_unchanged_knots():
    x=np.column_stack([np.arange(101),np.zeros(101),np.ones(101)])
    transform=Features('spline_lr').fit(x)
    knots=transform.knots_.copy()
    tail=np.array([[105,0,1],[106,0,1],[107,0,1]])
    np.testing.assert_allclose(np.diff(transform.transform(tail),n=2,axis=0),0,atol=1e-10)
    np.testing.assert_array_equal(transform.knots_,knots)

def test_metrics_known_values():
    m=metrics([0,0,1,1],[.1,.2,.8,.9])
    assert m['roc_auc']==1
    assert m['average_precision']==1
    assert abs(m['brier']-.025)<1e-12

def test_constant_calibration_is_undefined():
    assert np.isnan(metrics([0,1],[.5,.5])['calibration_slope'])
