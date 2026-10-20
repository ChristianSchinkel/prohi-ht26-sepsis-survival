"""Evaluate frozen models. Never fit or recalibrate a predictive model here."""
import json
from datetime import datetime, timezone
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from .core import ROOT, FILES, FEATURES, load_data, predict_death, digest
from .evaluation import metrics, bootstrap

def run():
    out = ROOT/'predictive_results'
    manifest = json.loads((out/'frozen_manifest.json').read_text())
    if digest(ROOT/'predictive_protocol.json') != manifest['protocol_sha256']:
        raise ValueError('Protocol changed after model freeze.')
    # Model and protocol integrity verified before external predictions are made.
    for cohort in ['primary', 'study']:
        if digest(out/f'{cohort}_model.joblib') != manifest[cohort]['model_sha256']:
            raise ValueError('Frozen model changed.')
    df, y = load_data('korea')
    result = {'evaluated_utc': datetime.now(timezone.utc).isoformat(), 'n': len(y), 'deaths': int(y.sum()),
              'data_sha256': digest(ROOT/'resources'/'sepsis_survival_dataset'/FILES['korea']),
              'recalibration_applied': False, 'models': {}}
    predictions = df.copy()
    for cohort in ['primary', 'study']:
        model = joblib.load(out/f'{cohort}_model.joblib')
        p = predict_death(model, df[FEATURES])
        predictions[f'{cohort}_death_probability'] = p
        result['models'][cohort] = {'model': manifest[cohort]['selected_model'], 'metrics': metrics(y,p),
                                   'ci': bootstrap(y,p,manifest['bootstrap_repetitions'])}
    predictions.to_csv(out/'korea_predictions.csv',index=False)
    (out/'external_test.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
