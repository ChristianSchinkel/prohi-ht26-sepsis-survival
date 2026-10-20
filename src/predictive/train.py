"""Development only. This module never loads the Korean external test file."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from .core import ROOT, FILES, FEATURES, CANDIDATES, load_data, make_model, predict_death, digest
from .evaluation import metrics, bootstrap

def run(repeats=10, bootstrap_n=1000):
    out = ROOT / 'predictive_results'
    out.mkdir(exist_ok=True)
    protocol = json.loads((ROOT / 'predictive_protocol.json').read_text())
    if (out / 'external_test.json').exists():
        raise RuntimeError('External test results already exist. Use a separate development copy for a new experiment.')
    summary, manifest, fold_metrics = [], {}, []
    start = time.time()
    for cohort in ['primary', 'study']:
        full, _ = load_data(cohort)
        manifest[cohort] = {'file': FILES[cohort], 'sha256': digest(ROOT/'resources'/'sepsis_survival_dataset'/FILES[cohort]),
                            'rows': len(full), 'missing': int(full.isna().sum().sum()),
                            'identical_rows': int(full.duplicated().sum()), 'patient_id_available': False,
                            'ranges': {c: [int(full[c].min()), int(full[c].max())] for c in full}}
        for sample in ['all', 'first_episode']:
            df = full if sample == 'all' else full.loc[full.episode_number == 1]
            x = df[FEATURES].reset_index(drop=True)
            y = 1 - df.iloc[:, -1].to_numpy(dtype=int)
            scores = {name: np.zeros((repeats, len(y))) for name in CANDIDATES}
            fold_ids = np.zeros((repeats, len(y)), dtype=int)
            for repeat in range(repeats):
                cv = StratifiedKFold(5, shuffle=True, random_state=protocol['seed']+repeat)
                for fold, (tr, va) in enumerate(cv.split(x, y)):
                    fold_ids[repeat, va] = fold
                    for name in CANDIDATES:
                        model = make_model(name)
                        model.fit(x.iloc[tr], y[tr])
                        pred = predict_death(model, x.iloc[va])
                        scores[name][repeat, va] = pred
                        fold_metrics.append({'cohort': cohort, 'sample': sample, 'repeat': repeat,
                                             'fold': fold, 'model': name, **metrics(y[va], pred)})
                print(f'{cohort}/{sample}: repeat {repeat+1}/{repeats}, elapsed {time.time()-start:.1f}s', flush=True)
            local = []
            for name in CANDIDATES:
                per_repeat = [metrics(y, p) for p in scores[name]]
                row = {'cohort': cohort, 'sample': sample, 'model': name, 'n': len(y), 'deaths': int(y.sum())}
                for key in per_repeat[0]:
                    values = np.array([m[key] for m in per_repeat])
                    row[key] = float(np.mean(values))
                local.append(row)
                summary.append(row)
                np.savez_compressed(out/f'{cohort}_{sample}_{name}_oof.npz', y=y, predictions=scores[name],
                                    folds=fold_ids, source_rows=df.index.to_numpy())
            if sample == 'all':
                selected = min([r for r in local if r['model'] != 'prevalence'], key=lambda r:r['brier'])['model']
                manifest[cohort]['selected_model'] = selected
                model = make_model(selected)
                model.fit(x, y)
                joblib.dump(model, out/f'{cohort}_model.joblib')
                manifest[cohort]['model_sha256'] = digest(out/f'{cohort}_model.joblib')
                manifest[cohort]['training_prevalence'] = float(y.mean())
                manifest[cohort]['internal_first_repeat'] = metrics(y, scores[selected][0])
                print(f'{cohort}: bootstrap for frozen internal predictions', flush=True)
                manifest[cohort]['internal_first_repeat_ci'] = bootstrap(y, scores[selected][0], bootstrap_n)
    pd.DataFrame(summary).to_csv(out/'internal_summary.csv', index=False)
    pd.DataFrame(fold_metrics).to_csv(out/'fold_metrics.csv', index=False)
    manifest['protocol_sha256'] = digest(ROOT/'predictive_protocol.json')
    manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['repeats'] = repeats
    manifest['bootstrap_repetitions'] = bootstrap_n
    (out/'frozen_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=True))
    print('Frozen models saved. Korea has not been loaded by this training program.', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.repeats, args.bootstrap)
