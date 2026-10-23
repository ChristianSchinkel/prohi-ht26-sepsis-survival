# prohi-ht26-sepsis-survival

Dataset: Sepsis Survival Minimal Clinical Records, UCI Machine Learning Repository dataset 827. It contains 110,204 hospital admissions from 84,811 patients in Norway during 2011–2012.

## Description

This dataset contains minimal clinical records for patients who were admitted to hospitals in Norway during 2011–2012. The goal is to predict the survival of these patients based on their clinical data.

## License

The project is licensed under the terms in the [LICENSE](LICENSE) file.

## Running the app

**Important:** Use Python 3.13. Python 3.14 or later is not supported.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/main.py               # or: streamlit run src/hello.py
```

Run from the repository root so `.streamlit/config.toml` (shared theme) is picked up.

## App structure

| Page | File | Analysis code | Owner |
|---|---|---|---|
| Home | `src/home.py` | – | Kai |
| 1 · Descriptive | `src/pages/1_descriptive.py` | in the page | Christian |
| 2 · Diagnostic | `src/pages/2_diagnostic.py` | `src/diagnostic/analysis.py` | Anupom |
| 3 · Predictive | `src/pages/3_predictive.py` | `src/predictive/`, frozen models in `predictive_results/` | Tauska |
| 4 · Prescriptive | `src/pages/4_prescriptive.py` | `src/prescriptive/uc4_prescriptive.py`, outputs in `results/uc4/` | Kevin |

`src/hello.py` is the entry point and declares the navigation (`st.navigation`);
shared paths and the sidebar footer are in `src/ui_common.py`. All pages read the
datasets from `resources/sepsis_survival_dataset/`.

## Tests

```bash
python -m pytest -q tests
```

`tests/test_navigation.py` opens every page from the entry point;
`tests/test_app.py` and `tests/test_predictive.py` cover the predictive models.
