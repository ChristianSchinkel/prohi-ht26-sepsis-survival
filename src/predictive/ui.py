import json
import joblib
import pandas as pd
import streamlit as st
from .core import ROOT, FEATURES, predict_death, digest

LABELS={'primary':'Primary cohort · 110,204 admissions','study':'Sepsis-3 study cohort · 19,051 admissions'}

@st.cache_resource
def load_frozen(cohort, checksum):
    path=ROOT/'predictive_results'/f'{cohort}_model.joblib'
    if digest(path)!=checksum:
        raise ValueError('Model checksum does not match the frozen manifest.')
    return joblib.load(path)

def render():
    st.title('Sepsis survival prediction')
    st.caption('Predictive analysis · UCI dataset 827')
    out=ROOT/'predictive_results'
    if not (out/'frozen_manifest.json').exists():
        st.info('Trained models are unavailable. Follow the training instructions in README_PREDICTIVE.md.')
        return
    manifest=json.loads((out/'frozen_manifest.json').read_text())
    cohort=st.selectbox('Norwegian training cohort',['primary','study'],format_func=LABELS.get)
    selected=manifest[cohort]['selected_model']
    st.write(f'Model selected using Norwegian internal validation: **{selected}**')
    st.info('Coursework research model. Clinical use has not been validated. The output describes an estimated in-hospital outcome and does not recommend treatment or admission priority.')
    predict_tab,evaluation_tab,methods_tab=st.tabs(['Prediction','Evaluation','Methods and data'])
    with predict_tab:
        with st.form('patient'):
            age=st.number_input('Age at admission (years)',min_value=0,max_value=100,value=65,step=1)
            sex=st.selectbox('Sex as encoded in the source dataset',[0,1],format_func=lambda v:'Male (0)' if v==0 else 'Female (1)')
            episode=st.number_input('Septic episode number',min_value=1,max_value=5,value=1,step=1)
            submitted=st.form_submit_button('Estimate probability')
        if submitted:
            x=pd.DataFrame([[age,sex,episode]],columns=FEATURES)
            p=float(predict_death(load_frozen(cohort,manifest[cohort]['model_sha256']),x)[0])
            st.metric('Estimated in-hospital death probability',f'{p:.1%}')
            st.metric('Estimated survival probability',f'{1-p:.1%}')
            st.caption('Population evaluation intervals do not provide an individual probability confidence interval.')
    with evaluation_tab:
        summary=pd.read_csv(out/'internal_summary.csv')
        st.subheader('Norwegian internal model comparison')
        st.caption('Mean of 10 repeats of stratified five-fold cross-validation. The selected model minimizes mean Brier score. These comparisons have no nested model-selection correction.')
        st.dataframe(summary[(summary.cohort==cohort)&(summary['sample']=='all')].drop(columns=['cohort','sample']),hide_index=True)
        if (out/'external_test.json').exists():
            test=json.loads((out/'external_test.json').read_text())
            result=test['models'][cohort]
            st.subheader('Korean external test')
            st.caption(f"{test['n']} patients; {test['deaths']} deaths. All predictions use the frozen Norwegian model. No Korean recalibration is applied.")
            rows=[]
            for key,value in result['metrics'].items():
                ci=result['ci'][key]
                rows.append({'Metric':key,'Estimate':value,'95% lower':ci['low'],'95% upper':ci['high']})
            st.dataframe(pd.DataFrame(rows),hide_index=True)
        if (out/f'{cohort}_evaluation.png').exists():
            st.image(str(out/f'{cohort}_evaluation.png'))
        st.subheader('First-episode sensitivity analysis')
        st.dataframe(summary[(summary.cohort==cohort)&(summary['sample']=='first_episode')].drop(columns=['cohort','sample']),hide_index=True)
    with methods_tab:
        st.write('Inputs: age in years, sex encoded 0/1, and sequential septic episode number. Target: death = 1 minus the original alive outcome label.')
        st.write('Norwegian primary and study cohorts overlap. Each cohort is analyzed independently. Korea is used exclusively for external performance evaluation.')
        st.write('Patient IDs and dates are unavailable. Repeated admissions cannot be grouped by patient, and temporal validation cannot be performed. Identical clinical records are retained. First-episode analysis reduces repeated-admission concerns but does not recover patient identifiers.')
        st.write('Average precision is the precision–recall summary. Calibration intercept and slope are jointly estimated as evaluation diagnostics. Bootstrap intervals condition on the saved predictions and exclude training variability. Internal intervals use the first out-of-fold repeat; the main table averages ten repeats.')
        st.link_button('UCI dataset and documentation','https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records')
