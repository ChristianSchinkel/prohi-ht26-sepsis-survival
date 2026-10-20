"""Export reproducible evaluation figures from saved predictions."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from .core import ROOT, TARGET

def calibration_points(y,p):
    data=pd.DataFrame({'y':y,'p':p})
    data['bin']=pd.qcut(data.p, q=10, duplicates='drop')
    return data.groupby('bin',observed=True).agg(predicted=('p','mean'),observed=('y','mean'),n=('y','size'))

def run():
    out=ROOT/'predictive_results'
    manifest=json.loads((out/'frozen_manifest.json').read_text())
    korea=pd.read_csv(out/'korea_predictions.csv')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    for cohort in ['primary','study']:
        selected=manifest[cohort]['selected_model']
        data=np.load(out/f'{cohort}_all_{selected}_oof.npz')
        internal=(data['y'],data['predictions'][0])
        external=(1-korea[TARGET].to_numpy(),korea[f'{cohort}_death_probability'].to_numpy())
        fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
        for row,(label,(y,p)) in enumerate([('Norway OOF repeat 1',internal),('Korea external test',external)]):
            fpr,tpr,_=roc_curve(y,p)
            axes[row,0].plot(fpr,tpr,color='#1f5c75',lw=2)
            axes[row,0].plot([0,1],[0,1],color='gray',ls=':')
            axes[row,0].set(xlabel='False positive rate',ylabel='Sensitivity',title=f'{label}: ROC',xlim=(0,1),ylim=(0,1))
            precision,recall,_=precision_recall_curve(y,p)
            axes[row,1].step(recall,precision,where='post',color='#1f5c75')
            axes[row,1].axhline(y.mean(),color='gray',ls=':',label='Observed death prevalence')
            axes[row,1].set(xlabel='Recall of deaths',ylabel='Precision for deaths',title=f'{label}: precision–recall',xlim=(0,1),ylim=(0,1))
            points=calibration_points(y,p)
            points.to_csv(out/f'{cohort}_{row}_calibration.csv',index=False)
            axes[row,2].plot([0,1],[0,1],color='gray',ls=':')
            axes[row,2].scatter(points.predicted,points.observed,s=35,color='#1f5c75')
            axes[row,2].set(xlabel='Mean predicted death probability',ylabel='Observed death proportion',title=f'{label}: calibration',xlim=(0,1),ylim=(0,1))
        fig.suptitle(f'{cohort.capitalize()} training cohort | {selected} | death is the positive class')
        fig.savefig(out/f'{cohort}_evaluation.png',dpi=160)
        fig.savefig(out/f'{cohort}_evaluation.pdf')
        plt.close(fig)

if __name__=='__main__':
    run()
