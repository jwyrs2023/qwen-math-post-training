"""Standalone figures from the complete measured V3 results only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];report=ROOT/'grpo-study-v3/reports'
def main():
 d=json.loads((report/'results.json').read_text());names=['uniform','mixture'];labels=['Uniform','Difficulty mixture'];colors=['#4878A8','#E19545'];fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
 ax=axes[0,0];m=d['results']['official_test']['metrics'];values=[m[n]['accuracy']*100 for n in names];bars=ax.bar(labels,values,color=colors,width=.55);ax.set_ylim(0,100);ax.set_ylabel('Official test accuracy (%)');ax.set_title('A  Test accuracy (1,319 questions)',loc='left')
 for bar,n,v in zip(bars,names,values):ax.text(bar.get_x()+bar.get_width()/2,v+2,f"{v:.2f}%\n{m[n]['correct']}/1319",ha='center',fontsize=10)
 ax=axes[0,1];c=d['results']['official_test']['uniform_to_mixture'];lo,hi=c['paired_ci95_pp'];delta=c['delta_pp'];ax.errorbar([delta],[0],xerr=[[delta-lo],[hi-delta]],fmt='o',color=colors[1],capsize=7);ax.axvline(0,color='#777',ls='--',lw=1);ax.set_yticks([]);ax.set_xlim(-.6,1.9);ax.set_xlabel('Mixture minus uniform (percentage points)');ax.set_title('B  Paired question bootstrap, 95% interval',loc='left');ax.text(.04,.85,f'{delta:+.2f} pp; 95% interval [{lo:.2f}, {hi:.2f}]\n18 gained / 10 lost; interval includes zero',transform=ax.transAxes,fontsize=10)
 ax=axes[1,0];mixed=[d['signals'][n]['mixed_groups']/640*100 for n in names];ax.bar(labels,mixed,color=colors,width=.55,label='Nonzero reward variance');ax.bar(labels,[100-v for v in mixed],bottom=mixed,color='#D9DEE5',width=.55,label='Zero variance');ax.set_ylim(0,100);ax.set_ylabel('Rollout groups (%)');ax.set_title('C  More groups with useful reward differences',loc='left');
 for x,n,v in zip(range(2),names,mixed):ax.text(x,v/2,f"{d['signals'][n]['mixed_groups']}/640\n{v:.2f}%",ha='center',va='center',color='white',weight='bold')
 ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),frameon=False,fontsize=9)
 ax=axes[1,1];train=[d['costs']['training_yuan'][n] for n in names];scoring=[0,d['costs']['scoring_generation_yuan']];ax.bar(labels,train,color=colors,width=.55,label='Training');ax.bar(labels,scoring,bottom=train,color='#738A78',width=.55,label='Scoring generation');ax.set_ylabel('Estimated yuan at 2.24 yuan/hour');ax.set_title('D  Incremental cost includes difficulty scoring',loc='left');
 for x,a,b in zip(range(2),train,scoring):ax.text(x,a+b+.08,f'{a+b:.2f}',ha='center')
 ax.set_ylim(0,5.8);ax.legend(frameon=False,fontsize=9);fig.suptitle('GRPO difficulty sampling: signal improves; accuracy gain remains uncertain\nSeed 43 | 160 updates per branch | 4 completions per question',fontsize=13)
 fig.supxlabel('Cost panel excludes evaluation, loading and idle; full session cost is reported separately.\nSingle training seed; reused GSM8K test; four-draw difficulty proxy.',fontsize=9)
 for ax in axes.ravel():ax.spines[['top','right']].set_visible(False)
 for ext in ['png','svg','pdf']:fig.savefig(report/f'results-overview.{ext}',dpi=180)
 print('Measured-result figures written.')
if __name__=='__main__':main()
