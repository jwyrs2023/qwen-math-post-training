"""Publication-style standalone plots from verified completed results only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'grpo-study-v2/reports'
def main():
 r=json.loads((OUT/'formal-results.json').read_text())
 names=['historical_sft','continue_sft','grpo_acc','grpo_accfmt']
 labels=['SFT anchor\n(historical)','Continue-SFT','GRPO-Acc','GRPO-AccFmt']
 colors=['#64748b','#2675aa','#12816d','#b47d20']
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 vals=[r['metrics'][n]['accuracy']*100 for n in names]
 bars=axes[0].bar(labels,vals,color=colors,width=.65)
 axes[0].set_ylim(0,100);axes[0].set_ylabel('GSM8K accuracy (%)');axes[0].set_title('Official test: 1,319 questions')
 for bar,n,v in zip(bars,names,vals):axes[0].text(bar.get_x()+bar.get_width()/2,v+1.4,f"{v:.2f}%\n{r['metrics'][n]['correct']}/1319",ha='center',fontsize=9)
 comps=['continue_sft→grpo_acc','continue_sft→grpo_accfmt'];ys=[1,0]
 for key,y,color in zip(comps,ys,colors[2:]):
  c=r['comparisons'][key];lo,hi=c['paired_ci95_pp'];v=c['delta_pp']
  axes[1].errorbar(v,y,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',capsize=5,color=color)
 axes[1].axvline(0,color='#888',lw=1,ls='--');axes[1].set_yticks(ys,['GRPO-Acc vs Continue-SFT','GRPO-AccFmt vs Continue-SFT']);axes[1].set_ylim(-.6,1.6)
 axes[1].set_xlabel('Accuracy difference (percentage points)');axes[1].set_title('Paired 95% bootstrap intervals')
 fig.suptitle('V2 fixed-endpoint comparison — one training seed (43)',fontsize=14)
 for ext in ['png','svg']:fig.savefig(OUT/f'formal-accuracy-comparison.{ext}',bbox_inches='tight')
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
 for n,label,color in zip(names[1:],labels[1:],colors[1:]):
  minutes=r['training_seconds'][n]/60;accuracy=r['metrics'][n]['accuracy']*100
  ax.scatter(minutes,accuracy,c=color,s=80);ax.annotate(label,(minutes,accuracy),xytext=(7,7),textcoords='offset points')
 ax.set_xlabel('Measured training + save time (minutes)');ax.set_ylabel('Official test accuracy (%)');ax.set_title('Observed outcomes at approximately matched time')
 ax.margins(x=.35,y=.35);ax.grid(alpha=.2)
 fig.text(.5,-.04,'Single seed; supervision and sample exposure differ. Time is not FLOPs.',ha='center',fontsize=9)
 for ext in ['png','svg']:fig.savefig(OUT/f'formal-time-vs-accuracy.{ext}',bbox_inches='tight')
 plt.close(fig)
 print('Saved verified-result accuracy, paired intervals, and measured-time plots.')
if __name__=='__main__':main()
