"""Generate standalone SVG figures and unlabeled, blinded annotation records."""
import csv,json,tarfile,random,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'grpo-study-v2/reports'
def chart(name,title,bars,max_value):
 parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{100+len(bars)*75}" viewBox="0 0 1000 {100+len(bars)*75}"><rect width="100%" height="100%" fill="white"/><text x="30" y="35" font-family="sans-serif" font-size="22">{html.escape(title)}</text>']
 for i,(label,value,color) in enumerate(bars):
  y=70+i*75;parts.append(f'<text x="30" y="{y+22}" font-family="sans-serif" font-size="17">{html.escape(label)}</text><rect x="300" y="{y}" width="{value/max_value*570}" height="32" fill="{color}"/><text x="{310+value/max_value*570}" y="{y+22}" font-family="sans-serif" font-size="17">{value}</text>')
 parts.append('</svg>');(OUT/name).write_text(''.join(parts))
def main():
 audit=json.loads((OUT/'history-audit.json').read_text())
 bars=[]
 for n in ['a','b']:
  for key,color in [('gained','#167a63'),('lost','#c85342')]:bars.append((f'GRPO-{n.upper()} {key}',len(audit['transfers'][n][key]),color))
 chart('historical-transfers.svg','V1: gains and regressions vs SFT (single seed)',bars,22)
 bars=[]
 for n in ['a','b']:
  for key,color in [('mixed_accuracy','#235fa4'),('format_mixed','#bc8420'),('advantage_sign_changed','#7859a4')]:bars.append((f'{n.upper()} {key}',audit['reward_diagnostics'][n][key],color))
 chart('historical-reward-signals.svg','V1: fixed-group reward diagnostics (800 groups each)',bars,450)
 with tarfile.open(ROOT/'reports/official-test-backup.tar.gz') as t:
  data={n:{r['sample_id']:r for r in map(json.loads,t.extractfile(f'runs/official_test_four_groups/{n}.jsonl').read().splitlines())} for n in ['sft','a','b']}
 rng=random.Random(43);records=[];key=[]
 for n in ['a','b']:
  for category in ['gained','lost']:
   ids=audit['transfers'][n][category][:];rng.shuffle(ids)
   for sid in ids: # all discordant pairs, not selectively chosen examples
    models=['sft',n];rng.shuffle(models);pair=f'pair_{len(key)+1:03d}'
    key.append(dict(pair_id=pair,sample_id=sid,branch=n,transition=category,left_model=models[0],right_model=models[1]))
    records.append(dict(pair_id=pair,sample_id=sid,gold_reference=data[n][sid]['answer'],left_output=data[models[0]][sid]['completion'],right_output=data[models[1]][sid]['completion'],error_category='',confidence='',notes=''))
 for name,rs in [('annotation-blinded.csv',records),('annotation-key.csv',key)]:
  with (OUT/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 print(f'Created two SVG plots and {len(records)} unlabeled paired records; no human labels invented.')
if __name__=='__main__':main()
