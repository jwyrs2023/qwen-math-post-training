"""Fail closed on incomplete pilot status, lineage or missing prompt-linked logs."""
import argparse,json
from pathlib import Path
from math_posttrain.checkpoints import verify
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--sft',default='runs/sft_frozen');p.add_argument('--kind',choices=['sft','grpo'],required=True);a=p.parse_args()
r=Path(a.run); expected=verify(a.sft)
status=json.loads((r/'status.json').read_text())
assert status['status']=='complete' and status['step']==status['max_steps'],'pilot stopped/incomplete'
assert json.loads((r/'sft-hashes.json').read_text())==expected,'wrong starting checkpoint'
verify(r/'final')
if a.kind=='grpo':
 batches=[json.loads(x) for x in (r/'reward-groups.jsonl').read_text().splitlines()]
 assert batches
 for b in batches:
  assert b['rows'] and all(x['sample_id'] and x['prompt'] and isinstance(x['completion'],str) for x in b['rows']),'missing prompt-linked telemetry'
  assert b['generated_tokens'] is not None,'missing token accounting'
else:
 ids=json.loads((r/'training-ids.json').read_text()); assert len(ids)==len(set(ids))==1999
 assert not json.loads((r/'excluded.json').read_text()),'SFT excluded common-pool questions'
print('Pilot artifact checks passed. Numerical merge/reference checks still require dedicated GPU verification.')
