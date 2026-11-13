"""Verify real Qwen1.5B SFT merge/new-adapter/reference equivalence on CUDA."""
import hashlib,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel,LoraConfig,get_peft_model
from huggingface_hub import snapshot_download
from math_posttrain.checkpoints import verify
ROOT=Path(__file__).resolve().parents[2]
REV='8faed761d45a263340a0528343f099c05c9a4323'
def main():
 (ROOT/'grpo-study-v2/reports').mkdir(parents=True,exist_ok=True)
 assert torch.cuda.is_available()
 hashes=verify(ROOT/'runs/sft_frozen')
 cache=Path(snapshot_download('Qwen/Qwen2.5-1.5B',revision=REV,local_files_only=True))
 weight_checks={}
 for f in cache.glob('*.safetensors'):
  blob=f.resolve();sha=hashlib.sha256(blob.read_bytes()).hexdigest()
  assert blob.name==sha,'weight blob does not match content-addressed SHA256'
  weight_checks[f.name]=sha
 assert weight_checks
 tok=AutoTokenizer.from_pretrained(cache,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(cache,local_files_only=True,torch_dtype=torch.float32).cuda().eval()
 model=PeftModel.from_pretrained(model,ROOT/'runs/sft_frozen').eval()
 tokens=tok('Solve: 7 + 8.\nFinal Answer:',return_tensors='pt').to('cuda')
 def logits(m):
  with torch.inference_mode():return m(**tokens).logits[:,-1,:].float().cpu()
 before=logits(model)
 model=model.merge_and_unload().eval();merged=logits(model)
 torch.testing.assert_close(before,merged,atol=1e-3,rtol=1e-3)
 torch.manual_seed(43)
 model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM')).eval()
 new=logits(model)
 with model.disable_adapter():reference=logits(model)
 torch.testing.assert_close(merged,new,atol=1e-5,rtol=1e-5)
 torch.testing.assert_close(merged,reference,atol=1e-5,rtol=1e-5)
 result=dict(status='passed',model_revision=REV,weight_sha256=weight_checks,sft_hashes=hashes,torch=torch.__version__,gpu=torch.cuda.get_device_name(),merge_max_abs=float((before-merged).abs().max()),new_adapter_max_abs=float((new-merged).abs().max()),disabled_adapter_reference_max_abs=float((reference-merged).abs().max()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),checked='last-token logits on one fixed synthetic prompt; not all-input equivalence',timestamp=time.time())
 (ROOT/'grpo-study-v2/reports/gpu-startpoint.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['sft_hashes','weight_sha256']},indent=2))
if __name__=='__main__':main()
