"""A100 microbenchmarks; run on an idle GPU, without a serving process.

python -m dsv41.bench_long_prefill --device 2 --baseline-ref c182930
These timings measure components, not end-to-end generation throughput.
"""
import argparse
import subprocess

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--device', type=int, default=2)
ap.add_argument('--baseline-ref', default='c182930')
a = ap.parse_args()
import json, time
import torch
from dsv41 import model
ns = dict(vars(model))
src = subprocess.check_output(['git', 'show', f'{a.baseline_ref}:dsv41/model.py'], text=True)
exec(src[src.index('class Compressor:'):src.index('class Indexer:')], ns)
Old = ns['Compressor']
torch.manual_seed(42)
torch.cuda.set_device(a.device)
cfg = model.Args(dict(compress_ratios=[2], head_dim=512, norm_eps=1e-20))
w = {'compressor.norm.weight':torch.ones(512, device=f'cuda:{a.device}', dtype=torch.bfloat16),
     'compressor.wkv.weight':torch.randn(512,5120,device=f'cuda:{a.device}')*.01,
     'compressor.wgate.weight':torch.randn(512,5120,device=f'cuda:{a.device}')*.01}
for n in (128,512,1024,2048):
 x=torch.randn(1,n,5120,device=f'cuda:{a.device}',dtype=torch.bfloat16)
 outputs=[]; timings=[]
 for cls in (Old,model.Compressor):
  c=cls(cfg,0,w,f'cuda:{a.device}')
  for i in range(3): out=c(x,32768)
  torch.cuda.synchronize()
  t=time.perf_counter()
  for i in range(10): out=c(x,32768)
  torch.cuda.synchronize()
  timings.append((time.perf_counter()-t)*100)
  outputs.append(out.clone())
 torch.testing.assert_close(*outputs,rtol=0,atol=0)
 print(json.dumps(dict(tokens=n,old_ms=timings[0],new_ms=timings[1],speedup=timings[0]/timings[1],exact=True)),flush=True)

import json, os, time
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn.functional as F
from dsv41 import model

torch.cuda.set_device(a.device)
torch.manual_seed(17)
device = torch.device(f'cuda:{a.device}')
obj = model.Indexer.__new__(model.Indexer)
obj.ratio, obj.rope_head_dim = 1, 0
obj.owns_k = False
obj.device = device
obj.n_heads, obj.head_dim = 32, 128
obj.wq_b = torch.randn(4096, 128, device=device)*.01
obj.weights_proj = torch.randn(32, 128, device=device)*.01
obj.softmax_scale = 128**-.5
obj.cos = obj.sin = None
obj.topk = 512
obj.is_candidate_source = obj.uses_candidates = False
n, start = 1024, 117000
obj.shared = SimpleNamespace(index_owner=0, index_k={(0, device): torch.randn(1,start+n,128,device=device)})
x,qr = torch.randn(1,n,128,device=device),torch.randn(1,n,128,device=device)
results=[]
with patch.object(model,'linear_fp8',F.linear),patch.object(model,'rope_',lambda *a,**k: None),patch.object(model,'fake_quant_fp4',lambda x,*a: x):
 for chunk in (10000,128):
  os.environ['DSV41_INDEX_QUERY_CHUNK']=str(chunk)
  out=obj(x,qr,None,start,128)
  torch.cuda.synchronize()
  torch.cuda.reset_peak_memory_stats(device)
  baseline=torch.cuda.memory_allocated(device)
  t=time.perf_counter()
  for _ in range(3): out=obj(x,qr,None,start,128)
  torch.cuda.synchronize()
  results.append(out.cpu())
  print(json.dumps(dict(chunk=chunk,tokens=start+n,ms=(time.perf_counter()-t)*1000/3,peak_scratch_mib=(torch.cuda.max_memory_allocated(device)-baseline)/2**20)),flush=True)
torch.testing.assert_close(*results,rtol=0,atol=0)
print('indices exact match',flush=True)
