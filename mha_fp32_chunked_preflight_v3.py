import json
import os
from pathlib import Path

def run_validation():
    import torch
    import infinicore
    from infinicore.ops.mha import mha
    rows=[]
    with torch.random.fork_rng(devices=list(range(8))):
        for i in range(8):
            torch.cuda.set_device(i)
            infinicore.set_device(infinicore.device('cuda',i))
            for dtype in [torch.float32,torch.bfloat16]:
                torch.manual_seed(1234)
                q=torch.randn((1,2051,2,128),device=f'cuda:{i}',dtype=dtype)
                k=torch.randn((1,2081,2,128),device=f'cuda:{i}',dtype=dtype)
                v=torch.randn_like(k)
                torch.cuda.synchronize(i)
                output=mha(infinicore.from_torch(q),infinicore.from_torch(k),infinicore.from_torch(v),scale=128**-.5,is_causal=False)
                torch.cuda.synchronize(i)
                value=torch.empty_like(q)
                infinicore.from_torch(value).copy_(output)
                infinicore.sync_device()
                torch.cuda.synchronize(i)
                reference=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2).float(),k.transpose(1,2).float(),v.transpose(1,2).float()).transpose(1,2).to(dtype)
                error=(value.float()-reference.float()).abs().max().item()
                limit=0.001 if dtype==torch.bfloat16 else 1e-5
                assert error<=limit,(i,str(dtype),error)
                rows.append({'device':i,'dtype':str(dtype),'query_tokens':2051,'key_tokens':2081,'max_abs':error})
    torch.cuda.set_device(0)
    infinicore.set_device(infinicore.device('cuda',0))
    p=Path(os.environ['IFAB_AFS_HOME'])/'diagnostics/mha_fp32_chunked_preflight_v3.json'
    p.write_text(json.dumps(rows,indent=2));p.chmod(0o660)
    print('CHUNKED_FP32_MHA_PREFLIGHT_PASSED',len(rows),flush=True)
