"""H3 unmasked attention: split heads across devices, retain all K/V."""
import concurrent.futures
import torch
import torch.nn.functional as F
_POOL=None

def head_parallel(query,key,value,devices,score_mib=1024,scale=None):
 global _POOL
 if _POOL is None:_POOL=concurrent.futures.ThreadPoolExecutor(max_workers=len(devices))
 source=query.device;heads=query.shape[2];per=heads//len(devices)
 assert heads%len(devices)==0
 torch.cuda.current_stream(source).synchronize()
 def worker(item):
  rank,device=item;start=rank*per;stop=start+per
  with torch.cuda.device(device),torch.inference_mode():
   q=query[:,:,start:stop,:].permute(0,2,1,3).contiguous().to(device)
   k=key[:,:,start:stop,:].permute(0,2,1,3).contiguous().to(device)
   v=value[:,:,start:stop,:].permute(0,2,1,3).contiguous().to(device)
   output=torch.empty_like(q)
   row=q.shape[0]*per*k.shape[2]*max(4,q.element_size())
   chunk=max(1,min(4096,int(score_mib*1024**2)//max(1,row)))
   for pos in range(0,q.shape[2],chunk):
    output[:,:,pos:pos+chunk]=F.scaled_dot_product_attention(q[:,:,pos:pos+chunk],k,v,dropout_p=0.0,is_causal=False,scale=scale)
   return output.permute(0,2,1,3).contiguous().to(source)
 return torch.cat(list(_POOL.map(worker,enumerate(devices))),dim=2)

def validate_runtime(devices):
 results=[]
 for source in [devices[0],devices[3]]:
  with torch.cuda.device(source),torch.inference_mode():
   for dtype in [torch.float32,torch.float16,torch.bfloat16]:
    for batch,length,heads,dim in [(1,17,8,16),(2,33,16,32),(1,65,56,128)]:
     q=torch.randn((batch,length,heads,dim),device=source,dtype=dtype);k=torch.randn_like(q);v=torch.randn_like(q)
     y=head_parallel(q,k,v,devices,score_mib=0.01,scale=0.125)
     r=F.scaled_dot_product_attention(q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3),scale=0.125).permute(0,2,1,3)
     err=(y-r).abs().max().item();torch.testing.assert_close(y,r,atol=0.003,rtol=0.02)
     results.append({'source':source,'dtype':str(dtype),'shape':list(q.shape),'max_abs_diff':err})
 for device in devices:
  with torch.cuda.device(device):torch.cuda.empty_cache()
 print('[H3 head parallel] runtime validation passed',len(results),'cases',flush=True)
 return results

def install(devices=tuple(range(8)),score_mib=1024,min_tokens=2048,validate=True):
 import diffusers.models.transformers.transformer_minimax_h3 as h3
 original=h3.dispatch_attention_fn
 if getattr(original,'_h3_head_parallel',False):return original
 devices=tuple(devices)
 assert len(devices)==8 and torch.cuda.device_count()>=8
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
 checks=[]
 if validate:
  with torch.random.fork_rng(devices=list(devices)):
   checks=validate_runtime(devices)
 def dispatch(query,key,value,*args,**kwargs):
  unsafe=torch.is_grad_enabled() and any(t.requires_grad for t in (query,key,value))
  if (args or query.ndim!=4 or key.ndim!=4 or value.ndim!=4 or query.device.type!='cuda' or query.shape[1]<=min_tokens or query.shape[2]%len(devices) or query.shape[2]!=key.shape[2] or key.shape[2]!=value.shape[2] or kwargs.get('attn_mask') is not None or kwargs.get('is_causal',False) or kwargs.get('dropout_p',0)!=0 or kwargs.get('parallel_config') is not None or unsafe):
   return original(query,key,value,*args,**kwargs)
  return head_parallel(query,key,value,devices,score_mib,kwargs.get('scale'))
 dispatch._h3_head_parallel=True;dispatch._h3_original=original;dispatch._h3_validation=checks;h3.dispatch_attention_fn=dispatch
 print('[H3 head parallel] enabled',devices,'FP32 math intermediates; full K/V',flush=True)
 return dispatch
