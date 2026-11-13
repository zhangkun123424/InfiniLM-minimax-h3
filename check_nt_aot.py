import json, math, sys
from pathlib import Path
import torch,ctypes as C,concurrent.futures,threading
hip=C.CDLL('libgalaxyhip.so.5');hip.hipCtxGetCurrent.argtypes=[C.POINTER(C.c_void_p)];hip.hipGetDevice.argtypes=[C.POINTER(C.c_int)];hip.hipGetErrorString.restype=C.c_char_p
class T(C.Structure):_fields_=[('data',C.c_void_p),('shape',C.POINTER(C.c_uint64)),('strides',C.POINTER(C.c_int64))]
libs=[]
for so in sorted(Path(sys.argv[1]).rglob('*.so')):
 name=so.stem;lib=C.CDLL(str(so));f=getattr(lib,'launch_'+name);f.argtypes=[C.c_void_p,T,T,T,T];f.restype=C.c_int;libs.append((name,lib,f))
if len(libs) != 6 or torch.cuda.device_count() != 8:
 raise RuntimeError('Expected six AOT libraries and eight visible DCUs')
print('GPUS',torch.cuda.device_count(),'LIBS',len(libs),flush=True)
def run(dev,barrier=None):
 torch.cuda.set_device(dev);torch.cuda.init();ctx=C.c_void_p();c=hip.hipCtxGetCurrent(C.byref(ctx));di=C.c_int();g=hip.hipGetDevice(C.byref(di));print('CONTEXT',dev,c,ctx.value,g,di.value,flush=True)
 if barrier:barrier.wait()
 results=[]
 for name,lib,f in libs:
  dim=256 if 'd256' in name else 16;dtype=torch.float16 if 'fp16' in name else torch.bfloat16 if 'bf16' in name else torch.float32
  for b in [1,4,17]:
   x=torch.linspace(-1000,1000,b,device='cuda',dtype=dtype);y=torch.empty((b,dim),device='cuda',dtype=torch.float32);shape=(C.c_uint64*1)(b);stride=(C.c_int64*1)(1);osh=(C.c_uint64*2)(b,dim//2);ost=(C.c_int64*2)(dim,1);period=C.c_double(10000.)
   status=f(torch.cuda.current_stream().cuda_stream,T(x.data_ptr(),shape,stride),T(C.addressof(period),None,None),T(y.data_ptr(),osh,ost),T(y.data_ptr()+dim//2*4,osh,ost))
   if status:results.append({'device':dev,'name':name,'batch':b,'status':status,'error':hip.hipGetErrorString(status).decode()});continue
   torch.cuda.synchronize();angles=x.float()[:,None]*torch.exp(-torch.log(torch.tensor(10000.,device='cuda'))*torch.arange(dim//2,device='cuda')/(dim//2));ref=torch.cat([angles.cos(),angles.sin()],1);err=(y-ref).abs().max().item();results.append({'device':dev,'name':name,'batch':b,'status':status,'maxdiff':err})
 return results
all_results=[]
for dev in range(torch.cuda.device_count()):all_results.extend(run(dev))
barrier=threading.Barrier(torch.cuda.device_count())
with concurrent.futures.ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as pool:
 for rows in pool.map(lambda d:run(d,barrier),range(torch.cuda.device_count())):all_results.extend(rows)
bad=[r for r in all_results if r['status'] or not math.isfinite(r['maxdiff']) or r['maxdiff']>0.002]
out={'cases':len(all_results),'bad':bad,'results':all_results};Path(sys.argv[2]).write_text(json.dumps(out,indent=2));print('MULTIGPU_RESULT',json.dumps({'cases':len(all_results),'bad_count':len(bad),'first_bad':bad[:12]}),flush=True)
raise SystemExit(bool(bad) or len(all_results) != 288)
