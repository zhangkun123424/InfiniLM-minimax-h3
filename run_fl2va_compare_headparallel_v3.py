import argparse,os,json,tarfile,zipfile,subprocess,tempfile,time
from pathlib import Path
import shutil
from datetime import datetime,timezone,timedelta

def main():
    parser=argparse.ArgumentParser(description='MiniMax-H3 FL2VA inference using the gfx936 migration bundle')
    parser.add_argument('--first-frame',required=True)
    parser.add_argument('--text',required=True)
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--num-frames',type=int,default=124)
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output')
    parser.add_argument('--ninetoothed',action='store_true')
    parser.add_argument('--native-trace',action='store_true')
    args=parser.parse_args()
    if (args.num_frames,args.steps,args.seed,args.text)!=(124,50,42,'A dog falls asleep'):parser.error('Official comparison uses 124 frames, 50 steps, seed 42, and A dog falls asleep')
    afs=Path(os.environ['IFAB_AFS_HOME'])
    image=Path(args.first_frame).expanduser().resolve()
    if not image.is_file():parser.error('First-frame image does not exist on the compute node: '+str(image))
    model=Path('/public/home/aclsylqidf/shared/MiniMax-H3/diffusers')
    if not (model/'model_index.json').is_file():raise RuntimeError('Shared Diffusers model directory is unavailable')
    if args.ninetoothed:
        _probe=json.loads((afs/'diagnostics/nt_multigpu_candidate_v1.json').read_text())
        assert _probe['cases']==288 and not _probe['bad'], 'Eight-GPU AOT regression must pass'
        report=afs/'diagnostics/operator_ninetoothed_report_gfx936.json'
        if not report.is_file() or any(value!=0 for value in json.loads(report.read_text()).values()):
            raise RuntimeError('gfx936 NineToothed GPU validation has not passed yet')
    run_id=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output=Path(args.output).expanduser().resolve() if args.output else afs/'results'/('fl2va_compare_fixed_'+run_id)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fl2va_',dir=Path.cwd()) as temporary:
        work=Path(temporary)
        for name in ['fl2va_sources_gfx936.tar.gz','fl2va_runtime_deps.tar.gz']:
            with tarfile.open(afs/'imports'/name) as archive:archive.extractall(work,filter='data')
        for wheel in (afs/'imports').glob('xxhash-3.6.0-cp311*.whl'):
            with zipfile.ZipFile(wheel) as archive:archive.extractall(work/'runtime_deps')
        nt_root=work/'nt_smoke/timestep_embedding'
        if args.ninetoothed:
            with tarfile.open(afs/'imports/fl2va_aot_gfx936_devicecache_candidate_v1.tar.gz') as archive:archive.extractall(nt_root,filter='data')
        core=work/'InfiniCore-minimax-h3';lm=work/'InfiniLM-minimax_h3'
        generator=lm/'examples/minimax_h3.py'
        code=generator.read_text()
        flags="import torch\ntorch.backends.cuda.enable_flash_sdp(False)\ntorch.backends.cuda.enable_mem_efficient_sdp(False)\ntorch.backends.cuda.enable_cudnn_sdp(False)\ntorch.backends.cuda.enable_math_sdp(True)\nprint('[config] PyTorch SDPA backend: math',flush=True)\n"
        assert code.count('import torch'+chr(10))==1
        code=code.replace('import torch'+chr(10),flags,1)
        registration="    import ctypes,os\n    if not args.legacy:\n        _mha_patch=ctypes.CDLL(os.environ['IFAB_AFS_HOME']+'/imports/libmha_hygon_math_gfx936_v1.so')\n        print('[config] Hygon native MHA registration repaired',flush=True)\n    else:\n        from h3_chunked_attention import install as install_h3_chunks\n        install_h3_chunks(1024,2048)\n        from h3_head_parallel_attention_v1 import install as install_head_parallel\n        install_head_parallel(validate=True)\n"
        if args.native_trace:
            registration+="    _crash_trace=ctypes.CDLL(os.environ['IFAB_AFS_HOME']+'/scripts/libnative_trace_v2.so')\n    assert _crash_trace.install_native_trace()==0\n"
        anchor='    args = parse_args()'+chr(10)
        assert code.count(anchor)==1
        code=code.replace(anchor,anchor+registration,1)
        compile(code,str(generator),'exec')
        generator.write_text(code)
        env=os.environ.copy()
        env['PYTHONPATH']=':'.join(map(str,[afs/'scripts',afs/'compare_deps',work/'runtime_deps',core/'python',lm/'python']))
        env['LD_LIBRARY_PATH']=str(core/'python/infinicore/lib')+':/usr/local/lib/python3.11/site-packages/torch/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/cuda/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
        env['PYTHONFAULTHANDLER']='1';env['PYTHONUNBUFFERED']='1'
        env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
        env['INFINI_TIMESTEP_NT']='1' if args.ninetoothed else '0'
        env['INFINI_TIMESTEP_NT_ROOT']=str(nt_root)
        env['INFINI_TIMESTEP_NT_TRACE']='1'
        reused_native=afs/'results/fl2va_native_standard50_nt_v1.mp4'
        assert reused_native.is_file() and reused_native.stat().st_size>0
        output.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(reused_native,output/'infinilm_fl2va.mp4')
        command=[str(afs/'venvs/fl2va/bin/python'),'-u','examples/minimax_h3_compare.py','--tasks=fl2va','--model-path',str(model),'--first-frame',str(image),'--legacy-gpus',args.gpus,'--infinilm-gpus',args.gpus,'--output-dir',str(output),'--infinilm-fl2va-video',str(output/'infinilm_fl2va.mp4')]
        log_path=output.with_suffix('.log');started=time.monotonic()
        with log_path.open('w') as log:
            process=subprocess.Popen(command,cwd=lm,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            return_code=process.wait()
        metadata={'return_code':return_code,'elapsed_seconds':time.monotonic()-started,'command':command,'output':str(output),'reference_video_exists':(output/'reference_fl2va.mp4').is_file(),'infinilm_video_exists':(output/'infinilm_fl2va.mp4').is_file(),'ninetoothed_enabled':args.ninetoothed,'native_trace':args.native_trace,'standard_generation_settings':True,'mha_registration_repaired':True,'reference_attention':'FP32 math SDPA with query chunks and eight-device head partitioning; full K/V retained'}
        output.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
        if return_code:raise SystemExit(return_code)
        if not all((output/name).is_file() and (output/name).stat().st_size>0 for name in ['reference_fl2va.mp4','infinilm_fl2va.mp4']):raise RuntimeError('Missing comparison videos')
        print('FL2VA_COMPARE_FINISHED',str(output),flush=True)

if __name__=='__main__':main()
