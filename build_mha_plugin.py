"""Build the existing Hygon math MHA registration without FlashAttention."""
import argparse,hashlib,json,os,shutil,subprocess,tarfile,tempfile
from pathlib import Path

def main():
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--core-root', type=Path)
    parser.add_argument('--core-lib-dir', type=Path)
    parser.add_argument('--dtk-root', type=Path, default=Path('/opt/dtk'))
    args=parser.parse_args()
    afs=Path(os.environ['IFAB_AFS_HOME'])
    output=Path(args.output) if args.output else afs/'imports/libmha_hygon_math_fp32_chunked_gfx936_v3.so'
    torch_root=Path(torch.__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='mha_build_',dir=Path.cwd()) as tmp:
        work=Path(tmp)
        if args.core_root:
            core = args.core_root.resolve()
        else:
            with tarfile.open(afs/'imports/fl2va_sources_gfx936.tar.gz') as archive:
                for member in archive.getmembers():
                    if member.name.startswith(('InfiniCore-minimax-h3/include/', 'InfiniCore-minimax-h3/src/infinicore/', 'InfiniCore-minimax-h3/python/infinicore/lib/')):
                        archive.extract(member, work, filter='data')
            core = work/'InfiniCore-minimax-h3'
        source = work/'mha_flashattn_hygon.cc'
        shutil.copyfile(core/'src/infinicore/ops/multi_head_attention/hygon/mha_flashattn_hygon.cc', source)
        library_dir = args.core_lib_dir or core/'python/infinicore/lib'
        output.parent.mkdir(parents=True, exist_ok=True)
        (afs/'diagnostics').mkdir(parents=True, exist_ok=True)
        text=source.read_text()
        assert 'defined(ENABLE_HYGON_API) && defined(ENABLE_FLASH_ATTN)' in text
        text=text.replace('defined(ENABLE_HYGON_API) && defined(ENABLE_FLASH_ATTN)','defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)')
        text=text.replace('#include "../../../adaptor/flash_attn/hygon/flash_attn_hygon.hpp"'+chr(10),'')
        text=text.replace('ENABLE_HYGON_API && ENABLE_FLASH_ATTN','ENABLE_HYGON_API && ENABLE_ATEN')
        for name in ('q','k','v'):
            old='auto '+name+' = infinicore::adaptor::to_aten_tensor(p->'+name+');'
            assert old in text
            text=text.replace(old,'auto '+name+' = infinicore::adaptor::to_aten_tensor(p->'+name+').to(at::kFloat);')
        text=text.replace('    auto attn_weight = at::matmul(q.permute({0, 2, 1, 3}), k_attn.permute({0, 2, 3, 1})) * scale;','    if (!is_causal && q.size(1) > 1024) {\n        auto key_transposed = k_attn.permute({0, 2, 3, 1});\n        auto value_transposed = v_attn.permute({0, 2, 1, 3});\n        for (int64_t start = 0; start < q.size(1); start += 1024) {\n            int64_t count = q.size(1) - start;\n            if (count > 1024) count = 1024;\n            auto query_chunk = q.narrow(1, start, count).permute({0, 2, 1, 3});\n            auto weights = at::matmul(query_chunk, key_transposed) * scale;\n            auto probabilities = at::softmax(weights, -1);\n            auto result = at::matmul(probabilities, value_transposed).permute({0, 2, 1, 3});\n            out_work.narrow(1, start, count).copy_(result);\n        }\n        if (out_need_copy_back) p->out->copy_from(out_work_ic);\n        return;\n    }\n'+'    auto attn_weight = at::matmul(q.permute({0, 2, 1, 3}), k_attn.permute({0, 2, 3, 1})) * scale;')
        source.write_text(text)
        command=['g++','-std=c++17','-shared','-fPIC','-O2','-g','-D__HIP_PLATFORM_AMD__','-DENABLE_HYGON_API','-DENABLE_ATEN','-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),'-I'+str(core/'include'),'-I'+str(core/'src/infinicore/ops/multi_head_attention/hygon'),'-I'+str(torch_root/'include'),'-I'+str(torch_root/'include/torch/csrc/api/include'),'-I'+str(args.dtk_root/'include'),'-I'+str(args.dtk_root/'hip/include'),str(source),'-L'+str(library_dir),'-linfinicore_cpp_api','-L'+str(torch_root/'lib'),'-ltorch','-ltorch_cpu','-ltorch_hip','-lc10','-lc10_hip','-o',str(output)]
        print('MHA_PLUGIN_BUILD',json.dumps(command),flush=True)
        subprocess.run(command,check=True)
        output.chmod(0o770)
        manifest=dict(output=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),torch=torch.__version__,cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),source_sha256=hashlib.sha256(text.encode()).hexdigest(),command=command)
        (afs/'diagnostics/mha_hygon_fp32_chunked_build_manifest_v3.json').write_text(json.dumps(manifest,indent=2))
        print('MHA_PLUGIN_BUILD_COMPLETE',json.dumps(manifest),flush=True)
if __name__=='__main__':main()
