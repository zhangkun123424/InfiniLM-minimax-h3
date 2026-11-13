"""Apply the verified inference-time source transformations to a checkout."""
import argparse
import py_compile
import shutil
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one match in the pinned InfiniLM source: {old}')
    return text.replace(old, new, 1)


def prepare(lm, output):
    bridge = lm / 'python/infinilm/multimodal/minimax_h3.py'
    code = bridge.read_text()
    anchor = '        infinicore.from_torch(output).copy_(tensor)'
    code = replace_once(code, anchor, '        torch.cuda.synchronize(device)\n' + anchor)
    for indent in (8, 12):
        prefix = '\n' + ' ' * indent
        code = replace_once(
            code, prefix + 'return infinicore.from_torch(tensor.contiguous())',
            prefix + 'tensor = tensor.contiguous()'
            + prefix + 'torch.cuda.synchronize(tensor.device)'
            + prefix + 'return infinicore.from_torch(tensor)',
        )
    compile(code, str(bridge), 'exec')
    generator = (lm / 'examples/minimax_h3.py').read_text()
    generator = replace_once(generator, 'import torch\n', '''import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
''')
    anchor = '    args = parse_args()\n'
    generator = replace_once(generator, anchor, anchor + '''    import ctypes
    import os
    if args.legacy:
        from h3_chunked_attention import install as install_chunks
        from h3_head_parallel_attention_v1 import install as install_head_parallel
        install_chunks(1024, 2048)
        install_head_parallel(validate=True)
    else:
        mha_plugin = ctypes.CDLL(os.environ["SONGSHAN_MHA_PLUGIN"])
        from mha_fp32_chunked_preflight_v3 import run_validation
        run_validation()
''')
    compile(generator, str(output / 'minimax_h3.py'), 'exec')
    output.mkdir(parents=True, exist_ok=True)
    bridge.write_text(code)
    (output / 'minimax_h3.py').write_text(generator)
    shutil.copyfile(Path(__file__).with_name('minimax_h3_compare_official.py'),
                    output / 'minimax_h3_compare.py')
    py_compile.compile(str(output / 'minimax_h3_compare.py'), doraise=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lm', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prepare(args.lm, args.output)
