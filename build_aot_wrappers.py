"""Compile the archived gfx936 AOT source with the validated HIP cache fix."""
import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNELS = {
    'aot_d256_fp32': 'nt_timestep_aot_d256_fp32',
    **{f'aot_generic_d{d}_{t}': f'nt_timestep_generic_d{d}_{t}'
       for d in (16, 256) for t in ('bf16', 'fp16', 'fp32') if (d, t) != (256, 'fp32')},
}
OLD = ('hipCtx_t current_ctx; hipError_t ctx_status = hipCtxGetCurrent(&current_ctx); '
       'if (ctx_status != hipSuccess) return (NineToothedResult)ctx_status; '
       'ctx_id = (unsigned long long)(uintptr_t)current_ctx;')
NEW = ('int current_device = -1; hipError_t device_status = hipGetDevice(&current_device); '
       'if (device_status != hipSuccess) return (NineToothedResult)device_status; '
       'ctx_id = (unsigned long long)current_device;')


def build(output, dtk, original=False):
    for folder, name in KERNELS.items():
        target = output / folder
        target.mkdir(parents=True, exist_ok=True)
        sources = sorted(HERE.glob(name + '.*.cpp'))
        if len(sources) != 2:
            raise RuntimeError(f'{name} requires i32 and i64 AOT variants')
        shutil.copyfile(HERE / 'ninetoothed.h', target / 'ninetoothed.h')
        files = [target / (name + '.cpp')]
        shutil.copyfile(HERE / files[0].name, files[0])
        for source in sources:
            text = source.read_text()
            if text.count(OLD) != 1 or text.count('static ninetoothed::ThreadSafeUnorderedMap') != 1:
                raise RuntimeError(f'Unexpected AOT cache implementation in {source.name}')
            if not original:
                text = text.replace(OLD, NEW).replace(
                    'static ninetoothed::ThreadSafeUnorderedMap',
                    'static thread_local ninetoothed::ThreadSafeUnorderedMap',
                )
            path = target / source.name
            path.write_text(text)
            files.append(path)
        library = target / (name + '.so')
        temporary = library.with_suffix('.tmp.so')
        subprocess.run([
            'g++', '-std=c++17', '-shared', '-fPIC', '-O2', '-D__HIP_PLATFORM_AMD__',
            '-I' + str(dtk / 'include'), '-I' + str(dtk / 'hip/include'),
            *map(str, files), '-L' + str(dtk / 'hip/lib'), '-lgalaxyhip',
            '-o', str(temporary),
        ], check=True)
        temporary.replace(library)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dtk-root', type=Path, default=Path('/opt/dtk'))
    parser.add_argument('--original', action='store_true')
    args = parser.parse_args()
    build(args.output, args.dtk_root, args.original)
