#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# != 2 ]]; then
    echo "Usage: bash $0 MODEL_DIRECTORY NEW_WORK_DIRECTORY" >&2
    exit 2
fi
model=$(realpath -- "$1")
work=$(realpath -m -- "$2")
test -f "$model/model_index.json"
test -f "$here/unified_first_frame.png"
(cd "$here" && echo "b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54  unified_first_frame.png" | sha256sum -c -)
python=$(command -v "${PYTHON:-python}")
dtk=${DTK_HOME:-/opt/dtk}
test -d "$dtk/hip/include"
for executable in git xmake g++; do command -v "$executable" >/dev/null; done
if [[ -e "$work" ]]; then
    echo "Choose a fresh work directory: $work" >&2
    exit 2
fi
mkdir -p "$work"/{imports,diagnostics,results}
exec > >(tee "$work/reproduction.log") 2>&1
export IFAB_AFS_HOME="$work" PYTHON="$python" DTK_HOME="$dtk"
export PATH="$(dirname "$python"):$PATH"
export INFINI_ROOT="$work/install"
torch_root=$("$python" - <<'PY'
from pathlib import Path
import sys
import torch
assert sys.version_info[:2] == (3, 11), sys.version
assert torch.cuda.device_count() == 8, "Expose exactly eight gfx936 devices"
assert all(
    getattr(torch.cuda.get_device_properties(i), 'gcnArchName', '').split(':')[0] == 'gfx936'
    for i in range(8)
), 'Expected eight gfx936 devices'
print(Path(torch.__file__).resolve().parent)
PY
)
export INFINILM_CXX11_ABI=$("$python" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$torch_root/lib:$dtk/lib:$dtk/lib64:$dtk/hip/lib:$dtk/cuda/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HYGON_ARCH=gfx936

checkout() {
    git init -q "$3"
    git -C "$3" fetch --depth=1 "$1" "$2"
    git -C "$3" checkout -q --detach FETCH_HEAD
    test "$(git -C "$3" rev-parse HEAD)" = "$2"
}
core="$work/InfiniCore"
lm="$work/InfiniLM"
checkout https://github.com/zhangkun123424/InfiniCore-minimax-h3.git 96905861a312f1c32902912c9e462383a8a7068d "$core"
checkout https://github.com/zhangkun123424/InfiniLM-minimax-h3.git 5319a79381fbcc12f56746e1e059ce1d2c8e5c3e "$lm"
git -C "$core" submodule update --init third_party/spdlog
git -C "$lm" submodule update --init third_party/spdlog third_party/json
git -C "$core" apply "$here/infini-core-gfx936.patch" "$here/from_torch_strides.patch"
git -C "$lm" apply "$lm/songshan-source.patch"

(
    cd "$core"
    xmake f -y -c -m release --hygon-dcu=y --hygon-arch=gfx936 --aten=y --ccl=y \
        --cxxflags="-D_GLIBCXX_USE_CXX11_ABI=$INFINILM_CXX11_ABI"
    for target in infinirt infiniop infiniccl infinicore_cpp_api _infinicore; do
        xmake build -y "$target"
        xmake install "$target"
    done
)
(
    cd "$lm"
    xmake f -y -c -m release --cxx11-abi="$INFINILM_CXX11_ABI"
    xmake build -y _infinilm
    xmake install _infinilm
)
for spec in "$core:infinicore" "$lm:infinilm"; do
    root=${spec%:*}
    name=${spec#*:}
    dest="$root/python/$name/lib"
    mkdir -p "$dest"
    if ! compgen -G "$dest/_$name*.so" >/dev/null; then
        extension=$(find "$root/python/$name" "$root/build" -type f -name "_$name*.so" -print -quit)
        test -n "$extension"
        cp "$extension" "$dest/"
    fi
done
export PYTHONPATH="$here:$core/python:$lm/python${PYTHONPATH:+:$PYTHONPATH}"
"$python" -c 'import torch, infinicore; from infinilm.multimodal import minimax_h3'
export SONGSHAN_MHA_PLUGIN="$work/imports/libmha_hygon_math_fp32_chunked_gfx936_v3.so"
"$python" "$here/build_mha_plugin.py" --core-root "$core" --core-lib-dir "$INFINI_ROOT/lib" \
    --dtk-root "$dtk" --output "$SONGSHAN_MHA_PLUGIN"
export INFINI_TIMESTEP_NT=1 INFINI_TIMESTEP_NT_ROOT="$work/aot" INFINI_TIMESTEP_NT_TRACE=1
"$python" "$here/build_aot_wrappers.py" --output "$INFINI_TIMESTEP_NT_ROOT" --dtk-root "$dtk"
"$python" "$here/check_nt_aot.py" "$INFINI_TIMESTEP_NT_ROOT" "$work/diagnostics/nt_multigpu_candidate_v1.json"
"$python" "$here/check_framework.py" "$work/diagnostics/operator_ninetoothed_report_gfx936.json"
"$python" "$here/prepare_generator.py" "$lm" "$work/generator"
"$python" "$work/generator/minimax_h3_compare.py" --tasks fl2va \
    --model-path "$model" --first-frame "$here/unified_first_frame.png" \
    --legacy-gpus 0,1,2,3,4,5,6,7 --infinilm-gpus 0,1,2,3,4,5,6,7 \
    --output-dir "$work/results"
