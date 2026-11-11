# MiniMax-H3 FL2VA · Songshan Cup preliminary round

Base: InfiniTensor/InfiniLM `minimax_h3` at `65deb91f840ebaef752d028fad2f3217ce0950f4`. `songshan-source.patch` records the xmake dependency pin in the verified source tree. Apply it at the repository root with `git apply songshan-source.patch`.

Place the Python entry points from this branch in `$IFAB_AFS_HOME/scripts`. The verified run used the InfiniFabric gfx936 eight-device environment, its shared MiniMax-H3 Diffusers model, a Python 3.11 FL2VA environment, the built InfiniCore/InfiniLM deployment, the FP32 chunked MHA plugin and the repaired NineToothed AOT package. The companion repositories are InfiniCore-minimax-h3 and NineToothed-minimax-h3 under this GitHub account.

```bash
"$IFAB_AFS_HOME/venvs/fl2va/bin/python" "$IFAB_AFS_HOME/scripts/run_fl2va_fixed_v6.py" \
  --first-frame "$IFAB_AFS_HOME/imports/fl2va_report_first_frame.png" \
  --text "A dog falls asleep" --gpus 0,1,2,3,4,5,6,7 \
  --num-frames 124 --steps 50 --seed 42 --ninetoothed \
  --output "$IFAB_AFS_HOME/results/fl2va_native_standard50_nt_sync_fp32_v1.mp4"
```

The official comparison entry point is `minimax_h3_compare_official.py`. The original wrapper and dependency checks refer to the platform deployment archives and local validation files. The clean-checkout deployment has yet to be rerun.
