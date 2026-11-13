# 嵩山杯 MiniMax-H3 FL2VA 源码复现

本目录提供从固定提交构建 InfiniCore 和 InfiniLM、重建九齿 AOT 宿主库、生成两路视频并执行官方比较的入口。平台原报告使用的是已构建的部署包，本入口的八卡端到端重跑仍待完成。

## 环境

- Linux x86_64，8 张可见的海光 gfx936 DCU，配套 DTK 26.04 和 RCCL。
- Python 3.11 环境，平台验证版本为 `torch==2.9.0+das.opt1.dtk2604`，`torchvision==0.24.0+cpu`。先使用平台提供的这两个包，再安装本目录依赖。xmake 使用 3.0.5，并需要 Git、GCC、Python 开发头及访问 GitHub、xmake 包源。
- 使用赛事提供的 MiniMax-H3 Diffusers 权重目录，内含 `model_index.json`。平台共享位置可作为 `MODEL` 的输入，脚本接受其他可读的等价目录。
- 本目录包含模板统一首帧。SHA-256 为 `b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54`。

## 运行

进入包含本说明的目录。在评审机器上激活上述 Python 环境并执行：

```bash
python -m pip install -r requirements-fl2va.txt
export DTK_HOME=/opt/dtk
MODEL=/path/to/MiniMax-H3/diffusers
bash reproduce_fl2va.sh "$MODEL" "$PWD/fl2va_reproduction"
```

依赖安装可使用平台镜像或离线 wheel 源。脚本把构建与输出写入一个新的工作目录。失败后请保留该目录的日志，并以新的目录重新执行。

脚本先检出固定的 InfiniCore 和 InfiniLM 提交，拉取所需子模块并构建动态库。它使用本目录的 gfx936 AOT 生成源码重编六个宿主库，并生成新的九齿回归结果及框架验证记录。原始设备代码随生成的 C++ 源码保留。重新生成设备代码所用的九齿及 FlagTree 编译器代码见以下提交：

- [NineToothed](https://github.com/zhangkun123424/NineToothed-minimax-h3/tree/ea3e98701800df8ce1ea7219e9f64b514268c628)
- [FlagTree](https://github.com/zhangkun123424/FlagTree-minimax-h3/tree/e8c350f2daf7125f6988682bb524d9319ff2fbc8)

随后脚本生成 PyTorch 参考视频和 InfiniLM 视频，统一使用首帧、提示词 `A dog falls asleep`、124 帧、50 步和种子 42。官方比较脚本输出帧数、尺寸及平均 SSIM。两路生成日志和 MP4 位于 `results/`，全流程日志位于 `reproduction.log`。

平台原验证结果为 768×928、124 帧、平均 SSIM 95.936%，InfiniLM 端到端 731.006 秒。以上数字来自原部署，应以本次复现日志为准。
