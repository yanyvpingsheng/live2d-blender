# Live2D Offline Converter for Blender

Blender 用 Live2D 离线转换研究原型，版本 **0.3.0**。

读取用户自行取得并有权使用的模型，在指定动作片段上采样并拟合为 Blender 普通骨骼、顶点权重及 Action。转换结果可在不加载本插件或 Live2D Core 的情况下播放；它是近似的已烘焙动作，不是可任意操作原 Live2D 参数的完整角色。

## 当前状态

- 已测试：**Linux x86_64、Blender 4.3.2**。
- 声明的最低 API 版本为 Blender 4.2；4.2、其他 Blender 版本、Windows 和 macOS 尚未完成验证。
- 支持全局运动聚类、逐 Drawable 分组、用户手工 JSON 分组、训练姿态内的退化 bind 修复、独立静态影响及矩阵关键帧加密。
- 导出固定拓扑近似几何、每顶点至多 4 个非负归一权重，以及原生透明度、乘色/屏幕色、可见状态、图层深度和根对象 TRS 动画。每个逻辑影响对应两个 Blender 骨骼。
- **v0.3 已把通用遮罩/混合修复接入 Convert**：逐整数帧从输出骨骼几何烘焙 alpha-only 遮罩序列；保留原颜色贴图，支持 union、inverted mask 和 legacy normal/add/multiply 材质。
- 新外观路径需明确选择 **Native masks + blends (Cycles approximation)**。Multiply 还需接受不透明背景限制；跨层场景线性混合、有限分辨率与每层截断差异仍存在。默认 Basic / strict 不会静默丢弃遮罩。见[使用和限制](docs/MASKS_BLENDS_zh.md)。
- 不承诺任意模型、长动作、未采样姿态、所有驱动/NLA 或跨平台保真。训练几何误差不等于最终渲染像素误差；报告默认不声称已验证保留帧。

## 安装与 SDK-free 演示

1. 下载仓库源码，在根目录运行 `python scripts/package.py`。
2. 在 Blender 的 Preferences → Add-ons → Install from Disk 中选择生成的 `dist/live2d_offline_converter-0.3.0.zip`，启用插件。GitHub 的源码 ZIP 不是可直接安装的插件 ZIP。
3. 如果安装过旧版，先保存工作并停用旧版；模块名相同，不能同时启用。
4. 在 3D View 侧栏打开 Live2D，选择 **Create SDK-free synthetic demo**。原创小芽夹具不需要任何 SDK。
5. 对模型根对象的 Live2D 参数插入关键帧，设置 Bake start/end、Sample step 和逻辑骨预算，再选择 **Convert to independent bones + native effects**。
6. 遮罩/混合模型：先保存 `.blend`、选择 Cycles，再在 Appearance 选新模式；将 `.blend` 与生成的相对路径遮罩目录一起保留。颜色贴图会复制并打包，遮罩序列为外部文件。
7. 输出骨架的 `conversion_report` 属性记录拟合模式、误差、骨预算、bind 修复、静态点和外观损失。请检查结果后再用于生产。

仓库不含预生成 `.blend` 或私有素材；`tests/blender_v02_install_convert.py` 可重新生成原创演示。安装与测试步骤见 [测试说明](docs/TESTING.md)，手工分组见 [分组说明](docs/GROUPING_zh.md)。

## 使用真实模型

- 自行从官方来源取得并授权安装 Cubism Native SDK，在插件 Preferences 中选择适配当前系统的官方 Core 动态库。
- 导入 `.model3.json` 清单；`.moc3` 由官方 Core 求值。本项目不解析或重建完整 Cubism 编辑器变形器树。
- 可选物理/Pose 需要以用户本地 SDK 构建本项目薄桥，见 [native/README.md](native/README.md)。原生桥只验证过 Linux。
- 加载动态库会执行本机原生代码，仅选择可信且获授权的库。本插件不会下载 SDK、代替用户接受其条款或提供原生代码沙箱。
- Basic 模式输出仍可能引用原贴图，迁移前使用 Pack Resources。新 Cycles 模式会打包输出颜色贴图，但必须另带完整遮罩序列目录；Pack Resources 不替代复制序列。遵守原素材许可。
- 子对象额外 Action、driver、NLA、非默认局部变换及不可表达的根 shear 会明确拒绝。复杂子帧曲线、任意动画叠加和新参数组合不保证保留。

## 源码与许可

本仓库包含项目原创插件、自写 C++ 桥接源码、原创合成夹具和测试，以现有 [MIT License](LICENSE) 提供。四张微型程序纹理是原创夹具。

**不包含** Cubism SDK/Core/Framework 的源码、头文件或二进制，也不包含官方示例模型、用户模型/贴图/采样数据、已编译桥、私有日志或转换资产。外部依赖和组合发行边界见 [THIRD_PARTY.md](THIRD_PARTY.md)。MIT 不授予第三方模型或 SDK 的使用、再分发权；发布组合产品前需自行核对适用许可。

提交前运行 `PYTHONDONTWRITEBYTECODE=1 python scripts/audit_public.py`。`PUBLIC_FILES.txt` 是公开源文件清单；测试和构建输出只应保留在本地。

## English summary

Experimental **v0.3.0** offline Live2D-to-Blender converter. It samples an authorized model and fits an ordinary Blender armature, vertex weights, bone/root Actions and selected material effects. The baked result can play without the add-on or Cubism Core. This is approximate clip conversion, not a general independent Live2D rig.

Tested on **Linux x86_64 / Blender 4.3.2 only**. The real Convert operator now offers alpha-only mask baking from exported bone animation and native legacy normal/add/multiply nodes. Select the explicit Cycles approximation mode; multiply requires an opaque backdrop. Scene-linear compositing, finite-resolution masks and missing per-layer clamping mean this is not general SDK pixel parity. Move the external mask folder with the saved blend. Windows/macOS and other Blender versions remain unverified.

Build the installable add-on with `python scripts/package.py`; install the resulting ZIP from `dist/`. The original synthetic demo needs no SDK. Real models require a separately installed, appropriately licensed official Cubism Core; optional physics uses the original bridge source in `native/`.

Original project code and fixtures retain the MIT license. No SDK, Core/Framework code or binaries, user assets, private samples or generated model data are distributed. See [testing](docs/TESTING.md) and [third-party boundaries](THIRD_PARTY.md).
