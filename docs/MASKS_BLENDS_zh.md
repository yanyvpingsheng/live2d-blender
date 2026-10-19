# v0.3 原生遮罩与混合

这条路径已经接入插件的 **Convert to independent bones + native effects** 操作。整条 Convert 仍会近似拟合几何。在已拟合的独立骨骼结果上，遮罩/混合阶段保留原颜色贴图、UV 和所有 Drawable，不再改动几何、权重或骨骼 Action；只增加普通 Blender 材质节点与纯灰度遮罩序列。不会用预渲染角色 RGB 视频替换模型。输出播放不需要插件、Core、Python handler 或 driver。

## 在界面中使用

1. 使用自己有权转换的模型。导入时若模型有遮罩/非普通混合，勾选允许有损的实时预览；这仅控制源模型预览，之后独立转换仍可选择新遮罩路径。
2. 先保存 `.blend`。把 Render Engine 设为 **Cycles**；设好源模型参数关键帧、采样起止时间和骨预算。
3. 保留 **Bake native opacity / tint / layer effects**，Appearance 选择 **Native masks + blends (Cycles approximation)**。
4. 若模型使用 legacy multiply，关闭 Film → Transparent，并勾选 **Accept opaque-backdrop-only multiply**。这表示接受此限制；并非通用透明导出支持。插件不会自动改变你的渲染引擎或背景。
5. 选择 Mask resolution per group（默认 512）和 Mask output directory（默认 `//live2d_masks`）。目录下每次建立独立子目录，不覆盖旧转换。转换前检查磁盘空间；高分辨率、长片段和大量遮罩组会增加时间与空间。当前上限为 2,000 整数帧、1,073,741,824 输出灰度像素、20,000,000 相关顶点帧样本；超限请缩短范围或降低分辨率。
6. 执行 Convert，然后保存 `.blend`。检查骨架 `conversion_report` 中的 `alpha_masks`、`native_effects`、拟合误差与外观限制。
7. 在 Cycles 的 Light Paths → Transparent 中增加透明反弹上限，比较两次更高上限的渲染直到画面稳定。默认上限对多层角色可能太低，会出现黑块；插件只记录上限，不声称自动校准成功。

默认 **Basic / strict** 保留严格行为：遮罩/非普通混合会拒绝。仅明确勾选 Drop masks / non-normal blends 时，才使用旧的有损普通 alpha 转换。新模式遇到不支持的 blend 类型会直接拒绝，不会退回普通 alpha。

## 遮罩语义

- 对每个不同的源 Drawable 集合生成一组 8-bit 灰度 PNG；相同组的目标共享序列。
- 每个整数帧（包含首尾）从输出的 Blender Armature/Action 求值源几何。遮罩来自原贴图 alpha 与原 UV，而不是 Core RGB 渲染。
- 原始遮罩忽略源的 opacity、visibility、tint、blend 和源自身的 clipping。多个源先按 `1 - product(1 - alpha)` 合并，再在目标材质上反转。
- 目标有效 alpha 只应用一次：贴图 alpha × 目标 opacity × visibility × 遮罩 coverage。
- 投影在输出根对象的局部 XY 空间，裁剪范围覆盖该组的源和目标整个片段并加留边。整体移动、旋转、缩放根对象不会把遮罩留在世界原点。
- 新的骨骼动作、改动后的输出网格/UV、局部对象变换、延长帧范围需要重新转换/烘焙；现有序列不会在播放时重新生成。子帧使用离散图像序列，运动模糊与整数帧之间的边缘不保证一致。

## 混合语义及边界

- Normal：普通透明 / emission alpha 混合。
- Legacy Add：未染色 Transparent + 乘以有效 alpha 的 Emission。黑色加法 texel 不再遮挡背景，目标背景 alpha 保留。
- Legacy Multiply：Transparent 的 RGB 为 `(1 - alpha) + source_rgb * alpha`。在完全不透明背景上工作；半透明背景的最终 alpha 不是 legacy SDK 公式，必须显式接受不透明背景限制。
- 每 Drawable 的 multiply/screen tint 保留原有 sRGB 编码空间数学，再转线性 emission。跨 Drawable 的 Cycles 混合是场景线性，且没有每层 UNORM 饱和截断。因此这不是任意 SDK 图像的逐像素一致实现。
- 有限分辨率、8-bit alpha/预乘颜色量化、PNG 过滤、拟合误差、透明射线深度、不同渲染色彩管理都会影响结果。遮罩边缘不保证等同 SDK atlas/MSAA。
- 只处理已采样固定拓扑模型的 legacy 0/1/2 blend。新式 offscreen/group blending、任意新参数组合、通用透明背景导出与跨渲染引擎一致性没有保证。

## 迁移与独立播放

输出颜色贴图会复制并打包到 `.blend`。灰度图像序列是**外部文件**，不是可保证被 Blender Pack Resources 打包的内容。

必须把 `.blend` 与生成的遮罩子目录一起复制，保持相对位置。默认结构为：

```
character.blend
live2d_masks/
  native_alpha_masks_<unique>/
    mask_0000/
      000001.png
      ...
```

实际子目录/文件名以 `conversion_report` 为准。若另存到不同目录，检查 Blender 外部文件路径并同时迁移遮罩文件夹。只发送 `.blend` 会缺失遮罩。开始/结束帧、场景 FPS 和骨骼动作要与转换片段一致。

可在全新 Blender 进程以 `--factory-startup --disable-autoexec` 打开迁移后的文件，不加载插件与 SDK，检查首帧、中间帧、末帧，以及 blink/变化边缘帧。仓库原创集成测试覆盖这条路径；它不替代你的模型的目视检查。
