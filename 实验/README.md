# 车载 ISAC 图7至图9复现与维护回波候选搜索

本工程保留旧模型作为单独基准，按 Gate 0–4 逐级验证。`legacy/` 中的 Python 和 SUMO 文件是原工程的逐字节快照；原工程、历史 CSV 和原图不修改。

## 当前问题编号

对外使用新论文编号：**P1 为系统感知总时间与功率优化，P2 为目标级驻留时间分配**。
旧函数 `solve_p1` 对应论文 P2，`alternating_optimize_system` 对应论文 P1。保留旧函数名避免意外改变数值。

## PyCharm 运行

1. 在 PyCharm 中打开本目录，选择 Python 3.12，安装 NumPy、SciPy、Matplotlib；需要新ML研究时再安装CPU版PyTorch。
2. 安装 SUMO 1.27.1，将 `SUMO_HOME` 设置到该版本安装目录。TraCI 从匹配版本的 `tools` 读取。
3. 按顺序直接运行下列脚本。默认参数适合 PyCharm 的 Run 按钮；输出路径均相对于工程目录解析。

```text
python scripts/reproduce_legacy.py --stage all
python scripts/export_figures.py
python scripts/validate_echo.py
python scripts/run_full_ml.py --stage all
python scripts/compare_methods.py --stage data
python scripts/compare_methods.py --stage train
python scripts/build_report.py
```

依赖示例：

```text
python -m pip install numpy==2.5.4 scipy==1.18.1 matplotlib==3.11.2 PyMuPDF==1.28.2
python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

本次工作环境将依赖安装在 `.deps/` 中；脚本在该目录存在时自动读取。在自己的虚拟环境中按上面的命令安装即可。

## 结果位置

- `../图片/`：重新计算后的图7、图8(a)、图8(b)、图8(c)、图9，共五张，均提供PNG、PDF和SVG，以及对应源数据。
- `../结果/legacy_reproduction/`：环境版本、401周期交通复现、逐点偏差检查；`rerun/` 为原程序完整重新运行的结果，包含图8组合图。
- `../结果/full_ml/`：维护回波参数、采样与FIM检查、三个快照和整个时域的ML估计、似然图、原始复数观测和耗时。
- `../结果/dnn_ml/`：独立场景、训练验证测试划分、CNN权重、训练记录、阈值选择和四种方法的公平测试。
- `实验报告.md`：中文结果、模型差异、全部检查与是否允许后续集成的结论。

## 模型边界

旧图使用原来的未知复幅度投影后的对角EFIM和SUMO真值状态，不与新版完整FIM混称。
新增估计实验是独立的维护回波影子实验：已知RCS和相位、距离相关幅度、200 MHz LFM、250 MHz原始波形表示，经粗距离去斜和理想滤波后的2 MHz低维观测。输出完整候选网格上的保留观测似然，不声称它保留被丢弃原始ADC样本的全部信息。维护观测的时间和能量不计入旧图预算，尚不构成实时系统集成。

CNN仅预测多个候选区域。最终DNN-Adaptive-ML估计必须通过与Full-Grid ML相同的物理似然选择；教师结果只用于离线标签和事后评分。训练、验证、测试使用不同SUMO场景和不同噪声种子；阈值仅在验证集选定。未达到Gate4全部条件时，不将其接入系统资源分配，也不声称DNN带来性能改善。

图8(a)的阴影为四目标贡献的标准差，不是重复试验置信区间。所有曲线使用原始数值，没有拟合、缩放或平滑。数值相同不要求不同Matplotlib版本输出字节相同的PNG。


本仓库中脚本位于本目录的`scripts/`；从仓库根目录执行时在路径前加`实验/`。参考CSV与源代码哈希已经随工程提供，不依赖原电脑上的历史目录。上级`结果/`保存所有运行输出，上级`图片/`保存五张主图。
