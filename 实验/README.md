# 最新图7–图9运行说明

Python 3.12，安装`latest/fig8a/requirements.txt`；安装SUMO 1.27.1。

在PyCharm打开本目录，运行`run_latest.py`；或从仓库根目录执行：

```text
python 实验/run_latest.py --sumo-home "C:\Program Files (x86)\Eclipse\Sumo"
```

该入口在结果目录新建时间戳运行，先执行图7五场景，再实际运行SUMO五方法实验，从该新运行冻结参考快照，重算距离/速度两组五方法敏感性，最后更新主图原位置。不会以旧CSV冒充新实验。仅刷新已经算完的最新图片可运行`python 实验/scripts/export_figures.py`，这不是重新仿真。

- `latest/fig7/`：图7实际使用的脚本、原模型与冻结快照；5种配置见结果中的scenario_settings.csv。
- `latest/fig8a/`：五种策略、完整动态仿真、共同评价器、公式、统计及核查代码。
- `latest/fig8bc/`：同五方法、同物理模型和求解器的距离/闭合速度受控扫描；prepare_snapshot.py从已完成(a)提取参考，run_sensitivity.py实际重算510次AO。
- `../结果/fig7_latest/`、`../结果/fig8a_five_methods/`、`../结果/fig8bc_five_methods/`：已完成结果的完整可复现包。
- `../图片/`：独立图片，每张仅保留一个正式文件名的PNG/PDF/SVG。

当前编号按本次任务：内层P1为目标级驻留，外层P2为系统Ts/Ps。历史论文编号可能相反，原函数名保持不变。

`main.py`、`model.py`、`legacy/`、`sumo/`仍是原物理模型参考；直接运行旧主程序只生成旧版诊断，不能代替latest入口。Full ML/CNN旧流程见[历史运行说明](reference/README_20261019.md)。

图8(b)(c)当前为五方法总wCRB对相对距离/径向闭合速度的受控敏感性曲线；此前四目标距离/速度时域图留在结果/trajectory_figures作辅助资料。旧target_distance/radial_velocity等重复文件已删除；当前正式文件为fig8b_five_methods和fig8c_five_methods。图8(a)是五条真实策略曲线，不含跨目标标准差或未实算的重复实验阴影。

导出入口不再生成重复别名；相同文件按SHA-256去重，旧路径通过`结果/deduplication.json`与`scripts/artifact_paths.py`定位唯一副本。独立重跑所需冻结输入会复制到新运行目录。旧的source_artifact_manifest.json及delivery_manifest.json记录当时版本，不作为当前文件清单；当前清单为upload_manifest.json及各实验包artifact_manifest.json。
