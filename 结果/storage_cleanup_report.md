# 2026-10-30：图8标签与重复文件清理

图8(a)(b)(c)横坐标分别为`t(s)`、`d(m)`、`v(m/s)`，纵坐标统一`ωCRB`；独立图和三联图同步更新。只从此前实际实验数据重绘，未重新宣称执行仿真，物理模型、求解器和全部数值未更改。

按SHA-256检查，删除120个完全相同的图片/数值数据副本，另去重11个布局元数据副本，图片目录不再存放CSV或重复布局文件。不同图片格式和三联图属于必要独立导出，全部保留。只删除重复内容，不删除具有不同数值的历史研究结果。旧版可从Git历史追溯。

116个原CSV路径均通过唯一保留副本解析，SHA-256与清理前完全一致。所有删除路径均有保留位置；冻结输入可复制到新运行目录，已检查。详见[去重索引](deduplication.json)、[验收记录](figure_qa/labels_and_deduplication_review.json)、[正式图片与数据索引](../图片/README.md)。

当前文件清单是upload_manifest.json和各实验包artifact_manifest.json。source_artifact_manifest.json与delivery_manifest.json属于历史原始版本凭据，其旧路径可经去重索引解析，不作为最新布局的校验清单。
