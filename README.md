目前实现了 ingest（原始数据读取与抽取），clean（基础清洗与规则过滤），，还有一个下载 CommonCrawl 的脚本 `download-cc.sh`。以及部分实现 quality（质量评估与过滤），dedup（去重模块）。

整体目录结构说明
```
llm-data-pipeline/
├─ configs/                         # 各环境配置：输入/输出路径、Ray 并行度、阈值、step 开关、tokenizer 配置
├─ src/
│  └─ llm_data_pipeline/            # 项目主包（uv --package 生成；这里的代码都可被 uv run/安装引用）
│     ├─ pipeline/                  # 统一编排入口：from_step/to_step、checkpoint/resume、step 注册表、统一 IO/manifest
│     ├─ ingest/                    # 原始数据读取与抽取：WET/WARC/HTML -> 标准化记录（doc_id/url/text/metadata）
│     ├─ clean/                     # 基础清洗与规则过滤：normalize、语言识别、可解释的 rule-based filter
│     ├─ dedup/                     # 去重模块：exact dedup + MinHash/LSH near-dup；输出可追溯的 dup 标记
│     ├─ pii/                       # 隐私过滤：PII 检测/脱敏策略；输出命中类型与计数，便于审计与扩展
│     ├─ quality/                   # 质量评估与过滤：质量特征、质量分、阈值策略；形成高质量训练语料
│     ├─ export/                    # 导出模块：写 shards、生成 dataset_manifest、保证 schema/版本/可复现
│     ├─ tokenizer/                 # Tokenizer 相关：语料导出、训练、benchmark（vs tiktoken）、demo/分析脚本
│     └─ tools/                     # 通用工具：sanity check、metrics、funnel、compare runs、repro/manifest 校验等
├─ docs/                            # 面向读者的文档：runbook、架构、取舍、消融、简历 bullet（不放大数据）
├─ outputs/                         # 管线产物与 checkpoint：按 sample/dev/stage/prod 分层，gitignore 掉
├─ runs/                            # 每次跑的指标快照（默认不进 git）：用于 before/after 对比与性能回归，gitignore 掉
└─ data/                            # 数据“指针”目录：只放软链接/路径说明/小样索引；raw 数据实际放项目外，gitignore 掉

```