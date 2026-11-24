# 品牌商品的可解释归类与混装承诺复查

`资料/演练.json` 保存本地合成的业务记录，不含个人信息及实际企业数据。各资料集的字段、单位和有效范围见[资料格式](资料格式.md)。需要 Python 3.11.2，仅使用标准库，从项目根目录运行，支持隔离模式 `python3 -I`。

## 这是什么

围绕“苹果商品归类 + 组合级采购承诺”的可持续应用，由七个职责单一的模块组成（`apple_review/`）：

| 模块 | 职责 |
| --- | --- |
| `evidence.py` | 资料读取、引用完整性校验、版本化证据（配料事实/文案/裁决）、防泄漏连通组 |
| `taxonomy.py` | 类目版次、叶类/父类、跨版兼容映射、标签版本语义 |
| `classifier.py` | 可解释规则参照 `RuleReference` 与带弃权的朴素贝叶斯 `NaiveBayesClassifier` |
| `evaluation.py` | 按历史截止点的分组切分、回测、稀有类/换版/未知类三个可靠度视图 |
| `commitment.py` | 组件不确定性向组合级承诺传播、场景枚举、三类处置 |
| `workload.py` | 人工复核范围锁定、共享事实排序、容量约束、回退不重派 |
| `reporting.py` | 运营视图（触发证据链）与供应商视图（适用标签/可供货版本） |

## 关键业务规则如何落地

- **“礼盒”不是品类证据**：包装/营销词（礼盒、甄选、手作等）进 `NON_EVIDENCE_TOKENS`，不参与判定。
- **标签属于版次**：v1 父类标签（fruit/processed）在 v2 下只能经 `compatible` 圈定候选集，不能当子类真值；这类 SKU 在 v2 显式弃权（`ABSTAIN_OLD_PARENT`）。
- **未知不硬塞叶类**：人工 `label=null`、配料缺失且文案无品类词、跨源互斥冲突，一律弃权并给出理由码。
- **模型受规则护栏约束**：规则弃权时模型不得用弱后验硬判；模型与规则互斥时转人工；冷启动（单类/零类训练）显式回退规则参照并标注 `model_fallback`。
- **防泄漏分组**：同 `family` 或在同一 package 共同出现的 SKU 构成连通组，回测时跨截止点的整组剔除，同一商品的改标题（多 revision）与组合子件不会被拆进训练和考核。
- **承诺是组合级**：以“买家×组合”建单，共享组件在不同组合里的出现不算同一客单；复查不改写承诺原文。
- **两层标签传播**：`human_label`（同版人工确认）与 `inferred_label`（证据/模型推断）分开，占比同时给人工确认口径与推断口径。
- **三类处置**：`pause_new`（暂停新增预订）、`continue_old`（按旧约继续交付）、`notify_buyer`（运营明确通知采购方）；暂停与通知可并存（`notify_required`）。
- **人工复核纪律**：先锁定范围再排序；误分类代价优先，高代价小品类有 `reserve_high` 保底；已派出工作不并发重派，未派出且被新裁决覆盖的同证据工作回退而不是重复派发。

## 命令行用法

```sh
# 单条/全量归类（人工裁决优先；--method rule 看可解释参照，默认 model 看选用做法）
python3 -I apple_cli.py classify --sku S006
python3 -I apple_cli.py classify --sku S001 --taxonomy v1
python3 -I apple_cli.py classify --sku S200 --as-of 2026-11-01

# 分组防泄漏回测与三个可靠度视图（稀有类、换版前后、未知类）
python3 -I apple_cli.py backtest

# 在途采购单复查：输出三类处置、触发证据链、分场景占比
python3 -I apple_cli.py review-orders
python3 -I apple_cli.py trace --package P3

# 人工复核范围锁定与排班（含回退/对账/容量/护栏）
python3 -I apple_cli.py work-plan

# 供应商视图：只看自己的适用标签与可供货文案版本
python3 -I apple_cli.py supplier --merchant M4

# 任一命令加 --json（子命令前后均可）输出机器可读 JSON
python3 -I apple_cli.py review-orders --json
```

上新、类目切换、在途单复查都直接读同一资料包结构；把新 SKU/文案/裁决并入 JSON 后重新运行即可，时间点复查用 `--as-of`。

## 测试

```sh
python3 -m compileall -q apple_review apple_cli.py run_tests.py tests
python3 -I run_tests.py          # 隔离模式下运行全部 42 个测试
python3 -m unittest discover -s tests   # 常规模式同样通过
```

测试覆盖：证据缺失语义、防泄漏分组、版次标签语义、规则/模型护栏（含未知硬判为零）、跨边整组剔除、承诺三档处置、原承诺不被改写、共享组件不并单、复核回退不重派与容量约束。

## 原入口保留

```sh
python3 inspect_packet.py    # 仅列出资料包记录规模，不产生业务判定
```
