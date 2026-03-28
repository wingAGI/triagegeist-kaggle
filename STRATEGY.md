# Triagegeist 冲奖方案

## 目标

- 提交一个不只是“预测 `triage_acuity`”的 notebook。
- 把作品包装成一个面向急诊分诊的辅助决策与质控工具。
- 争取在 `Clinical Relevance`、`Technical Quality`、`Insight and Findings` 三项上形成明显优势。

## 核心定位

### 我们不做什么

- 不把项目写成普通的多分类 baseline。
- 不把主要精力花在极限刷分或堆复杂模型上。
- 不把重点放在华丽 UI 或不易复现的外部服务上。

### 我们要做什么

- 预测分诊等级 `ESI 1-5`
- 识别高风险漏分诊样本
- 给出可解释的高风险因素
- 检查不同人群与站点上的性能差异
- 用病例级分析说明模型怎样帮助临床分诊

一句话版本：

`Acuity prediction + undertriage risk flagging + subgroup audit + interpretable error analysis`

## 为什么这条路最适合这题

- 这是评委制 hackathon，不是纯 leaderboard 赛。
- 当前公开参赛队伍数不高，完成度和叙事质量会比极小分差更重要。
- 数据里结构化变量和标签关系很强，强 baseline 并不难做。
- 官方样例已经给了 “LightGBM + chief complaint NLP” 方向，继续走同质化路线很难显得新。

## 数据判断

## 已确认事实

- `train.csv` 约 `80,000` 行，`test.csv` 约 `20,000` 行。
- 还有 `chief_complaints.csv` 和 `patient_history.csv` 可按 `patient_id` 拼接。
- `disposition` 和 `ed_los_hours` 只在训练集出现，属于潜在泄漏列，建模时应删除。
- 目标分布不均衡，但不极端。
- `news2_score`、`gcs_total`、`spo2`、`respiratory_rate`、`pain_score` 与分诊等级高度相关。
- 数据是 synthetic，主办方更看重临床价值表达与方法完整性。

## 已知风险

- synthetic 数据可能让模型看起来“过于容易”，单纯报高分说服力不够。
- complaint text 带模板味，单靠 NLP 新意有限。
- 如果只做分类，不做校准、病例分析和偏差审计，容易像课程作业。

## 提交主题

## 推荐标题

`Beyond Acuity Prediction: An Interpretable Triage Support Pipeline for Undertriage Detection`

## 推荐副标题

`Structured vitals, complaint text, and patient history for emergency triage decision support and bias auditing`

## Notebook 结构

1. 临床问题定义
2. 数据说明与限制
3. 任务设计
4. 特征工程
5. 模型与验证策略
6. 整体结果
7. 高风险漏分诊分析
8. 可解释性分析
9. 亚组公平性与稳健性
10. 病例级案例研究
11. 局限性与临床落地边界
12. 可复现性说明

## 建模路线

## 任务定义

### 主任务

- 五分类预测 `triage_acuity`

### 辅任务

- 二分类识别高风险患者
  - 建议定义 A：`triage_acuity <= 2`
  - 建议定义 B：模型预测等级高于真实等级至少 `2` 级时记为潜在 undertriage 风险
- 亚组性能评估
  - `age_group`
  - `sex`
  - `language`
  - `site_id`
  - `arrival_mode`

## 数据拼接

- 主表：`train.csv` / `test.csv`
- 文本表：`chief_complaints.csv`
- 病史表：`patient_history.csv`

拼接后保留：

- 基础人口学
- 分诊上下文
- 生理指标
- complaint 原文
- 既往病史 flags

删除或只用于分析、不用于训练：

- `patient_id`
- `disposition`
- `ed_los_hours`

## 特征工程

### 结构化特征

- 数值列直接输入树模型
- 缺失值保留并加缺失指示
- `pain_score == -1` 单独视为“未记录”
- 针对 vitals 构造少量临床派生特征
  - 是否低氧
  - 是否发热
  - 是否心动过速
  - 是否低血压
  - 是否意识异常

### 文本特征

- complaint 原文做轻量 TF-IDF baseline
- 再加少量 clinically informed keyword flags
  - chest pain
  - stroke / seizure
  - shortness of breath
  - trauma
  - overdose
  - bleeding
  - pregnancy-related

### 病史特征

- 直接拼接二值病史
- 再增加简单聚合
  - cardio burden
  - respiratory burden
  - neuro burden
  - immunocompromised flag

## 模型顺序

### 第 1 层：快速强基线

- RandomForest / ExtraTrees
- HistGradientBoosting
- 有条件时补 LightGBM / XGBoost / CatBoost

目标：

- 先拿到稳定的交叉验证结果
- 找出最重要特征
- 验证文本与病史是否有增益

### 第 2 层：冲奖模型

- 结构化主模型
- 文本辅助模型
- 概率融合或简单 stacking

这里重点不在复杂，而在：

- 结果稳定
- 解释清楚
- 消融完整

## 验证与指标

## 交叉验证

- `StratifiedKFold`
- 做两版验证
  - 普通分层 CV
  - 按 `site_id` 或 `triage_nurse_id` 的稳健性切分

## 报告指标

- Macro-F1
- 每一类的 precision / recall / F1
- 混淆矩阵
- 高风险患者召回率
- 校准表现

## 必做分析

### 1. 泄漏检查

- 明确说明不使用 `disposition`、`ed_los_hours`
- 说明为什么这两列只能作为事后结果，不可用于分诊时点预测

### 2. 消融实验

- 仅结构化特征
- 结构化 + 病史
- 结构化 + 文本
- 结构化 + 病史 + 文本
- 加或不加临床规则特征

### 3. 错误分析

- 哪些 ESI 相邻等级最容易混淆
- 哪类 complaint 或 arrival mode 最容易错
- 低氧、意识改变、休克指标异常患者中是否仍有错分

### 4. Undertriage 分析

- 找到真实高危但被模型打低的病例
- 解释这些错误代表什么风险
- 反过来识别模型可用作“第二读者”的场景

### 5. 亚组偏差分析

- 不同 `language`、`age_group`、`site_id` 上的 Macro-F1
- 比较高危召回在各亚组是否一致
- 写清楚 synthetic 数据下结论边界

### 6. 可解释性

- permutation importance
- SHAP 或替代解释方法
- 重点展示高危预测时哪些信号主导了模型

## 写作策略

## 评委最容易买单的叙事

- 目标不是取代护士，而是做第二层安全网
- 重点不是“分类准确率很高”，而是“减少危险漏分诊”
- 结果不是“模型万能”，而是“在哪些临床场景有帮助，哪些场景仍需人工主导”

## 一定要主动写出的限制

- synthetic data，外部有效性有限
- 真实医院部署前需要前瞻性验证
- 文本输入和站点流程差异会影响泛化
- 模型输出只能作为 decision support，不能单独替代临床判断

## 推荐交付节奏

### 第 1 阶段：打底

- 解压并整理数据
- 完成拼表与特征清单
- 跑无泄漏 baseline
- 确定主模型

### 第 2 阶段：形成亮点

- 加文本特征和病史特征
- 做消融
- 做高风险召回与 undertriage 分析
- 做 subgroup audit

### 第 3 阶段：形成作品感

- 补病例分析
- 补可解释性图
- 整理 notebook narrative
- 写 Kaggle writeup
- 准备 cover image

## 具体执行清单

1. 建一个数据准备脚本，产出拼接后的 train/test 表
2. 建一个 baseline notebook 或脚本，先输出 CV 结果
3. 确认泄漏列和禁止使用的结果列
4. 做结构化模型 ablation
5. 做 complaint 文本特征实验
6. 做 high-risk recall 和 undertriage 风险分析
7. 做 subgroup audit
8. 整理最终 notebook 与 writeup

## 判断标准

如果最后作品满足下面四条，就有真实冲奖资格：

- baseline 指标稳定且无明显泄漏
- notebook 不只报分，还能讲清楚临床价值
- 有病例分析与 subgroup audit
- writeup 看起来像研究原型，而不是练手作业
