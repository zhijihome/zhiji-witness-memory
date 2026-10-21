# AML_OFFICIAL_CONTRACT_R2

FROZEN BY: AML Control Tower
SNAPSHOT DATE: 2026-09-21
STATUS: ACTIVE
SUPERSEDES: `00_contract/AML_OFFICIAL_CONTRACT_R1.md`
RULE SNAPSHOT: `07_docs/competition/AML_CYCLE2_RULE_FREEZE_R2_20260921.md`

本文件只登记当前官方页面可核的 Cycle 2 事实。历史 R1 保留作 provenance，不再作为当前规则权威。

## 0. Fresh official sources

- https://agentmemoryleaderboard.ai/competition/ — fresh verified 2026-09-21
- https://agentmemoryleaderboard.ai/home — fresh verified 2026-09-21
- AML-20 Final Rule Lock (author's internal work order AML20-20260920-FINAL-RULE-LOCK-R1)

## 1. Cycle 2 status and dates

```text
CYCLE2_STATUS        = OPEN
OPENED_AT            = 2026-09-20 00:00 UTC+8
MATERIALS_DEADLINE   = 2026-10-31 23:59 UTC+8
EVALUATION_CLOSES    = 2026-11-04 23:59 UTC+8
OFFICIAL_LEADERBOARD = MID_NOVEMBER_2026
```

报名/评测入口已实际开放，不再使用 EXPECTED/PENDING 口径。

## 2. Tracks and divisions

- Evaluation tracks: **Textual / Coding / Multimodal**。
- Divisions: **Open-source Methods / Commercial Products**。
- 两类参赛者都必须自行部署稳定的 Add / Search API。
- Open-source Methods 需要公开、可核验的仓库/固定版本及来源与改动披露。
- Commercial Products 可闭源，但需固定、可核验的产品/API 版本。

## 3. Add contract

标准协议仍是同步完成：

- 写入完成、相关记忆可立即检索后，Add 返回 HTTP 200。
- 服务内部可以异步处理，但标准 Add 端点默认等待完成再返回。

当前 HTTP 202 规则：

```text
HTTP 202 / task_id =
CONDITIONALLY ALLOWED ONLY WITH PRE-APPROVED STATUS ENDPOINT
```

具体条件：

1. 提交版本已经审核绑定 Add 状态查询地址；
2. 该地址包含 `{task_id}`；
3. 202 响应必须返回 `task_id`；
4. 该 202 响应不要求 `memory_ids`；
5. 未绑定上述状态查询地址时，HTTP 202 仍属于契约错误。

**ZHIJI AML 当前候选策略：继续同步 HTTP 200，不启用可选 202 分支。**
规则放宽不构成代码级变更要求。

## 4. Add/Search shape

- Ordinary Textual：达到 **20 messages 或 2,000 Adapter-counted words** 任一边界时确定性分块。
- 单次 Add / Search 请求最长 **30 分钟**。
- Search 必须使用对应 Add 的精确 `user_id` 隔离范围。
- 正式外部评测 `top_k = 100`。
- Search 返回记忆证据，不得替换为最终答案或使用 benchmark gold。

## 5. Textual Streaming

Streaming 是 **Textual 内的正式持续记忆评测能力/子流程**，不是第四条独立赛道。

- Add：Streaming 按冻结的增量会话/任务单元写入。
- Search：Streaming 可在多个增量节点对**同一道题**发起检索。
- 选择题 options 仍以 Search 顶层数组传入。

因此旧规则中的“每题一次 Search”只适用于普通基础流，**不得拿来否定 Streaming 多节点检索**。

## 6. Evaluation quotas

```text
SMOKE = max 1/hour; max 30 per track this edition
FULL  = max 2 per AML Key and track
SECOND_FULL = requires 30-day interval
```

当前官方页面不同位置对 30 天起点存在 “first Full submission” / “first Full completed” 两种文字。
这属于流程排期细节，不影响候选代码；在进一步官方澄清前按更保守排期处理。

## 7. Candidate impact

```text
CODE_LEVEL_RULE_REQUIRED = NONE
CURRENT_SYNC_200_IMPLEMENTATION = COMPLIANT
OPTIONAL_ASYNC_202_IMPLEMENTATION = NOT_REQUIRED
STREAMING = SCORE_RELEVANT / CONTRACT_COMPATIBLE
CANONICAL = V1 A1-kr (UNCHANGED BY THIS RULE UPDATE)
```

本次规则更新不得自动触发 candidate mutation、Official Smoke/Full、push、deploy 或 promotion。
