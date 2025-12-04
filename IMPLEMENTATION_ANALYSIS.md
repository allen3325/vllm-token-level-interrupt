# Interruptible Inference 實作分析文件

## 目錄
1. [概述](#1-概述)
2. [相關 Commits](#2-相關-commits)
3. [核心檔案變更分析](#3-核心檔案變更分析)
4. [資料流與架構](#4-資料流與架構)
5. [實作細節深度解析](#5-實作細節深度解析)
6. [潛在問題與風險](#6-潛在問題與風險)
7. [測試覆蓋分析](#7-測試覆蓋分析)
8. [建議與改進](#8-建議與改進)

---

## 1. 概述

### 1.1 功能描述
此實作為 vLLM 引擎增加了 **token-level 可中斷推理** 功能，讓引擎可以：
- 在有活躍請求時安全進入 sleep 模式
- 保存完整的請求狀態（包含 KV cache）
- 在 wake_up 後無縫恢復推理，從中斷處繼續

### 1.2 設計原則
- **Opt-in 設計**：透過 `preserve_state=True` 參數啟用（預設 `False`）
- **向後相容**：不使用新參數時，行為與原本完全相同
- **安全性**：Sleep 前會先 preempt 所有 running 的請求

### 1.3 核心概念
```
睡眠流程：
Running Requests → Preempt → Save Checkpoint → Offload GPU Memory → Sleep

喚醒流程：
Wake → Restore GPU Memory → Load Checkpoint → Restore Scheduler State → Resume
```

---

## 2. 相關 Commits

### 2.1 主要功能實作
| Commit | 描述 |
|--------|------|
| `76e3559ab` | **[Feature]** 初始實作 - 建立 checkpoint 系統、修改 scheduler/KV cache manager/engine core |
| `919af8559` | **[Refactor]** 將功能改為 opt-in 設計，新增 `preserve_state` 參數 |

### 2.2 關鍵 Bug 修復
| Commit | 描述 | 嚴重程度 |
|--------|------|----------|
| `3fe517da4` | **[CRITICAL FIX]** 修復 KV cache 在 sleep 時被丟棄的問題 | 🔴 Critical |
| `92efdfd55` | **[Fix]** 反序列化時重建 ConstantList wrappers | 🟠 Major |
| `6ae8a8d91` | **[Fix]** 清除 input_batch 而非 requests cache | 🟡 Medium |
| `f8d8227ea` | **[Fix]** 在 checkpoint-based sleep 時清除 model runner cache | 🟡 Medium |
| `d16d6f9c4` | **[Fix]** 恢復 checkpoint 時清除 prev_step_scheduled_req_ids | 🟡 Medium |
| `53c9af932` | **[Fix]** 修正 RequestQueue 方法名稱 | 🟢 Minor |
| `b790e930a` | **[Fix]** 無 checkpoint 時不保留 model buffers | 🟠 Major |
| `9358f4f76` | **[Fix]** 無 checkpoint 時重置 prefix cache | 🟠 Major |
| `766d036d1` | **[Fix]** 只有在有活躍請求時才建立 checkpoint | 🟢 Minor |

### 2.3 文件更新
| Commit | 描述 |
|--------|------|
| `5d0966de1` | 新增測試指南 |
| `f11205b6e` | 新增架構圖文件 |
| `52ab3c5af` | 更新所有文件至最新版本 |
| `667c8cce9` | 更新文件 |

---

## 3. 核心檔案變更分析

### 3.1 新增檔案

#### `vllm/v1/engine/checkpoint.py` - Checkpoint 管理器
```python
# 核心資料結構
@dataclass
class SchedulerCheckpoint:
    requests: dict[str, Request]           # 所有追蹤的請求
    waiting_queue_data: list[tuple]        # 等待佇列 (request_id, priority, arrival_time)
    running_request_ids: list[str]         # 執行中請求 IDs
    kv_block_allocations: dict[str, list]  # KV cache block 分配
    prefix_cache_state: dict | None        # Prefix cache 映射

@dataclass
class EngineCheckpoint:
    scheduler_checkpoint: SchedulerCheckpoint
    output_processor_checkpoint: OutputProcessorCheckpoint | None
    timestamp: float
    vllm_version: str

# 核心類別
class CheckpointManager:
    def save_checkpoint(...)    # 儲存引擎狀態
    def restore_checkpoint(...) # 恢復引擎狀態
    def clear_checkpoint(...)   # 清除 checkpoint
    def has_checkpoint(...)     # 檢查是否有 checkpoint
```

**序列化/反序列化函數**：
- `serialize_request()`: 將 Request 物件轉為字典
- `deserialize_request()`: 從字典重建 Request 物件

**序列化內容包含**：
- 基本資訊：request_id, client_index, priority, arrival_time
- Token 資料：prompt_token_ids, _output_token_ids, _all_token_ids
- 執行狀態：num_computed_tokens, num_cached_tokens, status
- 參數：sampling_params, pooling_params (使用 pickle)
- 多模態：mm_features, prompt_embeds
- KV 相關：block_hashes, kv_transfer_params

### 3.2 修改檔案

#### `vllm/v1/core/sched/scheduler.py` - 排程器增強
```python
# 新增方法

def prepare_for_sleep(self) -> None:
    """將所有 running 請求 preempt 回 waiting queue"""
    for request in self.running:
        if not request.is_finished():
            request.status = RequestStatus.PREEMPTED
            request.num_preemptions += 1
            self.waiting.add_request(request)
    self.running.clear()

def get_checkpoint_state(self) -> dict[str, Any]:
    """匯出排程器狀態為 checkpoint"""
    # 序列化所有請求
    # 收集 waiting queue 資料
    # 收集 running request IDs
    # 匯出 KV block allocations
    # 匯出 prefix cache 狀態

def restore_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
    """從 checkpoint 恢復排程器狀態"""
    # 清除所有狀態
    self.requests.clear()
    self.waiting.clear()
    self.running.clear()
    self.finished_req_ids.clear()
    self.prev_step_scheduled_req_ids.clear()
    # 清除多引擎/KV connector 狀態
    if self.finished_req_ids_dict:
        self.finished_req_ids_dict.clear()
    self.finished_recving_kv_req_ids.clear()
    self.failed_recving_kv_req_ids.clear()
    # 反序列化並恢復請求
    # 恢復 waiting/running queues
    # 恢復 KV block allocations 和 prefix cache
```

#### `vllm/v1/core/kv_cache_manager.py` - KV Cache 管理
```python
# 新增方法

def get_block_allocations(self) -> dict[str, list[int]]:
    """匯出所有請求的 block 分配"""
    allocations = {}
    for request_id in self.coordinator.get_all_request_ids():
        blocks = self.coordinator.get_blocks(request_id)
        block_ids = [block.block_id for group in blocks for block in group]
        if block_ids:
            allocations[request_id] = block_ids
    return allocations

def restore_block_allocations(self, allocations: dict[str, list[int]]) -> None:
    """恢復 block 分配（主要用於 logging/驗證）"""
    # 實際 block 內容由 CuMemAllocator 維護
    # 此方法記錄分配資訊供除錯用

def export_prefix_cache(self) -> dict[str, Any]:
    """匯出 prefix cache 狀態"""
    return self.coordinator.export_prefix_cache()

def restore_prefix_cache(self, state: dict[str, Any]) -> None:
    """恢復 prefix cache 狀態"""
    self.coordinator.restore_prefix_cache(state)
```

#### `vllm/v1/core/kv_cache_coordinator.py` - KV Cache 協調器
```python
# 新增方法（基礎實作）

def get_all_request_ids(self) -> list[str]:
    """取得所有有分配 blocks 的 request IDs"""
    request_ids = set()
    for manager in self.single_type_managers:
        request_ids.update(manager.req_to_blocks.keys())
    return list(request_ids)

def export_prefix_cache(self) -> dict:
    """匯出 prefix cache 狀態（子類別可覆寫）"""
    return {}

def restore_prefix_cache(self, state: dict) -> None:
    """恢復 prefix cache 狀態（子類別可覆寫）"""
    pass
```

#### `vllm/v1/engine/core.py` - 引擎核心
```python
def __init__(self, ...):
    # 初始化 checkpoint manager
    from vllm.v1.engine.checkpoint import CheckpointManager
    self.checkpoint_manager = CheckpointManager()

def sleep(self, level: int = 1, preserve_state: bool = False):
    """進入睡眠模式"""
    preserve_buffers = False
    
    if preserve_state:
        # 檢查是否有請求需要保留
        has_requests = (
            len(self.scheduler.requests) > 0 or
            len(self.scheduler.running) > 0 or
            len(self.scheduler.waiting) > 0
        )
        
        if has_requests:
            # 1. Preempt 所有 running 請求
            self.scheduler.prepare_for_sleep()
            # 2. 匯出 scheduler 狀態
            scheduler_checkpoint = self.scheduler.get_checkpoint_state()
            # 3. 儲存 checkpoint
            self.checkpoint_manager.save_checkpoint(checkpoint)
            preserve_buffers = True
        else:
            self.checkpoint_manager.clear_checkpoint()
    else:
        self.checkpoint_manager.clear_checkpoint()
    
    # 4. Offload GPU 記憶體
    self.model_executor.sleep(level, preserve_buffers=preserve_buffers)

def wake_up(self, tags: list[str] | None = None):
    """從睡眠中喚醒"""
    # 1. 恢復 GPU 記憶體
    self.model_executor.wake_up(tags)
    
    # 2. 恢復 checkpoint（如果有）
    if self.checkpoint_manager.has_checkpoint():
        checkpoint = self.checkpoint_manager.restore_checkpoint()
        if checkpoint:
            self.scheduler.restore_checkpoint_state(scheduler_checkpoint)
            # 3. 重新建立 block hashers
            if self.request_block_hasher:
                for request in self.scheduler.requests.values():
                    if request.get_hash_new_full_blocks is None:
                        request.get_hash_new_full_blocks = partial(
                            self.request_block_hasher, request
                        )
            self.checkpoint_manager.clear_checkpoint()
    else:
        # 無 checkpoint 時重置 prefix cache
        self.scheduler.reset_prefix_cache()
```

#### `vllm/v1/worker/gpu_worker.py` - GPU Worker
```python
def sleep(self, level: int = 1, preserve_buffers: bool = True) -> None:
    if preserve_buffers:
        # 清除 input batch mappings
        self.model_runner.input_batch.req_id_to_index.clear()
        self.model_runner.input_batch._req_ids.clear()
    
    # 決定 offload 策略
    if preserve_buffers:
        # 保留狀態時：必須同時 offload weights 和 kv_cache
        offload_tags = ("weights", "kv_cache")
    else:
        # 原始行為：只 offload weights（level 1）或不 offload（level 2）
        offload_tags = ("weights",) if level == 1 else tuple()
    
    allocator.sleep(offload_tags=offload_tags)
```

#### API 層修改
**`vllm/entrypoints/openai/api_server.py`**:
```python
@router.post("/sleep")
async def sleep(raw_request: Request):
    level = raw_request.query_params.get("level", "1")
    preserve_state = raw_request.query_params.get("preserve_state", "false").lower() == "true"
    await engine_client(raw_request).sleep(int(level), preserve_state=preserve_state)
```

**`vllm/v1/engine/async_llm.py`**, **`vllm/v1/engine/llm_engine.py`**, **`vllm/v1/engine/core_client.py`**:
- 所有相關方法都新增 `preserve_state` 參數

---

## 4. 資料流與架構

### 4.1 Sleep 流程
```
┌─────────────────────────────────────────────────────────────────────┐
│                         SLEEP flow                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Client                                                             │
│    │                                                                │
│    ▼ sleep(level=1, preserve_state=True)                            │
│  ┌─────────────┐                                                    │
│  │ API Server  │                                                    │
│  └──────┬──────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────┐                                                │
│  │   EngineCore    │                                                │
│  │    .sleep()     │                                                │
│  └────────┬────────┘                                                │
│           │                                                         │
│           │ 1. prepare_for_sleep()                                  │
│           ▼                                                         │
│  ┌─────────────────┐     ┌──────────────────────────────────────┐   │
│  │   Scheduler     │────▶│ • Preempt running requests           │   │
│  │                 │     │ • Set status = PREEMPTED             │   │
│  │                 │     │ • Move to waiting queue              │   │
│  └────────┬────────┘     └──────────────────────────────────────┘   │
│           │                                                         │
│           │ 2. get_checkpoint_state()                               │
│           ▼                                                         │
│  ┌─────────────────┐     ┌──────────────────────────────────────┐   │
│  │ CheckpointMgr   │────▶│ • Serialize all requests             │   │
│  │                 │     │ • Export waiting queue data          │   │
│  │                 │     │ • Export KV block allocations        │   │
│  │                 │     │ • Export prefix cache state          │   │
│  └────────┬────────┘     └──────────────────────────────────────┘   │
│           │                                                         │
│           │ 3. model_executor.sleep()                               │
│           ▼                                                         │
│  ┌─────────────────┐     ┌──────────────────────────────────────┐   │
│  │   GPU Worker    │────▶│ • Clear input_batch mappings         │   │
│  │                 │     │ • Offload weights to CPU             │   │
│  │                 │     │ • Offload KV cache to CPU            │   │
│  │                 │     │ • Free GPU memory                    │   │
│  └─────────────────┘     └──────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Wake 流程
```
┌─────────────────────────────────────────────────────────────────────┐
│                         WAKE flow                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Client                                                             │
│    │                                                                │
│    ▼ wake_up()                                                      │
│  ┌─────────────┐                                                    │
│  │ API Server  │                                                    │
│  └──────┬──────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────┐                                                │
│  │   EngineCore    │                                                │
│  │   .wake_up()    │                                                │
│  └────────┬────────┘                                                │
│           │                                                         │
│           │ 1. model_executor.wake_up()                             │
│           ▼                                                         │
│  ┌─────────────────┐     ┌──────────────────────────────────────┐   │
│  │   GPU Worker    │────▶│ • Restore weights from CPU           │   │
│  │                 │     │ • Restore KV cache from CPU          │   │
│  │                 │     │ • Allocate GPU memory                │   │
│  └────────┬────────┘     └──────────────────────────────────────┘   │
│           │                                                         │
│           │ 2. restore_checkpoint()                                 │
│           ▼                                                         │
│  ┌─────────────────┐     ┌──────────────────────────────────────┐   │
│  │ CheckpointMgr   │────▶│ • Return saved checkpoint            │   │
│  │                 │     │                                      │   │
│  └────────┬────────┘     └──────────────────────────────────────┘   │
│           │                                                         │
│           │ 3. restore_checkpoint_state()                           │
│           ▼                                                         │
│  ┌─────────────────┐     ┌──────────────────────────────────────┐   │
│  │   Scheduler     │────▶│ • Clear all queues                   │   │
│  │                 │     │ • Deserialize requests               │   │
│  │                 │     │ • Restore waiting queue              │   │
│  │                 │     │ • Restore KV block mappings          │   │
│  │                 │     │ • Restore prefix cache               │   │
│  └────────┬────────┘     └──────────────────────────────────────┘   │
│           │                                                         │
│           │ 4. Re-establish block hashers                           │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │ Resume Inference│                                                │
│  │ from where it   │                                                │
│  │ left off!       │                                                │
│  └─────────────────┘                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 記憶體管理
```
┌────────────────────────────────────────────────────────────────────┐
│                      Memory State Changes                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Normal Operation:                                                 │
│  ┌──────────────────────────────────────────┐                      │
│  │              GPU Memory                  │                      │
│  │  ┌────────────┐  ┌─────────────────────┐ │                      │
│  │  │  Weights   │  │    KV Cache         │ │                      │
│  │  │  (large)   │  │  Block 0,1,2...     │ │                      │
│  │  └────────────┘  └─────────────────────┘ │                      │
│  └──────────────────────────────────────────┘                      │
│                                                                    │
│  After Sleep (preserve_state=True):                                │
│  ┌──────────────────────────────────────────┐                      │
│  │              GPU Memory (empty)          │                      │
│  └──────────────────────────────────────────┘                      │
│  ┌──────────────────────────────────────────┐                      │
│  │              CPU Memory                  │                      │
│  │  ┌────────────┐  ┌─────────────────────┐ │                      │
│  │  │  Weights   │  │    KV Cache         │ │  ← offloaded         │
│  │  │  (backup)  │  │  (preserved)        │ │                      │
│  │  └────────────┘  └─────────────────────┘ │                      │
│  │  ┌─────────────────────────────────────┐ │                      │
│  │  │  Checkpoint (Request state)         │ │  ← saved             │
│  │  └─────────────────────────────────────┘ │                      │
│  └──────────────────────────────────────────┘                      │
│                                                                    │
│  After Sleep (preserve_state=False, Original Behavior):            │
│  ┌──────────────────────────────────────────┐                      │
│  │              GPU Memory (empty)          │                      │
│  └──────────────────────────────────────────┘                      │
│  ┌──────────────────────────────────────────┐                      │
│  │              CPU Memory                  │                      │
│  │  ┌────────────┐                          │                      │
│  │  │  Weights   │                          │  ← Weights Only      │
│  │  │  (backup)  │                          │                      │
│  │  └────────────┘                          │                      │
│  └──────────────────────────────────────────┘                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 5. 實作細節深度解析

### 5.1 Request 序列化策略

**序列化的欄位**：
```python
data = {
    # 基本識別
    "request_id": request.request_id,
    "client_index": request.client_index,
    "priority": request.priority,
    "arrival_time": request.arrival_time,
    
    # 執行狀態
    "status": request.status,
    "stop_reason": request.stop_reason,
    "num_computed_tokens": request.num_computed_tokens,
    "num_cached_tokens": request.num_cached_tokens,
    "num_preemptions": request.num_preemptions,
    
    # Token 資料
    "prompt_token_ids": request.prompt_token_ids,
    "_output_token_ids": request._output_token_ids.copy(),
    "_all_token_ids": request._all_token_ids.copy(),
    
    # Block hashes（用於 prefix caching）
    "block_hashes": [h.to_bytes() for h in request.block_hashes],
    
    # 複雜物件使用 pickle
    "sampling_params": pickle.dumps(request.sampling_params),
    "mm_features": pickle.dumps(request.mm_features),
    ...
}
```

**特殊處理**：
1. `prompt_embeds` - 如果是 tensor，先移到 CPU
2. `block_hashes` - 轉為 bytes 格式儲存
3. `ConstantList` - 反序列化後需重建 wrapper

### 5.2 KV Cache 保存機制

**關鍵發現**：實際的 KV cache 資料不是由 checkpoint 系統保存，而是由 `CuMemAllocator` 的 `sleep()` 機制處理。

```python
# gpu_worker.py
if preserve_buffers:
    offload_tags = ("weights", "kv_cache")  # 關鍵！
else:
    offload_tags = ("weights",) if level == 1 else tuple()

allocator.sleep(offload_tags=offload_tags)
```

這意味著：
- `weights` tag 的記憶體會被 offload 到 CPU
- `kv_cache` tag 的記憶體會被 offload 到 CPU（僅在 preserve_state=True）
- Wake up 時會從 CPU 恢復回 GPU

### 5.3 Preemption 機制

```python
def prepare_for_sleep(self) -> None:
    for request in self.running:
        if not request.is_finished():
            request.status = RequestStatus.PREEMPTED
            request.num_preemptions += 1
            self.waiting.add_request(request)
    self.running.clear()
```

**重要**：
- 所有 running 的請求都會被設為 `PREEMPTED` 狀態
- 移回 waiting queue，保持 priority 順序
- `num_preemptions` 計數器遞增（可用於監控）
- running list 完全清空

### 5.4 狀態清除的完整性

```python
def restore_checkpoint_state(self, checkpoint):
    # 核心狀態
    self.requests.clear()
    self.waiting.clear()
    self.running.clear()
    self.finished_req_ids.clear()
    self.prev_step_scheduled_req_ids.clear()  # 重要！防止 assertion error
    
    # 多引擎狀態
    if self.finished_req_ids_dict is not None:
        self.finished_req_ids_dict.clear()
    
    # KV Connector 狀態
    self.finished_recving_kv_req_ids.clear()
    self.failed_recving_kv_req_ids.clear()
```

---

## 6. 潛在問題與風險

### 6.1 已確認的問題

#### 🟡 Encoder Cache 未處理
**狀態**：已知限制
**影響**：多模態請求需要在 wake_up 後重新編碼圖片/音訊
**建議**：如果多模態推理是重要場景，考慮新增 encoder cache 保存

#### 🟡 Pipeline Parallelism 未驗證
**狀態**：未測試
**風險**：Checkpoint 只保存 rank 0 的 scheduler 狀態，其他 rank 可能不一致
**建議**：需要在 PP > 1 環境下測試

### 6.2 潛在風險

#### ⚠️ 風險 1: Pickle 安全性
```python
"sampling_params": pickle.dumps(request.sampling_params),
```
**問題**：使用 pickle 序列化複雜物件可能有安全風險
**影響**：如果 checkpoint 被惡意修改，可能導致任意代碼執行
**緩解**：Checkpoint 只存在記憶體中，不會寫入磁碟

#### ⚠️ 風險 2: Block ID 重複使用
**問題**：`restore_block_allocations()` 沒有實際執行任何操作
```python
def restore_block_allocations(self, allocations):
    # 只有 logging，沒有實際恢復
    for request_id, block_ids in allocations.items():
        logger.debug(...)
```
**分析**：這是設計上的決定 - 實際 block 內容由 CuMemAllocator 維護
**風險**：如果 allocator 行為改變，可能導致不一致

#### ⚠️ 風險 3: Structured Output 狀態
**問題**：`structured_output_request` 被序列化，但 `structured_output_manager` 中編譯的 FSM 可能遺失
**影響**：恢復後可能需要重新編譯語法/JSON schema
**緩解**：FSM 編譯通常很快

#### ⚠️ 風險 4: 並發 Sleep/Wake
**問題**：沒有明確的鎖機制防止並發操作
**影響**：如果多個執行緒同時呼叫 sleep/wake_up，可能導致狀態不一致
**建議**：API 層應該有排它鎖

### 6.3 序列化完整性問題

經過詳細分析 `serialize_request()` 和 `deserialize_request()` 函數，發現以下潛在問題：

#### 🔴 問題 1: `max_tokens` 序列化但未使用
**嚴重程度**: 中等
```python
# serialize_request:
"max_tokens": request.max_tokens,

# deserialize_request:
# max_tokens 未被恢復，依賴 constructor 從 sampling_params 重新計算
```
**影響**：如果 `max_tokens` 在初始化後被修改，修改會遺失
**緩解**：目前 vLLM 不會在初始化後修改 `max_tokens`

#### 🟡 問題 2: `num_prompt_tokens` 序列化但未恢復
**嚴重程度**: 低
```python
# 序列化但從未在 deserialize 中使用
"num_prompt_tokens": request.num_prompt_tokens,
```
**影響**：浪費的序列化，但功能上正確（constructor 會重新計算）

#### 🟡 問題 3: `num_encoder_inputs` 和 `has_encoder_inputs` 冗餘
**嚴重程度**: 低
```python
"num_encoder_inputs": request.num_encoder_inputs,  # 序列化但未恢復
"has_encoder_inputs": request.has_encoder_inputs,  # 序列化但未恢復
```
**影響**：這些值從 `mm_features` 派生，constructor 會重新計算

#### 🟢 問題 4: `mm_features` 的空列表處理
**嚴重程度**: 低（已處理）
```python
# serialize: 使用 falsy 檢查
"mm_features": pickle.dumps(request.mm_features) if request.mm_features else None,
# 空列表 [] 會被序列化為 None，但 deserialize 時 constructor 會轉為 []
```
**影響**：邊界情況處理正確，但語義上略有不一致

#### ✅ 問題 5: `get_hash_new_full_blocks` 未序列化
**嚴重程度**: 已解決
```python
# 這是一個函數，無法序列化
# 但在 wake_up() 中已正確重建：
if self.request_block_hasher is not None:
    for request in self.scheduler.requests.values():
        if request.get_hash_new_full_blocks is None:
            request.get_hash_new_full_blocks = partial(
                self.request_block_hasher, request
            )
```

#### 🟡 問題 6: `BlockHash` 重建依賴
**嚴重程度**: 中等
```python
# 假設 BlockHash 可以從 bytes 重建
request.block_hashes = [BlockHash(h) for h in data["block_hashes"]]
```
**影響**：如果 `BlockHash` 類別的建構函數語義改變，可能會失敗

### 6.4 程式碼品質問題

#### 📝 問題 1: Magic String
```python
offload_tags = ("weights", "kv_cache")
```
**建議**：使用常數或 enum

#### 📝 問題 2: 缺少型別註解
部分新增的方法缺少完整的型別註解

#### 📝 問題 3: 重複程式碼
`gpu_worker.py` 和潛在的 `cpu_worker.py` 可能有重複邏輯

#### 📝 問題 4: 序列化冗餘
多個欄位被序列化但從未在反序列化中使用：
- `max_tokens`
- `num_prompt_tokens`
- `num_encoder_inputs`
- `has_encoder_inputs`

**建議**：移除這些冗餘序列化以減少 checkpoint 大小

---

## 7. 測試覆蓋分析

### 7.1 現有測試
`tests/entrypoints/openai/test_interruptible_inference.py` 包含：

1. **test_interruptible_inference()**
   - ✅ 測試狀態保留：相同 prompt、temperature=0 應產生相同輸出
   - ✅ 測試 sleep/wake 基本流程
   - ✅ 測試 is_sleeping 端點

2. **test_sleep_with_active_requests()**
   - ✅ 測試無活躍請求時的 sleep
   - ✅ 測試 wake 後新請求正常運作

### 7.2 測試缺口
以下場景未被測試覆蓋：

| 場景 | 狀態 |
|------|------|
| Sleep 期間有正在生成的請求 | ❌ 未測試 |
| Level 2 sleep with active requests | ❌ 未測試 |
| 多個請求同時進行中 | ❌ 未測試 |
| 多模態請求 | ❌ 未測試 |
| Speculative decoding + sleep | ❌ 未測試 |
| Pipeline parallelism + sleep | ❌ 未測試 |
| KV connector + sleep | ❌ 未測試 |
| LoRA requests + sleep | ❌ 未測試 |
| Structured output + sleep | ❌ 未測試 |
| 連續多次 sleep/wake | ❌ 未測試 |
| 並發 sleep/wake 請求 | ❌ 未測試 |

### 7.3 建議新增測試

```python
# 建議新增的測試案例

def test_sleep_during_generation():
    """測試在生成過程中 sleep"""
    # 啟動 streaming 請求
    # 在生成中途 sleep
    # Wake up 後驗證能繼續生成
    pass

def test_multiple_requests_sleep():
    """測試多個請求同時被 sleep"""
    # 同時發送多個請求
    # Sleep
    # Wake up
    # 驗證所有請求都能繼續
    pass

def test_rapid_sleep_wake():
    """測試快速連續 sleep/wake"""
    for _ in range(10):
        sleep(preserve_state=True)
        wake_up()
        # 驗證狀態一致性
    pass
```

---

## 8. 建議與改進

### 8.1 短期改進 (P0)

1. **新增並發保護**
```python
# 在 EngineCore 新增鎖
self._sleep_lock = threading.Lock()

def sleep(self, ...):
    with self._sleep_lock:
        # 現有邏輯
```

2. **新增更多日誌**
```python
logger.info("Sleep: requests=%d, running=%d, waiting=%d",
            len(self.scheduler.requests),
            len(self.scheduler.running),
            len(self.scheduler.waiting))
```

3. **新增監控指標**
```python
# 追蹤 sleep/wake 次數和持續時間
metrics.counter("interruptible_inference_sleep_count").inc()
metrics.histogram("interruptible_inference_sleep_duration_seconds").observe(duration)
```

### 8.2 中期改進 (P1)

1. **Checkpoint 壓縮**
```python
import zlib
compressed = zlib.compress(pickle.dumps(checkpoint))
```

2. **非同步 Checkpoint**
```python
async def save_checkpoint_async(self, checkpoint):
    await asyncio.get_event_loop().run_in_executor(
        None, self.save_checkpoint, checkpoint
    )
```

3. **完善 Encoder Cache 處理**
```python
def get_checkpoint_state(self):
    return {
        ...
        "encoder_cache_state": self.encoder_cache_manager.export_state()
    }
```

### 8.3 長期改進 (P2)

1. **持久化 Checkpoint**
   - 支援寫入磁碟
   - 支援跨程序恢復
   - 考慮安全性（不使用 pickle）

2. **增量 Checkpoint**
   - 只保存變更的請求
   - 減少記憶體使用

3. **Pipeline Parallelism 支援**
   - 同步多個 rank 的 checkpoint
   - 協調 sleep/wake 時機

---

## 總結

### 實作評估
| 面向 | 評分 | 說明 |
|------|------|------|
| 功能完整性 | ⭐⭐⭐⭐ | 核心功能完整，但多模態/PP 支援待加強 |
| 向後相容性 | ⭐⭐⭐⭐⭐ | 完美的 opt-in 設計 |
| 程式碼品質 | ⭐⭐⭐ | 可讀性好，但有些地方可改進 |
| 測試覆蓋 | ⭐⭐ | 基本測試存在，但覆蓋不足 |
| 文件品質 | ⭐⭐⭐⭐⭐ | 非常完整的文件 |
| 錯誤處理 | ⭐⭐⭐ | 有基本處理，但可加強 |

### 關鍵優勢
1. **設計良好**：Opt-in 設計確保向後相容
2. **核心問題已解決**：所有 Critical 和 Major bug 都已修復
3. **文件完整**：有詳細的架構圖和使用說明

### 需要注意的風險
1. Pipeline Parallelism 未測試
2. 缺少並發保護
3. 多模態支援不完整
4. 測試覆蓋需要加強

### 建議優先處理
1. 新增更多測試案例（尤其是邊界情況）
2. 驗證 PP > 1 場景
3. 考慮新增並發保護機制
