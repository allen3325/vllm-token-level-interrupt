# Critical Issues Found in Interruptible Inference Implementation

## 🔴 CRITICAL BUG #1: KV Cache is Discarded During Sleep

### Problem
The KV cache is **completely discarded** during sleep, but we're trying to restore requests that expect their KV cache to still exist.

### Root Cause
```python
# gpu_worker.py:157
allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
```

This means:
- **Level 1 sleep**: Offload `weights`, **discard KV cache**
- **Level 2 sleep**: **Discard everything** (empty tuple = no offload)

### Expected Behavior
When `preserve_state=True`, we need to preserve the KV cache along with the request state.

### Impact
- After wake_up, restored requests have `num_computed_tokens > 0`
- They expect their KV cache to exist
- But KV cache blocks have been freed/discarded
- When scheduler tries to schedule them, it will:
  1. Either allocate new empty blocks (wrong data)
  2. Or try to access non-existent blocks (crash)

### Fix Required
```python
# When preserve_state=True, we need to preserve KV cache too
if preserve_buffers:
    # Offload both weights AND kv_cache
    allocator.sleep(offload_tags=("weights", "kv_cache"))
else:
    # Original behavior
    allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
```

---

## 🟡 MAJOR BUG #2: Block Allocations Not Actually Restored

### Problem
```python
# kv_cache_manager.py:464-476
def restore_block_allocations(self, allocations: dict[str, list[int]]) -> None:
    logger.info("Restoring block allocations for %d requests", len(allocations))

    for request_id, block_ids in allocations.items():
        logger.debug("Request %s has %d allocated blocks", request_id, len(block_ids))

    # The actual restoration happens when requests are re-added to the
    # scheduler and blocks are re-allocated. The coordinator maintains
    # block metadata across sleep/wake cycles.
```

This method **does nothing**! It just logs and returns.

### Impact
1. Block IDs are saved in checkpoint
2. But they're never restored to the coordinator's internal tracking
3. When requests are rescheduled, `allocate_slots()` will allocate **new** blocks
4. Old block IDs in checkpoint are meaningless

### Current Flow (Broken)
```
Sleep:
  Request has blocks [10, 11, 12]
  → Save to checkpoint
  → KV cache discarded

Wake:
  → restore_block_allocations() does nothing
  → Request rescheduled
  → allocate_slots() assigns NEW blocks [50, 51, 52]
  → But KV cache content is lost anyway (Bug #1)
```

### Fix Required
We need to actually restore the block allocations to the coordinator:
```python
def restore_block_allocations(self, allocations: dict[str, list[int]]) -> None:
    for request_id, block_ids in allocations.items():
        # Actually restore to coordinator's tracking
        self.coordinator.restore_request_blocks(request_id, block_ids)
```

---

## 🟠 MEDIUM BUG #3: Scheduler State Not Fully Cleared

### Problem
Several scheduler state variables are not cleared during checkpoint restoration:

```python
# scheduler.py:1632-1637
def restore_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
    self.requests.clear()
    self.waiting.clear()
    self.running.clear()
    self.finished_req_ids.clear()
    self.prev_step_scheduled_req_ids.clear()

    # But NOT cleared:
    # - self.finished_req_ids_dict (for multi-engine)
    # - self.finished_recving_kv_req_ids (for KV connector)
    # - self.failed_recving_kv_req_ids (for KV connector)
    # - self.encoder_cache_manager (for multimodal)
```

### Impact
- **Multi-engine mode**: Stale finished request IDs
- **KV Connector mode**: Stale KV transfer state
- **Multimodal requests**: Encoder cache mismatches

### Fix Required
```python
def restore_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
    # Clear ALL state
    self.requests.clear()
    self.waiting.clear()
    self.running.clear()
    self.finished_req_ids.clear()
    self.prev_step_scheduled_req_ids.clear()

    # Also clear optional state
    if self.finished_req_ids_dict is not None:
        self.finished_req_ids_dict.clear()

    self.finished_recving_kv_req_ids.clear()
    self.failed_recving_kv_req_ids.clear()

    # Clear encoder cache
    # (encoder outputs are not preserved across sleep)
```

---

## 🟠 MEDIUM BUG #4: Encoder Cache Not Handled

### Problem
Multimodal requests may have cached encoder outputs in `encoder_cache_manager`. These are not saved or restored.

### Impact
- After wake_up, encoder cache is empty
- Multimodal requests will need to re-encode images/audio
- This is inefficient but not catastrophic

### Options
1. **Acceptable**: Clear encoder cache and re-encode (current behavior)
2. **Better**: Save/restore encoder cache in checkpoint

---

## 🟢 MINOR ISSUE #5: Input Batch Clearing

### Current Implementation
```python
# gpu_worker.py:139-142
if preserve_buffers:
    self.model_runner.input_batch.req_id_to_index.clear()
    self.model_runner.input_batch._req_ids.clear()
```

This only clears the request ID mappings, but not the actual data tensors:
- `token_ids_cpu`
- `num_computed_tokens_cpu`
- `block_table`
- etc.

### Impact
- Probably safe because these get overwritten when requests are re-added
- But could cause issues if old data is accessed before being overwritten

### Recommendation
Add a full `input_batch.clear()` method or ensure all arrays are properly reset.

---

## 🔵 COMPATIBILITY CONCERNS

### 1. Speculative Decoding
- `spec_token_ids` are saved/restored ✅
- But speculative state in model runner is not preserved
- Likely OK: spec tokens will be regenerated

### 2. Pipeline Parallelism
- Different ranks have different states
- Our checkpoint only saves scheduler state (rank 0)
- Workers on other ranks need coordination
- **Status**: Unknown if this works correctly

### 3. LoRA
- `lora_request` is saved/restored ✅
- But LoRA adapter state in model runner is not preserved
- Likely OK: LoRA adapters will be reloaded

### 4. Structured Outputs (Grammar/JSON)
- `structured_output_request` is saved ✅
- FSM state is saved in `events` ✅
- But compiled FSM in `structured_output_manager` may be lost
- **Needs verification**

### 5. Chunked Prefill
- Should work: requests with partial `num_computed_tokens` will continue
- But KV cache bug (#1) breaks this

---

## 🎯 PRIORITY FIXES

### P0 (Critical - Breaks Core Functionality)
1. **Fix KV Cache preservation**: Must offload KV cache when preserve_state=True
2. **Fix block allocation restoration**: Actually restore block assignments

### P1 (Major - Causes Inconsistency)
3. **Clear all scheduler state**: Prevent stale state bugs

### P2 (Medium - Feature-Specific Issues)
4. **Handle encoder cache**: Clear or preserve
5. **Verify pipeline parallelism**: Test with PP > 1

### P3 (Minor - Optimization)
6. **Full input_batch cleanup**: Add comprehensive clear method

---

## 📋 TESTING GAPS

Current bugs suggest these scenarios were not tested:
1. ✗ Sleep during active generation (not just before)
2. ✗ Level 2 sleep with active requests
3. ✗ Multiple requests with different progress
4. ✗ Multimodal requests
5. ✗ Speculative decoding + sleep
6. ✗ Pipeline parallelism + sleep
7. ✗ KV connector + sleep

---

## 🔧 RECOMMENDED FIXES

See next commits for implementation.
