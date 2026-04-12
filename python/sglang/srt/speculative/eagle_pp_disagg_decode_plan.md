# DeepSeek-V2 + DeepSeek-NexTN (EagleWorker) Spec Decoding with `event_loop_pp_disagg_decode` – Implementation Plan

## Constraints / Assumptions

| Parameter | Value |
|---|---|
| Spec version | v1 (non-overlap, `disable_overlap_schedule=True`) |
| `pp_async_batch_depth` | 0 |
| Event loop | `event_loop_pp_disagg_decode` (decode-only disaggregated server) |
| Draft model ranks | PP-0 (`draft()`) and PP-N-1 (`draft_extend_after_decode()` + `verify()`) only |
| Intermediate ranks | PP-1 … PP-N-2: target-model layers only, no draft model |
| Cold start | Required **for the draft model split only**. The target KV cache is pre-populated by the prefill server — no target-model cold start needed. However, PP-0 has no draft model state on first entry; PP-N-1 must run `forward_draft_extend()` once to produce the initial `EagleDraftInput` + draft KV data and send them to PP-0 before the VERIFY/DRAFT cycle can begin. |
| Overlap schedule | Disabled |
| Grammar / constrained decoding | Not supported in v1 |

---

## High-Level Architecture

```
PP-0 (EaglePPFirstWorker)        PP-1 … PP-N-2 (tp_worker)       PP-N-1 (EaglePPLastWorker)
┌──────────────────────────┐     ┌─────────────────────────┐     ┌──────────────────────────┐
│ target layers [0..L₀)    │     │ target layers (mid)     │     │ target layers [Lₖ..L)    │
│ draft model (draft() only│     │                         │     │ draft model              │
│                          │ ──→ │ ──→ … ──→               │ ──→ │ (verify() +              │
│ prepare_for_verify()     │     │ prepare_for_verify()    │     │  draft_extend_after_     │
│ draft()                  │     │ (lazy KV free on ring)  │     │  decode() only)          │
└──────────────────────────┘     └─────────────────────────┘     └──────────────────────────┘
        ↑ ↓                                                               ↑ ↓
        │ └── proxy channel (fwd): EagleVerifyInput fields + ─────────────┘ │
        │                          draft_kv_slots + draft_kv_data            │
        │                                                                    │
        └────── output ring (bwd): EagleDraftInput fields + ─────────────────┘
                                   extend_kv_slots + extend_kv_data +
                                   evict_mask + accept_index
```

---

## Per-Iteration Data Flow (steady state, mb_id = M, outer iter K)

### Phase 1 – VERIFY forward pass (all ranks)

```
PP-0:
  batch.spec_info = _spec_verify_inputs[M]   (EagleVerifyInput from iter K-1)
  prepare_for_verify()                        allocate out_cache_loc for draft tokens
  run target layers [0..L₀)
  embed into proxy dict:
    hidden_states       (existing)
    spec_vi_*           (EagleVerifyInput fields)
    spec_draft_kv_slots / spec_draft_kv_data   (draft KV entries for PP-N-1)
  send proxy → PP-1

PP-1 … PP-N-2 (each):
  recv proxy from prev stage
  extract EagleVerifyInput from spec_vi_* → prepare_for_verify() locally
  run target layers
  pass proxy through (spec_vi_* + spec_draft_kv_* forwarded unchanged)
  send proxy → next stage

PP-N-1:
  recv proxy from PP-N-2
  extract EagleVerifyInput from spec_vi_* → prepare_for_verify() locally
  write spec_draft_kv_data → local draft KV pool at spec_draft_kv_slots
  run target layers [Lₖ..L)   (VERIFY mode, batch.spec_info = EagleVerifyInput)
  verify()                     → accept_index, evict_mask, EagleDraftInput
  draft_extend_after_decode()  → writes new draft KV entries at extend_out_cache_loc
  collect extend_kv_data = draft_kv_pool[extend_out_cache_loc]
  send via output ring:
    spec_di_*             (EagleDraftInput fields)
    spec_extend_kv_slots / spec_extend_kv_data
    spec_evict_mask / spec_accept_index
```

### Phase 2 – Post-forward: output ring processing (PP-0 and intermediate ranks)

```
PP-0 (_pp_process_batch_result):
  recv from output ring
  reconstruct EagleDraftInput from spec_di_*
  write spec_extend_kv_data → local draft KV pool at spec_extend_kv_slots
  free target KV: token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
  update batch.out_cache_loc = batch.out_cache_loc[accept_index]
  batch.spec_info = EagleDraftInput
  draft()                      → EagleVerifyInput[K]
  _spec_verify_inputs[M] = EagleVerifyInput[K]
  collect draft_kv_data from new draft KV slots (for next proxy send)

PP-1 … PP-N-2 (_pp_process_batch_result, via ring chain):
  recv evict_mask + accept_index
  free target KV: token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
  update batch.out_cache_loc = batch.out_cache_loc[accept_index]
```

---

## Draft KV Cache Transfer

### Why it's needed

- `draft()` on PP-0 writes new KV entries into PP-0's copy of the draft KV pool.
- `draft_extend_after_decode()` on PP-N-1 needs to attend over those entries as context.
- Conversely, `draft_extend_after_decode()` on PP-N-1 writes new KV entries that `draft()` on PP-0 will use as context in the next step.
- Because PP-0 and PP-N-1 are separate GPUs, these entries must be explicitly transferred.

### Why it's not expensive

The draft model (NexTN) is much smaller than the target model. The KV data transferred per step is of the same order of magnitude as the hidden-state proxy already being sent between PP stages for the target model:

| Tensor | Shape | Approx size (bs=32, fp16) |
|---|---|---|
| Target hidden state proxy | `(bs, hidden_dim_target)` = `(32, 5120)` | ~320 KB |
| Draft extend KV (backward) | `(bs × mean_accept_len, 2, kv_heads_draft, head_dim_draft)` | ~100–400 KB |
| Draft decode KV (forward) | `(bs × spec_steps × topk, 2, kv_heads_draft, head_dim_draft)` | ~500 KB–2 MB |

The forward transfer (draft decode KV) is the larger of the two and is embedded in the proxy dict alongside the existing hidden-state tensors. The backward transfer (extend KV) rides the output ring. Both are well within the bandwidth already in use.

### Mechanism

**PP-0 → PP-N-1 (via proxy channel, forward)**

```python
# After draft() on PP-0:
draft_out_cache_loc = batch.out_cache_loc     # slots written by draft()
draft_kv_data = draft_kv_pool.get_flat_data(draft_out_cache_loc)
# Embed into proxy dict alongside hidden states:
proxy_dict["spec_draft_kv_slots"] = draft_out_cache_loc
proxy_dict["spec_draft_kv_data"]  = draft_kv_data

# On PP-N-1, before draft_extend_after_decode():
draft_kv_pool.set_flat_data(
    pp_proxy["spec_draft_kv_slots"],
    pp_proxy["spec_draft_kv_data"]
)
```

**PP-N-1 → PP-0 (via output ring, backward)**

```python
# After draft_extend_after_decode() on PP-N-1:
extend_out_cache_loc = batch.out_cache_loc    # slots written by extend
extend_kv_data = draft_kv_pool.get_flat_data(extend_out_cache_loc)
# Embed into output ring tensor dict:
ring_dict["spec_extend_kv_slots"] = extend_out_cache_loc
ring_dict["spec_extend_kv_data"]  = extend_kv_data

# On PP-0, in _pp_process_batch_result:
draft_kv_pool.set_flat_data(
    ring_dict["spec_extend_kv_slots"],
    ring_dict["spec_extend_kv_data"]
)
```

### Allocator consistency

PP-0 is the authoritative allocator for the draft KV pool. It performs all alloc/free decisions and sends the resulting slot indices to PP-N-1 (embedded in the proxy dict). PP-N-1 writes to those exact slots without calling its own allocator for draft slots. When verify() evicts rejected draft slots, the eviction indices are sent backward (as part of `spec_evict_mask` / `spec_accept_index`) so PP-N-1 can mirror the free.

---

## Evict / Accept Propagation (Target KV Cache)

All PP ranks allocate target KV cache for draft tokens in `prepare_for_verify()`. After verify, rejected tokens must be freed on ALL ranks.

The `evict_mask` and `accept_index` produced by `verify()` on PP-N-1 travel the existing output ring chain:

```
PP-N-1 → [output ring] → PP-0 → PP-1 → … → PP-N-2
```

Each rank in `_pp_process_batch_result`:
```python
if spec_evict_mask is not None:
    token_to_kv_pool_allocator.free(batch.out_cache_loc[spec_evict_mask])
    batch.out_cache_loc = batch.out_cache_loc[spec_accept_index]
```

---

## Files Changed

### 1. `speculative/eagle_worker.py` – Add PP-disagg worker classes

Both worker classes live in the **same file** alongside `EAGLEWorker`. No new files are created.

**`EaglePPFirstWorker(StandaloneWorker)`**

```python
class EaglePPFirstWorker(StandaloneWorker):
    """
    Draft model worker for PP-0 in disagg-decode+spec mode.

    Runs draft() AFTER receiving EagleDraftInput + extend KV data from PP-N-1
    (triggered by scheduler in _pp_process_batch_result, not during the forward).
    During the forward pass it runs the target model in VERIFY mode and embeds
    EagleVerifyInput + draft KV data into the proxy dict for downstream ranks.
    """

    _pending_draft_kv_slots: Optional[torch.Tensor] = None
    _pending_draft_kv_data:  Optional[torch.Tensor] = None

    def forward_batch_generation(
        self, batch: ScheduleBatch, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_extend(batch, pp_proxy_tensors)
        # Disagg decode: no cold start — _spec_verify_inputs[mb_id] is always set.
        assert isinstance(batch.spec_info, EagleVerifyInput), (
            "EaglePPFirstWorker expects EagleVerifyInput on every decode step "
            "(cold start is handled by the prefill server in disagg mode)"
        )
        return self._forward_verify_pp0(batch, pp_proxy_tensors)

    def _forward_verify_pp0(self, batch, pp_proxy_tensors):
        """
        Run target forward in VERIFY mode. Does NOT call verify() or draft()
        here — those are PP-N-1's and post-ring PP-0's responsibilities.
        Attaches the EagleVerifyInput fields and pending draft KV data to the result
        so _pp_prepare_proxy_dict can embed them in the outgoing proxy.
        """
        result = self.target_worker.forward_batch_generation(
            batch.get_model_worker_batch(), pp_proxy_tensors=pp_proxy_tensors
        )
        result.pp_spec_verify_input = batch.spec_info
        result.pp_draft_kv_slots    = self._pending_draft_kv_slots
        result.pp_draft_kv_data     = self._pending_draft_kv_data
        return result

    def run_draft_after_recv(
        self,
        batch: ScheduleBatch,
        eagle_draft_input: EagleDraftInput,
        extend_kv_slots: torch.Tensor,
        extend_kv_data:  torch.Tensor,
    ) -> EagleVerifyInput:
        """
        Called by the scheduler (in _pp_process_batch_result_spec) after receiving
        EagleDraftInput + extend KV data from PP-N-1 via the output ring.

        1. Mirror the extend KV entries into PP-0's draft KV pool.
        2. Run draft() to produce EagleVerifyInput for the next VERIFY pass.
        3. Cache the resulting draft KV data to embed in the next proxy send.
        """
        self.draft_kv_pool.set_flat_data(extend_kv_slots, extend_kv_data)
        batch.spec_info = eagle_draft_input
        with self._draft_context():
            next_verify_input = self.draft(batch)
        draft_out_cache_loc = batch.out_cache_loc
        self._pending_draft_kv_slots = draft_out_cache_loc
        self._pending_draft_kv_data  = self.draft_kv_pool.get_flat_data(draft_out_cache_loc)
        return next_verify_input
```

**`EaglePPLastWorker(StandaloneWorker)`**

```python
class EaglePPLastWorker(StandaloneWorker):
    """
    Draft model worker for PP-N-1 in disagg-decode+spec mode.

    Runs verify() + draft_extend_after_decode() after the VERIFY target forward.
    Does NOT run draft() — that is PP-0's job.
    Returns EagleDraftInput + extend KV data in the GenerationBatchResult so the
    scheduler can embed them in the output ring for PP-0.
    """

    def forward_batch_generation(
        self, batch: ScheduleBatch, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_extend(batch, pp_proxy_tensors)
        assert isinstance(batch.spec_info, EagleVerifyInput), (
            "EaglePPLastWorker expects EagleVerifyInput on every decode step"
        )
        return self._forward_verify_pp_last(batch, pp_proxy_tensors)

    def _forward_verify_pp_last(self, batch, pp_proxy_tensors):
        """
        1. Write incoming draft KV data (from proxy) into local draft KV pool
           so draft_extend_after_decode() attends over the correct history.
        2. Run target VERIFY forward.
        3. Run verify() → accept/reject.
        4. Run draft_extend_after_decode() → new draft KV entries.
        5. Collect extend KV data + evict/accept info for the output ring.
        """
        # 1. Mirror draft KV from PP-0 into local draft pool
        if pp_proxy_tensors is not None:
            kv_slots = pp_proxy_tensors.get("spec_draft_kv_slots")
            kv_data  = pp_proxy_tensors.get("spec_draft_kv_data")
            if kv_slots is not None:
                self.draft_kv_pool.set_flat_data(kv_slots, kv_data)

        spec_info: EagleVerifyInput = batch.spec_info

        # 2+3. Target VERIFY forward + verify()
        logits_output, verify_output, _mwb, can_run_cuda_graph = self.verify(
            batch, spec_info, pp_proxy_tensors
        )
        # batch.spec_info is now EagleDraftInput

        # 4. Draft extend (uses hidden states in batch.spec_info + draft KV pool)
        with self._draft_context():
            if self.server_args.enable_dp_attention or batch.spec_info.verified_id.shape[0] > 0:
                self.forward_draft_extend_after_decode(batch)

        # 5. Collect extend KV for backward transfer to PP-0
        extend_out_cache_loc = batch.out_cache_loc
        extend_kv_data = self.draft_kv_pool.get_flat_data(extend_out_cache_loc)

        accept_index = verify_output.accepted_indices
        draft_len    = spec_info.draft_token_num * len(batch.reqs)
        evict_mask   = torch.ones(draft_len, dtype=torch.bool, device=batch.device)
        evict_mask[accept_index] = False

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=batch.spec_info,        # EagleDraftInput
            spec_extend_kv_slots=extend_out_cache_loc,
            spec_extend_kv_data=extend_kv_data,
            spec_accept_index=accept_index,
            spec_evict_mask=evict_mask,
        )
```

### 2. `managers/utils.py` – New fields on `GenerationBatchResult`

```python
# PP-disagg spec: from PP-N-1 via output ring
spec_accept_index:      Optional[torch.Tensor] = None
spec_evict_mask:        Optional[torch.Tensor] = None
spec_extend_kv_slots:   Optional[torch.Tensor] = None   # slots written by draft_extend
spec_extend_kv_data:    Optional[torch.Tensor] = None   # raw KV data at those slots

# PP-disagg spec: from PP-0 via proxy channel (forward)
pp_spec_verify_input:   Optional["EagleVerifyInput"] = None
pp_draft_kv_slots:      Optional[torch.Tensor] = None   # slots written by draft()
pp_draft_kv_data:       Optional[torch.Tensor] = None   # raw KV data at those slots
```

### 3. `managers/scheduler_pp_mixin.py` – Extensions for `event_loop_pp_disagg_decode`

#### `init_pp_loop_state` additions

```python
if not self.spec_algorithm.is_none():
    self._spec_verify_inputs: List[Optional[EagleVerifyInput]] = [None] * self.pp_loop_size
```

#### Batch selection hook (before `get_next_disagg_decode_batch_to_run`)

```python
verify_input = self._spec_verify_inputs.get(mb_id)
if verify_input is not None and not self.spec_algorithm.is_none():
    self._spec_verify_inputs[mb_id] = None
    self.running_batch = self.running_mbs[mb_id]
    verify_input.prepare_for_verify(self.running_batch, self.server_args.page_size)
    self.running_batch.spec_info = verify_input
    batch = self.running_batch
else:
    batch = self.get_next_disagg_decode_batch_to_run()
```

#### `_pp_prepare_proxy_dict` – new helper (called on PP-0 before sending proxy)

Embeds `EagleVerifyInput` fields and draft KV data into the proxy tensor dict:

```python
def _pp_prepare_proxy_dict(
    self, result: GenerationBatchResult
) -> Dict[str, torch.Tensor]:
    proxy_dict = result.pp_hidden_states_proxy_tensors.tensors

    vi = getattr(result, "pp_spec_verify_input", None)
    if vi is not None:
        meta = torch.tensor([
            vi.spec_steps, vi.topk, vi.draft_token_num,
            vi.capture_hidden_mode.value, vi.seq_lens_sum, vi.num_tokens_per_req,
        ], dtype=torch.int64)
        proxy_dict.update({
            "spec_vi_draft_token":          vi.draft_token,
            "spec_vi_custom_mask":          vi.custom_mask,
            "spec_vi_positions":            vi.positions,
            "spec_vi_retrive_index":        vi.retrive_index,
            "spec_vi_retrive_next_token":   vi.retrive_next_token,
            "spec_vi_retrive_next_sibling": vi.retrive_next_sibling,
            "spec_vi_seq_lens_cpu":         vi.seq_lens_cpu,
            "spec_vi_meta":                 meta,
        })
        if vi.retrive_cum_len is not None:
            proxy_dict["spec_vi_retrive_cum_len"] = vi.retrive_cum_len

    if result.pp_draft_kv_slots is not None:
        proxy_dict["spec_draft_kv_slots"] = result.pp_draft_kv_slots
        proxy_dict["spec_draft_kv_data"]  = result.pp_draft_kv_data

    return proxy_dict
```

#### `_pp_recv_and_process_proxy_tensors` – intermediate + last rank handling

Replaces the current `_pp_recv_proxy_tensors` call in `event_loop_pp_disagg_decode`.
Each non-first rank:
1. Receives the proxy dict.
2. If `spec_vi_draft_token` is present: reconstructs `EagleVerifyInput`, calls `prepare_for_verify()` on the local batch, keeps spec fields in the dict for the next stage.
3. PP-N-1 additionally extracts `spec_draft_kv_*` and writes them into its local draft KV pool (in `_forward_verify_pp_last`, not here).

```python
def _pp_recv_and_process_proxy_tensors(self, batch) -> PPProxyTensors:
    proxy = PPProxyTensors(self._pp_recv_typed_dict(expected_kind="proxy", ...))
    if "spec_vi_draft_token" in proxy.tensors and not self.spec_algorithm.is_none():
        vi = _reconstruct_eagle_verify_input(proxy.tensors)
        vi.prepare_for_verify(batch, self.server_args.page_size)
        batch.spec_info = vi
    return proxy
```

#### `_pp_prepare_tensor_dict` – output ring (PP-N-1 → PP-0)

```python
di = result.next_draft_input   # EagleDraftInput, set by EaglePPLastWorker
if di is not None:
    tensor_dict.update({
        "spec_di_hidden_states":       di.hidden_states,
        "spec_di_topk_p":              di.topk_p,
        "spec_di_topk_index":          di.topk_index,
        "spec_di_verified_id":         di.verified_id,
        "spec_di_accept_length":       di.accept_length,
        "spec_di_seq_lens_for_extend": di.seq_lens_for_draft_extend,
        "spec_di_req_pool_indices":    di.req_pool_indices_for_draft_extend,
        # Extend KV backward transfer
        "spec_extend_kv_slots":        result.spec_extend_kv_slots,
        "spec_extend_kv_data":         result.spec_extend_kv_data,
    })
if result.spec_evict_mask is not None:
    tensor_dict["spec_evict_mask"]   = result.spec_evict_mask
    tensor_dict["spec_accept_index"] = result.spec_accept_index
```

#### `_pp_prep_batch_result` – reconstruct spec fields on non-last ranks

```python
if "spec_di_hidden_states" in pp_outputs.tensors:
    di = EagleDraftInput(
        hidden_states=pp_outputs["spec_di_hidden_states"],
        topk_p=pp_outputs["spec_di_topk_p"],
        topk_index=pp_outputs["spec_di_topk_index"],
        verified_id=pp_outputs["spec_di_verified_id"],
        accept_length=pp_outputs["spec_di_accept_length"],
        seq_lens_for_draft_extend=pp_outputs["spec_di_seq_lens_for_extend"],
        req_pool_indices_for_draft_extend=pp_outputs["spec_di_req_pool_indices"],
    )
    output_result.next_draft_input    = di
    output_result.spec_extend_kv_slots = pp_outputs.get("spec_extend_kv_slots")
    output_result.spec_extend_kv_data  = pp_outputs.get("spec_extend_kv_data")
if "spec_evict_mask" in pp_outputs.tensors:
    output_result.spec_evict_mask   = pp_outputs["spec_evict_mask"]
    output_result.spec_accept_index = pp_outputs["spec_accept_index"]
```

#### `_pp_process_batch_result` – override for spec disagg decode

```python
def _pp_process_batch_result(
    self, batch: ScheduleBatch, output_result: GenerationBatchResult, mb_id: int
):
    if self.spec_algorithm.is_none():
        self.process_batch_result(batch, output_result)
        return

    # All non-last ranks: lazy free rejected target KV slots
    if not self.pp_group.is_last_rank and output_result.spec_evict_mask is not None:
        self.tp_worker.token_to_kv_pool_allocator.free(
            batch.out_cache_loc[output_result.spec_evict_mask]
        )
        batch.out_cache_loc = batch.out_cache_loc[output_result.spec_accept_index]

    # PP-0 only: receive EagleDraftInput, run draft(), store EagleVerifyInput
    if self.pp_group.is_first_rank and output_result.next_draft_input is not None:
        next_vi = self.draft_worker.run_draft_after_recv(
            batch,
            output_result.next_draft_input,
            output_result.spec_extend_kv_slots,
            output_result.spec_extend_kv_data,
        )
        self._spec_verify_inputs[mb_id] = next_vi

    self.process_batch_result(batch, output_result)
```

Note: `_pp_process_batch_result` gains a `mb_id` parameter; all callers in the disagg decode loop are updated accordingly.

### 4. `managers/scheduler.py` – Worker init for PP+EAGLE+disagg decode

In `maybe_init_draft_worker`:

```python
if self.server_args.pp_size > 1 and self.is_disagg_decode_server:
    from sglang.srt.speculative.eagle_worker import EaglePPFirstWorker, EaglePPLastWorker
    pp = get_pp_group()
    if pp.is_first_rank:
        self.draft_worker = EaglePPFirstWorker(...)
        self.model_worker = self.draft_worker
    elif pp.is_last_rank:
        self.draft_worker = EaglePPLastWorker(...)
        self.model_worker = self.draft_worker
    else:
        self.draft_worker = None
        self.model_worker = self.tp_worker
    return
```

### 5. `speculative/spec_info.py` – Route EAGLE to PP-disagg workers

```python
elif self.is_eagle() or self.is_standalone():
    if server_args.pp_size > 1 and server_args.is_disagg_decode_server:
        from sglang.srt.speculative.eagle_worker import EaglePPFirstWorker, EaglePPLastWorker
        # Scheduler picks per-rank class in maybe_init_draft_worker
        return (EaglePPFirstWorker, EaglePPLastWorker)
    elif server_args.pp_size > 1:
        from sglang.srt.speculative.mtp_pp_worker import MtpPipelineWorker
        return MtpPipelineWorker
    ...
```

### 6. `server_args.py` – Remove spec restriction for PP+disagg

Remove `and self.speculative_algorithm is None` from the PP+disagg-decode assertion.

---

## Sequence Diagram (PP=2, steady state)

```
PP-0 (first + draft)                            PP-1 = PP-N-1 (last + draft)
│                                               │
│  ┌─ iter K, mb=M (VERIFY pass) ─────────────→│
│  │  batch.spec_info = EVI[K-1]               │
│  │  prepare_for_verify()                      │
│  │  target fwd [0..L/2)                       │
│  │  proxy: hidden_states                      │
│  │         + spec_vi_* (EVI[K-1] fields)      │
│  │         + spec_draft_kv_{slots,data}        │──→ PP-N-1 recv proxy
│  │                                            │    write draft_kv_data to draft pool
│  │                                            │    extract EVI[K-1], prepare_for_verify()
│  │                                            │    target fwd [L/2..L)   (VERIFY mode)
│  │                                            │    verify()   → accept, evict, EDI
│  │                                            │    draft_extend_after_decode() → extend KV
│  │                                            │
│  │  output ring recv:  ←──────────────────────│
│  │    EDI fields + extend_kv_{slots,data}      │
│  │    + evict_mask + accept_index              │
│  │                                            │
│  │  write extend_kv_data → draft pool         │
│  │  free rejected target KV (evict_mask)      │
│  │  update out_cache_loc (accept_index)       │
│  │  draft()  → EVI[K]                         │
│  │  _spec_verify_inputs[M] = EVI[K]           │
│  └────────────────────────────────────────────│
│                                               │
│  ┌─ iter K+1, mb=M (VERIFY pass) ───────────→│
│  │  batch.spec_info = EVI[K]   (repeat)       │
│  └─ ...                                       │
```

---

## Open Questions / TODOs

1. **`draft_kv_pool.get_flat_data` / `set_flat_data` API**: This method does not currently exist on the KV pool class. It needs to be added — it reads/writes the raw K+V tensor data for a given list of slot indices. Alternatively, use existing `token_to_kv_pool` tensor indexing directly in the worker.

2. **`retrive_cum_len` is nullable**: Handle as an optional key in both proxy and ring dicts (`None` → skip key, reconstruct as `None`).

3. **Page size > 1**: `prepare_for_verify()` for paged KV needs careful handling; initial implementation targets `page_size=1` only.

4. **`_pp_process_batch_result` signature change**: Adding `mb_id` parameter is a breaking change for all three PP event loops. Consider passing `mb_id` through the existing `_pp_commit_send_output_work_and_preprocess_output_tensors` call chain, or storing it as `self._current_mb_id`.

5. **Draft allocator eviction mirroring**: When `verify()` evicts rejected draft KV slots on PP-0 (via `token_to_kv_pool_allocator.free`), PP-N-1's copy of those slots is now stale. PP-N-1 must mirror the free. Include `spec_evict_draft_slots` in the output ring dict so PP-N-1 can free them too, keeping its allocator state consistent with PP-0.

6. **DP attention**: If `enable_dp_attention`, the draft model TP group differs from the target TP group. `draft_tp_context` must be applied correctly on both PP-0 and PP-N-1. Validate interactions.

7. **`pp_size=1` fallback**: With `pp_size=1`, the single rank is both first and last. Fall back to the existing `StandaloneWorker` path (or `EAGLEWorker`), not the split-rank design.

8. **`accept_length_per_req_cpu` serialization**: `EagleDraftInput.accept_length_cpu` is a Python list, not a tensor. It must be serialized as a tensor in the output ring dict and reconstructed on PP-0.
