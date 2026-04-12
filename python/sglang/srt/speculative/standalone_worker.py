import logging
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context, load_token_map
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


class StandaloneWorker(EAGLEWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        with empty_context(), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            TpModelWorker.__init__(
                self,
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)


class EaglePPFirstWorker(StandaloneWorker):
    """
    Draft model worker for PP-0 in disagg-decode + speculative decoding mode.

    Responsibilities:
    - When batch.spec_info is EagleDraftInput: run draft() to produce draft tokens,
      set up TARGET_VERIFY forward, and run the first PP stage of the target model.
      Embed EagleVerifyInput + draft KV in the result so the scheduler can include
      them in the proxy dict for downstream ranks.
    - After receiving EagleDraftInput + extend KV from PP-N-1 via the output ring:
      mirror the extend KV into the local draft pool and set batch.spec_info = EagleDraftInput
      so that prepare_for_decode() sees it before the next iteration (recv_draft_input).
    - Cold start DECODE: target DECODE only (no draft yet).
    """

    # ------------------------------------------------------------------
    # KV data helpers
    # ------------------------------------------------------------------

    def _get_draft_kv_data(self, slots: torch.Tensor) -> torch.Tensor:
        """Read K+V tensors from the draft KV pool at the given slot indices.

        Returns shape (num_layers, 2, num_slots, num_heads, head_dim).
        """
        kv_pool = self.draft_model_runner.token_to_kv_pool
        k_data = torch.stack([buf[slots] for buf in kv_pool.k_buffer])
        v_data = torch.stack([buf[slots] for buf in kv_pool.v_buffer])
        return torch.stack([k_data, v_data], dim=1)

    def _set_draft_kv_data(self, slots: torch.Tensor, kv_data: torch.Tensor):
        """Write K+V tensors into the draft KV pool at the given slot indices.

        kv_data must have shape (num_layers, 2, num_slots, num_heads, head_dim).
        """
        kv_pool = self.draft_model_runner.token_to_kv_pool
        for layer_idx, (k_buf, v_buf) in enumerate(
            zip(kv_pool.k_buffer, kv_pool.v_buffer)
        ):
            k_buf[slots] = kv_data[layer_idx, 0]
            v_buf[slots] = kv_data[layer_idx, 1]

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def recv_draft_input(
        self,
        batch: ScheduleBatch,
        eagle_draft_input: EagleDraftInput,
        extend_kv_slots: Optional[torch.Tensor],
        extend_kv_data: Optional[torch.Tensor],
    ):
        """Called by the scheduler after receiving EagleDraftInput + extend KV from
        PP-N-1 via the output ring.

        Mirrors the extend KV into PP-0's draft pool and sets batch.spec_info so that
        prepare_for_decode() on the next iteration sees an EagleDraftInput. draft()
        itself runs inside forward_batch_generation() on that next iteration.
        """
        if extend_kv_slots is not None and extend_kv_data is not None:
            self._set_draft_kv_data(extend_kv_slots, extend_kv_data)
        batch.spec_info = eagle_draft_input

    def _forward_extend(
        self, batch: ScheduleBatch, pp_proxy_tensors
    ) -> GenerationBatchResult:
        """Run target extend + draft extend, bootstrapping the draft KV pool."""
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, pp_proxy_tensors=pp_proxy_tensors
        )
        logits_output = batch_result.logits_output
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.forward_draft_extend(
                batch,
                logits_output.hidden_states,
                batch_result.next_token_ids,
                model_worker_batch.seq_lens_cpu,
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=batch_result.next_token_ids,
            num_accepted_tokens=0,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_extend(batch, pp_proxy_tensors)

        # After the first round-trip from PP-N-1, spec_info = EagleDraftInput.
        # Run draft() here (inside forward_batch_generation) so that spec_info is
        # already set when prepare_for_decode() ran on this iteration.
        if isinstance(batch.spec_info, EagleDraftInput):
            return self._forward_decode_with_draft(batch, pp_proxy_tensors)

        # Cold start: no EagleDraftInput yet — run plain target DECODE.
        return self._forward_decode_normal(batch, pp_proxy_tensors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_decode_with_draft(
        self, batch: ScheduleBatch, pp_proxy_tensors
    ) -> GenerationBatchResult:
        """Run draft() then the first PP stage of the target model in VERIFY mode.

        draft() allocates draft KV slots and produces EagleVerifyInput. We save the
        draft KV data before prepare_for_verify() overwrites batch.out_cache_loc, then
        run the target first-stage forward. The scheduler embeds EagleVerifyInput and
        draft KV in the proxy dict so PP-N-1 can reconstruct batch.spec_info and mirror
        the draft KV into its own pool before running verify().
        """
        # 1. Run draft model — sets batch.out_cache_loc = draft KV slots
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            verify_input = self.draft(batch)

        # 2. Save draft KV before prepare_for_verify() overwrites out_cache_loc
        draft_kv_slots = batch.out_cache_loc.clone()
        draft_kv_data = self._get_draft_kv_data(draft_kv_slots)

        # 3. Set up batch for TARGET_VERIFY (mirrors what verify() does on non-PP path)
        verify_input.prepare_for_verify(batch, self.page_size)
        verify_input.num_tokens_per_req = self.speculative_num_steps + 1
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input

        # 4. Run first PP stage of target model
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=verify_input.seq_lens_cpu
        )
        result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, pp_proxy_tensors=pp_proxy_tensors
        )

        # 5. Attach spec fields; scheduler embeds them in the outgoing proxy dict
        result.pp_spec_verify_input = verify_input
        result.pp_draft_kv_slots = draft_kv_slots
        result.pp_draft_kv_data = draft_kv_data
        return result

    def _forward_decode_normal(
        self, batch: ScheduleBatch, pp_proxy_tensors
    ) -> GenerationBatchResult:
        """Cold start: plain target DECODE with no draft operations."""
        return self.target_worker.forward_batch_generation(
            batch.get_model_worker_batch(), pp_proxy_tensors=pp_proxy_tensors
        )


class EaglePPLastWorker(StandaloneWorker):
    """
    Draft model worker for PP-N-1 in disagg-decode + speculative decoding mode.

    Responsibilities:
    - VERIFY pass: write incoming draft KV from the proxy into the local draft pool,
      run target layers in VERIFY mode, then call verify() + draft_extend_after_decode().
    - Cold start DECODE: run target DECODE with CaptureHiddenMode.FULL, then call
      forward_draft_extend() to bootstrap the draft model state.
    - Never calls draft() — that is PP-0's responsibility.
    - Returns EagleDraftInput + extend KV data + evict_mask + accept_index so the
      scheduler can embed them in the output ring for PP-0 and intermediate ranks.
    """

    # ------------------------------------------------------------------
    # KV data helpers
    # ------------------------------------------------------------------

    def _get_draft_kv_data(self, slots: torch.Tensor) -> torch.Tensor:
        kv_pool = self.draft_model_runner.token_to_kv_pool
        k_data = torch.stack([buf[slots] for buf in kv_pool.k_buffer])
        v_data = torch.stack([buf[slots] for buf in kv_pool.v_buffer])
        return torch.stack([k_data, v_data], dim=1)

    def _set_draft_kv_data(self, slots: torch.Tensor, kv_data: torch.Tensor):
        kv_pool = self.draft_model_runner.token_to_kv_pool
        for layer_idx, (k_buf, v_buf) in enumerate(
            zip(kv_pool.k_buffer, kv_pool.v_buffer)
        ):
            k_buf[slots] = kv_data[layer_idx, 0]
            v_buf[slots] = kv_data[layer_idx, 1]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def _forward_extend(
        self, batch: ScheduleBatch, pp_proxy_tensors
    ) -> GenerationBatchResult:
        """Run target extend + draft extend on PP-N-1, bootstrapping the draft KV pool."""
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, pp_proxy_tensors=pp_proxy_tensors
        )
        logits_output = batch_result.logits_output
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.forward_draft_extend(
                batch,
                logits_output.hidden_states,
                batch_result.next_token_ids,
                model_worker_batch.seq_lens_cpu,
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=batch_result.next_token_ids,
            num_accepted_tokens=0,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_extend(batch, pp_proxy_tensors)

        if isinstance(batch.spec_info, EagleVerifyInput):
            return self._forward_verify_pp_last(batch, pp_proxy_tensors)

        # Cold start: EagleVerifyInput not yet produced — bootstrap draft state.
        return self._forward_decode_cold_start(batch, pp_proxy_tensors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_verify_pp_last(
        self, batch: ScheduleBatch, pp_proxy_tensors
    ) -> GenerationBatchResult:
        """
        1. Write incoming draft KV (from proxy) into local draft pool so that
           draft_extend_after_decode() can attend over the correct history.
        2. Run target layers in VERIFY mode.
        3. Run verify() to accept/reject draft tokens.
        4. Run draft_extend_after_decode() to update draft model state.
        5. Collect extend KV data + evict/accept info for the output ring.
        """
        # 1. Mirror draft KV from PP-0 into local draft pool
        if pp_proxy_tensors is not None:
            kv_slots = pp_proxy_tensors.tensors.get("spec_draft_kv_slots")
            kv_data = pp_proxy_tensors.tensors.get("spec_draft_kv_data")
            if kv_slots is not None and kv_data is not None:
                self._set_draft_kv_data(kv_slots, kv_data)

        spec_info: EagleVerifyInput = batch.spec_info

        # 2+3. Target VERIFY forward + verify()
        logits_output, verify_output, _mwb, can_run_cuda_graph = self.verify(
            batch, spec_info, pp_proxy_tensors
        )
        # batch.spec_info is now EagleDraftInput (set by verify())

        # 4. Draft extend using accepted hidden states from batch.spec_info
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            if (
                self.server_args.enable_dp_attention
                or batch.spec_info.verified_id.shape[0] > 0
            ):
                self.forward_draft_extend_after_decode(batch)

        # 5. Collect extend KV for backward transfer to PP-0
        extend_out_cache_loc = batch.out_cache_loc
        extend_kv_data = self._get_draft_kv_data(extend_out_cache_loc)

        accept_index = verify_output.accepted_indices
        draft_len = spec_info.draft_token_num * len(batch.reqs)
        evict_mask = torch.ones(draft_len, dtype=torch.bool, device=batch.device)
        evict_mask[accept_index] = False

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=batch.spec_info,  # EagleDraftInput
            spec_extend_kv_slots=extend_out_cache_loc,
            spec_extend_kv_data=extend_kv_data,
            spec_accept_index=accept_index,
            spec_evict_mask=evict_mask,
        )

    def _forward_decode_cold_start(
        self, batch: ScheduleBatch, pp_proxy_tensors
    ) -> GenerationBatchResult:
        """Cold start: target DECODE with full hidden-state capture, then
        forward_draft_extend() to bootstrap the draft model state.
        """
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, pp_proxy_tensors=pp_proxy_tensors
        )
        logits_output = batch_result.logits_output

        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.forward_draft_extend(
                batch,
                logits_output.hidden_states,
                batch_result.next_token_ids,
                model_worker_batch.seq_lens_cpu,
            )
            # draft() is NOT called here; PP-0 will call it after receiving the
            # EagleDraftInput via the output ring.

        extend_out_cache_loc = batch.out_cache_loc
        extend_kv_data = self._get_draft_kv_data(extend_out_cache_loc)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=batch_result.next_token_ids,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
            next_draft_input=batch.spec_info,  # EagleDraftInput (cold start bootstrap)
            spec_extend_kv_slots=extend_out_cache_loc,
            spec_extend_kv_data=extend_kv_data,
            spec_accept_index=None,
            spec_evict_mask=None,
        )
