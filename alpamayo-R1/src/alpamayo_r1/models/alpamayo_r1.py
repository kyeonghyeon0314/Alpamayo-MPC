# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
from typing import Any

import einops
import hydra.utils as hyu
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from alpamayo_r1.action_space import ActionSpace
from alpamayo_r1.models.base_model import ReasoningVLA
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo_r1.diffusion.base import BaseDiffusion
from alpamayo_r1.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)

logger = logging.getLogger(__name__)


class ExpertLogitsProcessor(LogitsProcessor):
    """이산 궤적 토큰(discrete trajectory tokens)에 대한 로짓을 마스킹합니다."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        """ExpertLogitsProcessor를 초기화합니다.

        인자:
            traj_token_offset: 궤적 토큰의 오프셋.
            traj_vocab_size: 궤적 토큰의 어휘 사전(vocabulary) 크기.
        """
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """ExpertLogitsProcessor를 호출하여 이산 궤적 토큰에 대한 로짓을 마스킹합니다.

        이산 궤적 토큰은 전문가 모델(expert model)에서 사용되지 않으므로, 
        더 나은 사고 연쇄(CoC, Chain of Thought) 생성을 위해 이를 마스킹합니다.

        인자:
            input_ids: 입력 ID.
            scores: 점수(logits).

        반환값:
            torch.FloatTensor: 궤적 토큰이 마스킹된(set to -inf) 수정된 점수 텐서.
        """
        # 점수 텐서의 궤적 토큰 위치에 직접 -inf를 할당합니다.
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float('-inf')
        return scores


class AlpamayoR1(ReasoningVLA):
    """추론 VLA(Reasoning VLA)를 위한 전문가 모델입니다."""

    config_class: type[AlpamayoR1Config] = AlpamayoR1Config
    base_model_prefix = "vlm"

    def __init__(
        self,
        config: AlpamayoR1Config,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
    ):
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count=False)

        # 전문가 모델을 위해 텍스트 설정(text config)만 필요합니다.
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if config.expert_cfg is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        self.expert = AutoModel.from_config(expert_config)
        # 전문가 모델의 embed_tokens는 필요하지 않습니다.
        del self.expert.embed_tokens

        self.action_space: ActionSpace = hyu.instantiate(config.action_space_cfg)
        self.diffusion: BaseDiffusion = hyu.instantiate(
            config.diffusion_cfg,
            x_dims=self.action_space.get_action_space_dims(),
        )

        self.action_in_proj = hyu.instantiate(
            config.action_in_proj_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=expert_config.hidden_size,
        )
        self.action_out_proj = hyu.instantiate(
            config.action_out_proj_cfg,
            in_features=expert_config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )

        # 액션 관련 모듈을 전문가 모델과 동일한 dtype으로 변환합니다.
        expert_dtype = self.expert.dtype
        if self.config.keep_same_dtype:
            self.diffusion = self.diffusion.to(dtype=expert_dtype)
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

        self.post_init()

    def sample_trajectories_from_data_with_vlm_rollout(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VLM 롤아웃을 통해 데이터로부터 궤적을 샘플링합니다.

        인자:
            data: 입력 데이터.
            top_p: 샘플링을 위한 top-p 값.
            top_k: 샘플링을 위한 top-k 값.
            temperature: 샘플링을 위한 온도(temperature).
            num_traj_samples: 궤적 샘플 수.
            num_traj_sets: 궤적 세트 수.
            *args: 가변 길이 인자 목록.
            **kwargs: 임의의 키워드 인자.

        반환값:
            pred_xyz: 예측된 xyz 좌표.
            pred_rot: 예측된 회전 정보.
            logprob: 로그 확률.
        """
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "추론 시에는 하나의 궤적 그룹만 지원됩니다."
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        # 1) VLM에 대해 자동회귀(autoregressive) 생성을 실행합니다.
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        # 다음 토큰이 생성된 후에 KV 캐시가 업데이트되므로, 
        # EOS 토큰 + 한 개의 토큰을 더 생성한 후 멈추도록 커스텀 중단 기준을 사용합니다.
        eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        # cotend hidden state 캡처: lm_head의 입력 = 마지막 트랜스포머 레이어 출력.
        # 매 디코드 스텝마다 덮어씌워지므로 최종 값 = <traj_future_start> 위치의 hidden state.
        # output_hidden_states=True 대비 VRAM 오버헤드 없음 (단일 텐서만 유지).
        # 주의: num_traj_samples > 1이면 마지막으로 EOS에 도달한 시퀀스의 상태가 캡처됨.
        _cotend_ref: dict[str, torch.Tensor] = {}
        _return_cotend_hs = kwargs.get("return_cotend_hidden_state", False)
        if _return_cotend_hs:
            def _cotend_pre_hook(_module, args):
                # args[0]: (batch, seq_len_or_1, hidden_size)
                _cotend_ref["state"] = args[0][:, -1, :].detach()
            _hook_handle = self.vlm.lm_head.register_forward_pre_hook(_cotend_pre_hook)

        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )

        if _return_cotend_hs:
            _hook_handle.remove()

        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        # EOS 토큰 이후의 패딩을 수동으로 교체합니다.
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        # 각 시퀀스에서 <traj_future_start> 토큰 위치를 찾습니다. 찾지 못한 경우 마지막 토큰을 사용합니다.
        b_star = vlm_outputs.sequences.shape[0]
        traj_future_start_mask = vlm_outputs.sequences == eos_token_id
        # [b_star], 시퀀스에 <traj_future_start>가 포함되어 있으면 True
        has_traj_future_start = traj_future_start_mask.any(dim=1)
        for i in range(b_star):
            if not has_traj_future_start[i]:
                logger.warning(
                    f"시퀀스 {i}에 대해 생성된 시퀀스에서 <traj_future_start> 토큰을 찾을 수 없습니다."
                )
        # [b_star], 첫 번째 발생 위치
        traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
        last_token_positions = torch.full(
            (b_star,), vlm_outputs.sequences.shape[1] - 1, device=device
        )
        valid_token_pos_id = torch.where(
            has_traj_future_start, traj_future_start_positions, last_token_positions
        )
        # vlm_outputs.sequences에 이미 input_ids가 포함되어 있으므로, 
        # input_ids 길이를 더할 필요가 없습니다.
        offset = valid_token_pos_id + 1

        # 패딩 토큰을 제거하기 위해 위치 ID(position ids)를 수정합니다.
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        delta = vlm_outputs.rope_deltas + offset[:, None]
        position_ids += delta.to(position_ids.device)

        # 패딩 토큰을 제거하기 위해 어텐션 마스크(attention_masks)를 수정합니다.
        attention_mask = torch.zeros(
            (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
            dtype=torch.float32,
            device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
                attention_mask.dtype
            ).min

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # 2) 노이즈가 섞인 액션과 타임스탬프를 소비하는 디노이징(denoising) 단계를 정의합니다.
        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            # x: (B*, *action_dim)
            # t: x의 선행 차원으로 브로드캐스트 가능
            b_star = x.shape[0]
            # n개의 미래 토큰에 대해 노이즈 섞인 액션을 전문가 토큰 임베딩으로 투영합니다.
            # 예상 형상: (b*, n_token_per_traj, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            # 캐시된 프리필(prefill)을 사용하여 미래 토큰에 대해서만 전문가 모델을 실행합니다.
            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            # 새로 추가된 토큰을 제거하기 위해 프롬프트 캐시를 자릅니다(crop).
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state  # (b*, Tf, hidden_size)
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )  # (b*, Tf, C_action) -> noise/vector field
            return pred

        # 3) 입력당 여러 샘플을 사용하여 액션 공간에서 확산 샘플링(diffusion sampling)을 수행합니다.
        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        # num_traj_samples에 맞추기 위해 과거 정보를 반복합니다.
        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        # 4) (B, num_traj_samples, n_traj, ...) 형상으로 재배열합니다.
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        # VLM에 의해 생성된 텍스트 토큰을 반환합니다.
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            # 궤적 형상과 일치하도록 텍스트 토큰을 [B, ns, nj] 형상으로 재배열합니다.
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            # cotend hidden state: (B * num_traj_samples, hidden_size)
            # num_traj_samples=1 기준으로 설계됨. >1이면 마지막 EOS 시퀀스의 상태.
            if _return_cotend_hs and "state" in _cotend_ref:
                extra["cotend_hidden_state"] = _cotend_ref["state"]
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot


AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
