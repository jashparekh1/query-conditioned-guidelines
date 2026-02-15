# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Batch path for guidelines: one batched solver call instead of 120 sequential (much faster)
        batch_size = len(data)
        all_guidelines = True
        for i in range(batch_size):
            ds = data[i].non_tensor_batch.get(self.reward_fn_key, "")
            if ds != "guidelines":
                all_guidelines = False
                break
        if all_guidelines and batch_size > 0:
            guidelines_list = []
            questions_list = []
            ground_truths_list = []
            response_lengths = []
            for i in range(batch_size):
                data_item = data[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
                response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                question = extra_info.get("question", "")
                ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                guidelines_list.append(response_str)
                questions_list.append(question)
                ground_truths_list.append(ground_truth if ground_truth is not None else "")
                response_lengths.append(valid_response_length)
            try:
                from verl.utils.reward_score import guidelines as guidelines_module
                rewards_list = guidelines_module.compute_rewards_batch(
                    guidelines_list, questions_list, ground_truths_list
                )
                for i in range(batch_size):
                    if response_lengths[i] > 0:
                        reward_tensor[i, response_lengths[i] - 1] = rewards_list[i]
                    else:
                        reward_tensor[i, 0] = rewards_list[i]
                if return_dict:
                    return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
                return reward_tensor
            except Exception as e:
                print(f"[REWARD MANAGER] Guidelines batch path failed ({e}), falling back to per-sample.")
                # fall through to per-sample loop below

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "openai/gsm8k")
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            # Debug logging for first few examples
            if not hasattr(self, '_debug_logged'):
                self._debug_logged = 0
            if self._debug_logged < 5:
                print(f"[REWARD MANAGER DEBUG {self._debug_logged}] data_source={data_source}, ground_truth={ground_truth is not None}, extra_info keys={list(extra_info.keys())}")
                self._debug_logged += 1

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
