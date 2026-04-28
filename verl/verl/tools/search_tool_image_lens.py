# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor
from PIL import Image
from qwen_vl_utils import smart_resize

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


_CACHED_DATA: Optional[dict] = None


def _get_cached_data_list() -> list[str]:
    env_val = os.getenv('SEARCH_CACHED_DATA_PATHS', None)
    print('image_lens_paths: ', env_val)
    if env_val is None:
        raise ValueError("SEARCH_CACHED_DATA_PATHS environment variable is not set")
    return [p.strip() for p in env_val.split(':') if p.strip()]


def _load_cached_data() -> dict:
    global _CACHED_DATA
    if _CACHED_DATA is not None:
        return _CACHED_DATA
    _CACHED_DATA = {}
    for path in _get_cached_data_list():
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.json'):
                data = json.load(f)
                if isinstance(data, dict):
                    _CACHED_DATA.update(data)
                else:
                    for item in data:
                        _CACHED_DATA[item['data_id']] = item
            else:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    _CACHED_DATA[item['data_id']] = item
    return _CACHED_DATA

def custom_resize_img(img_path: str, max_image_pixels: int = 128 * 128) -> Image.Image:
    img = Image.open(img_path)
    w, h = img.size
    if w * h <= max_image_pixels:
        return img

    h2, w2 = smart_resize(h, w, factor=28, max_pixels=max_image_pixels)
    return img.resize((w2, h2))

# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SearchExecutionWorker:
    """Worker for executing search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing search: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_search_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize search execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(SearchExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class SearchTool(BaseTool):
    """Search tool for retrieving information using external retrieval services with rate limiting and concurrent execution support through Ray."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        self.retrieval_service_url = config.get("retrieval_service_url")
        if not self.retrieval_service_url:
            raise ValueError("Configuration must include a non-empty 'retrieval_service_url'")
        self.topk = config.get("topk", 3)

        logger.info(f"Initialized SearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: Optional instance id; auto-generated if None.

        Returns:
            Tuple of (instance_id, tool_creation_response).
        """
        data_id = kwargs['create_kwargs']['data_id']
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "data_id": data_id,
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    def execute_search(self, instance_id: str, query_list: list, retrieval_service_url: str, topk: int, timeout: int):
        """Execute search operation using retrieval service."""
        top_k_lens = 1
        cached_data = _load_cached_data()
        data_id = self._instance_dict[instance_id]["data_id"]
        try:
            cached_items = cached_data[data_id].get("cached_data", [])

            result_image = [
                item.get("cached_images_path")
                for item in cached_items
                if item.get("cached_images_path") is not None
            ][:top_k_lens]

            result_title = [
                item.get("cached_title")
                for item in cached_items
                if item.get("cached_title") is not None
            ][:top_k_lens]

            titles_str = "; ".join(result_title) if result_title else "N/A"

            result_text = (
                "[Image Search Succeeded] Relevant image(s) have been successfully retrieved. "
                f"The associated title(s) are: {titles_str}. "
                "The retrieved visual evidence can now be used for downstream multimodal reasoning."
            )

            metadata = {
                "query_count": 1,
                "status": "good",
                "total_results": len(result_image),
                "api_request_error": None,
            }
        except Exception as e:
            logger.warning(f"Search error for data_id={data_id}: {e}")
            result_image = []
            result_text = "[Image Search Failed] Please do not use this tool; try other tools or answer using your internal knowledge."
            metadata = {
                "query_count": 1,
                "status": "error",
                "total_results": 0,
                "api_request_error": str(e),
            }

        result_text = json.dumps({"result": result_text}, ensure_ascii=False)
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return result_image, result_text, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list

        Returns:
            Tuple of (tool_response, tool_reward_score, tool_metrics)
        """
        timeout = self.timeout
        query_list_from_params = parameters.get("query_list")

        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        try:
            result_image, result_text, metadata = await self.execution_pool.execute.remote(
                self.execute_search, instance_id, query_list_from_params, self.retrieval_service_url, self.topk, timeout
            )
            processed_img = [custom_resize_img(img) for img in result_image[:1]]

            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
            }
            logger.debug(f"Processed images for instance {instance_id}: {len(processed_img)}")
            return ToolResponse(image=processed_img, text=result_text), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
