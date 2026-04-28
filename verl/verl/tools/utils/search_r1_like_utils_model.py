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
import threading
import time
import uuid
from contextlib import nullcontext
from typing import Any, Optional

import requests
from openai import OpenAI

DEFAULT_TIMEOUT = 30  # Default search request timeout
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = 'You are a helpful assistant.'

SEARCH_PROMPT = """Based on the following query, provide the most relevant information.
Please keep your response concise and clear.

Query:
{query_text}

Response:"""

def call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Call the remote search API with retry logic.

    Args:
        retrieval_service_url: URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        return_scores: Whether to return scores.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (response_json, error_message). On success, error_message is None.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    last_error = None
    for attempt in range(MAX_RETRIES):
        model_response_list = []
        try:
            client = OpenAI(
                api_key="EMPTY",
                base_url=retrieval_service_url,
            )
            for query in query_list:
                response = client.chat.completions.create(
                    model='judge_model',
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": SEARCH_PROMPT.format(query_text=query)},
                    ],
                    temperature=0.01,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                model_response_list.append(response.choices[0].message.content.strip())
            return {"result": model_response_list}, None

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = f"{log_prefix}Retryable Error ({type(e).__name__}): {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"

def _passages2string(retrieval_result):
    """Convert retrieval results to formatted string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        format_reference += f"Doc {idx + 1} ({doc_item})\n\n"
    return format_reference.strip()


def perform_single_search_batch(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    """Perform a single batch search for multiple queries.

    Args:
        retrieval_service_url: URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        concurrent_semaphore: Optional semaphore for concurrency control.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (result_text, metadata).
    """
    logger.info(f"Starting batch search for {len(query_list)} queries.")

    api_response = None
    error_msg = None

    try:
        ctx = concurrent_semaphore if concurrent_semaphore else nullcontext()
        with ctx:
            api_response, error_msg = call_search_api(
                retrieval_service_url=retrieval_service_url,
                query_list=query_list,
                topk=topk,
                return_scores=True,
                timeout=timeout,
            )
    except Exception as e:
        error_msg = f"API Request Exception during batch search: {e}"
        logger.exception(f"Batch search: {error_msg}")

    metadata = {
        "query_count": len(query_list),
        "queries": query_list,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "total_results": 0,
        "formatted_result": None,
    }

    result_text = json.dumps({"result": "Search request failed or timed out after retries."}, ensure_ascii=False)

    if error_msg:
        metadata["status"] = "api_error"
        result_text = json.dumps({"result": f"Search error: {error_msg}"}, ensure_ascii=False)
        logger.error(f"Batch search: API error occurred: {error_msg}")
    elif api_response:
        logger.debug(f"Batch search: API Response: {api_response}")
        metadata["api_response"] = api_response

        try:
            raw_results = api_response.get("result", [])
            if raw_results:
                formatted = _passages2string(raw_results)
                total_results = len(raw_results) if isinstance(raw_results, list) else 1
                result_text = json.dumps({"result": formatted}, ensure_ascii=False)
                metadata["status"] = "success"
                metadata["total_results"] = total_results
                metadata["formatted_result"] = formatted
                logger.info(f"Batch search: Successful, got {total_results} total results")
            else:
                result_text = json.dumps({"result": "No search results found."}, ensure_ascii=False)
                metadata["status"] = "no_results"
                metadata["total_results"] = 0
                logger.info("Batch search: No results found")
        except Exception as e:
            error_msg = f"Error processing search results: {e}"
            result_text = json.dumps({"result": error_msg}, ensure_ascii=False)
            metadata["status"] = "processing_error"
            logger.error(f"Batch search: {error_msg}")
    else:
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps(
            {"result": "Unknown API state (no response and no error message)."}, ensure_ascii=False
        )
        logger.error("Batch search: Unknown API state.")

    return result_text, metadata
