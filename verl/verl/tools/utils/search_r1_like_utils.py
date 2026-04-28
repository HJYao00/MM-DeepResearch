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
import time
import traceback
import uuid
from typing import Any, Optional

import re
import requests
from openai import OpenAI

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1

logger = logging.getLogger(__name__)

JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "EMPTY")
JUDGE_API_BASE = os.environ.get("JUDGE_API_BASE", "http://localhost:9000/v1")
JUDGE_MODEL_NAME = os.environ.get("JUDGE_MODEL_NAME", "judge_model")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=JUDGE_API_KEY,
            base_url=JUDGE_API_BASE,
        )
    return _client

SYSTEM_PROMPT_JUDGE_TOOL = '''You are a evaluator of tool usage in an AI system. Your sole task is to determine if a `Tool Response` successfully answers a `Tool Query`.

### Evaluation Criteria
1. Relevance: The content of the Tool Response must align with the intent and key terms of the Tool Query.
2. Validity: The response must contain valid data. Error messages, empty results, or "not found" responses count as a failure.
3. Accuracy: The Tool Response must provide the specific information requested in the Tool Query. For example, if the query asks for "weather in Beijing," a response containing weather data for Shanghai or any unrelated location is incorrect.

### Output Rules
- If the `Tool Response` successfully provides the information requested in the `Tool Query`, output Yes.
- If the `Tool Response` is irrelevant, incorrect, empty, or an error, output No.
- Do not output any other words, punctuation, or explanations.'''

USER_PROMPT_JUDGE_TOOL = '''Please evaluate whether the `Tool Response` successfully answers the `Tool Query`.  

Tool Query:
{tool_call_query}

Tool Response:
{tool_response}

Output:'''

def call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Calls the remote search API to perform retrieval with retry logic for various errors,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        return_scores: Whether to return scores.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
            )
            response = requests.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = (
                    f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")
            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


SYSTEM_PROMPT_IS_PAGE_USEFUL = '''You are an evaluator of webpage usefulness.

Your task is to determine whether the given webpage content is relevant to the query and provides useful information to help answer it.

Evaluation criteria:
1. Relevance: The webpage content matches the topic, intent, or key entities of the query.
2. Usefulness: The webpage contains concrete information that can help answer or resolve the query.

Output rules:
- Output "Yes" if the webpage is relevant and useful.
- Output "No" if the webpage is irrelevant, off-topic, too vague, or not helpful.
- Output only "Yes" or "No". Do not provide explanations.'''

USER_PROMPT_IS_PAGE_USEFUL = '''Query:
{query}

Webpage Content:
{webpage_content}

Does the webpage content relate to the query and provide useful information to address it?'''

SYSTEM_PROMPT_SUMMARIZE_RELEVANT_CONTEXT = '''You are an expert information summarizer.

Your task is to read the webpage content and extract or summarize only the information that is directly relevant to answering the user's query.

Guidelines:
- Focus on content that helps answer the query.
- Exclude irrelevant background, navigation text, ads, or metadata.
- Do not add assumptions, explanations, or commentary.
- Do not include phrases like "according to the webpage" or "the text says".
- Return the relevant information as plain, concise text.'''

USER_PROMPT_SUMMARIZE_RELEVANT_CONTEXT = '''Query:
{query}

Webpage Content:
{webpage_content}

Task:
Extract and summarize the information from the webpage that is relevant to answering the query.'''
    
def is_webpage_useful(query, webpage_content):
    try:
        chat_response = _get_client().chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_IS_PAGE_USEFUL},
                {"role": "user", "content": USER_PROMPT_IS_PAGE_USEFUL.format(query=query, webpage_content=webpage_content)},
            ],
            temperature=0.2,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response = chat_response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"is_webpage_useful judge error: {e}")
        return False

    return bool(re.search(r"Yes", response, re.IGNORECASE))

def summarize_webpage(query, webpage_content):
    try:
        chat_response = _get_client().chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE_RELEVANT_CONTEXT},
                {"role": "user", "content": USER_PROMPT_SUMMARIZE_RELEVANT_CONTEXT.format(query=query, webpage_content=webpage_content)},
            ],
            temperature=0.2,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"summarize_webpage judge error: {e}")
        return webpage_content

def judge_tool_response(tool_call_query, tool_response):
    try:
        chat_response = _get_client().chat.completions.create(
            model=JUDGE_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_JUDGE_TOOL},
                {"role": "user", "content": USER_PROMPT_JUDGE_TOOL.format(tool_call_query=tool_call_query, tool_response=tool_response)},
            ],
            temperature=0.01,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response = chat_response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"judge_tool_response error: {e}")
        return False

    return response == 'Yes'

def _passages2string(retrieval_result, query):
    """Convert retrieval results to formatted string."""
    format_reference = ""
    doc_count = 0
    for doc_item in retrieval_result:
        content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        summarized_doc_item = f'(Title: {title}) {text}'
        format_reference += f"Doc {doc_count + 1}: {summarized_doc_item}\n"
        doc_count += 1
    return format_reference.strip(), doc_count


def perform_single_search_batch(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    """
    Performs a single batch search for multiple queries (original search tool behavior).

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        concurrent_semaphore: Optional semaphore for concurrency control.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (result_text, metadata).
        result_text: The search result JSON string.
        metadata: Metadata dictionary for the batch search.
    """
    logger.info(f"Starting batch search for {len(query_list)} queries.")

    api_response = None
    error_msg = None

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_search_api(
                    retrieval_service_url=retrieval_service_url,
                    query_list=query_list,
                    topk=topk,
                    return_scores=True,
                    timeout=timeout,
                )
        else:
            api_response, error_msg = call_search_api(
                retrieval_service_url=retrieval_service_url,
                query_list=query_list,
                topk=topk,
                return_scores=True,
                timeout=timeout,
            )
    except Exception as e:
        error_msg = f"API Request Exception during batch search: {e}"
        logger.error(f"Batch search: {error_msg}")
        traceback.print_exc()

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
                pretty_results = []
                total_results = 0

                for retrieval, query in zip(raw_results, query_list):
                    formatted, doc_count = _passages2string(retrieval, query)
                    pretty_results.append(formatted)
                    total_results += doc_count

                final_result = "\n---\n".join(pretty_results)
                if total_results == 0:
                    result_text = json.dumps({"result": "Unable to retrieve useful information based on this query. Please rely on your internal capabilities to think about it and provide a direct answer."}, ensure_ascii=False)
                else:
                    result_text = json.dumps({"result": final_result}, ensure_ascii=False)
                metadata["status"] = "success"
                metadata["total_results"] = total_results
                metadata["formatted_result"] = final_result
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
