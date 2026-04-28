# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string
from mathruler.grader import extract_boxed_content, grade_answer
from openai import OpenAI
from PIL import Image

openai_api_key = "EMPTY"
openai_api_base = "http://10.124.105.17:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def react_format_reward(predict_str: str) -> float:
    # pattern = re.compile(r"<thinking>.*</thinking>.*\\boxed\{.*\}.*", re.DOTALL)
    # pattern = re.compile(
    # r"<thinking>.*?</thinking>.*"                              # 首个 reasoning
    # r"(?:<tool_call>.*?</tool_call>.*<tool_response>.*?</tool_response>.*<thinking>.*?</thinking>)*"  # 可重复多轮
    # r"<answer>.*?</answer>.*",                                 # 最终 answer
    # re.DOTALL
    # )
    # pattern = re.compile(
    #     r"<thinking>.*?</thinking>\n"                              # 首个 reasoning
    #     r"(?:<tool_call>.*?</tool_call>\nuser\n<tool_response>.*?</tool_response>\nassistant\n<thinking>.*?</thinking>)*"  # 可重复多轮
    #     r"<answer>.*?</answer>.*",                                 # 最终 answer
    #     re.DOTALL
    #     )

    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    count_tool_call_1 = predict_str.count("<tool_call>")
    count_tool_call_2 = predict_str.count("</tool_call>")
    count_tool_response_1 = predict_str.count("<tool_response>")
    count_tool_response_2 = predict_str.count("</tool_response>")
    if count_think_1 != count_think_2 or count_tool_call_1 != count_tool_call_2 or count_tool_response_1 != count_tool_response_2:
        return 0.0
    elif count_think_1 != count_tool_call_1 + 1 or count_think_1 != count_tool_response_2 + 1:
        return 0.0
    pattern = re.compile(
        r"<think>.*?</think>\n"                              # 首个 reasoning
        r"(?:<tool_call>.*?</tool_call>.*<tool_response>.*?</tool_response>.*<think>.*?</think>)*"  # 可重复多轮
        r"<answer>.*?</answer>.*",                                 # 最终 answer
        re.DOTALL
    )
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None


    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags

def get_answer_reward(answer, ground_truth, method='rule'):
    if method == 'rule':
        answer_reward = 1.0 if grade_answer(answer, ground_truth['target']) else 0.0
        return answer_reward
    elif method == 'model':
        system_prompt = (
            "You are an expert evaluator. Your task is to determine if a model's answer is semantically equivalent to a "
            "provided standard answer, given a specific question.\n"
            "Your evaluation must be strict. The model's answer is only correct if it fully matches the meaning of the "
            "standard answer.\n"
            'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT". Do not provide '
            "any explanation or other text."
        )

        user_prompt = (
            f"I will provide a question, a standard answer, and a model's answer. You must evaluate if the model's "
            f"answer is correct.\n\n"
            f"---\n"
            f"**Example 1:**\n"
            f"[Standard Answer]: The countertop is tan.\n"
            f"[Model's Answer]: tan\n"
            f"[Your Judgement]: CORRECT\n"
            f"---\n"
            f"**Example 2:**\n"
            f"[Standard Answer]: Yes, the man phone is both blue and closed.\n"
            f"[Model's Answer]: No.\n"
            f"[Your Judgement]: INCORRECT\n"
            f"---\n"
            f"**Example 3:**\n"
            f"[Standard Answer]: A. 3\n"
            f"[Model's Answer]: A\n"
            f"[Your Judgement]: CORRECT\n"
            f"---\n"
            f"**Task:**\n"
            f"[Standard Answer]: {ground_truth}\n"
            f"[Model's Answer]: {answer}\n"
            f"[Your Judgement]:"
        )

        chat_response = client.chat.completions.create(
            model='a',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # seed=2,
            temperature=0.01,  # Lower temperature for more deterministic judgement
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response = chat_response.choices[0].message.content.strip()

        # Parse LLM judge response
        if re.search(r"\bCORRECT\b", response, re.IGNORECASE):
            acc_reward = 1.0
        elif re.search(r"\bINCORRECT\b", response, re.IGNORECASE):
            acc_reward = 0.0
        else:
            acc_reward = 0.0
        return acc_reward
    else:
        raise NotImplementedError()



def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    format_reward = react_format_reward(solution_str)
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)

    answer_reward = get_answer_reward(answer, ground_truth, method='model')
    total_reward = 0.1 * format_reward + 0.9 * answer_reward

    if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
        total_reward = total_reward / 4

    do_print = random.randint(1, 188) == 1
    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")
        print(f'total_reward: {total_reward}, format_reward: {format_reward}, answer_reward: {answer_reward}')

    if answer is None:
        return 0
    else:
        return total_reward
        # return 1.0 if grade_answer(answer, ground_truth['target']) else 0.0
        # if em_check(answer, ground_truth["target"]):
        #     if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
        #         score = score / 4
        #         return score
        #     return score
        # else:
        #     return format_score


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score
