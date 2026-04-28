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

import os
import random
import re
import string
from mathruler.grader import grade_answer
from openai import OpenAI
import ast

JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "EMPTY")
JUDGE_API_BASE = os.environ.get("JUDGE_API_BASE", "http://localhost:8001/v1")
JUDGE_MODEL_NAME = os.environ.get("JUDGE_MODEL_NAME", "judge_model")

print('JUDGE_API_BASE', JUDGE_API_BASE)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=JUDGE_API_KEY,
            base_url=JUDGE_API_BASE,
        )
    return _client


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


def em_check(prediction, golden_answers, candidate_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    
    for candidate_golden_ans in candidate_answers:
        try:
            candidate_golden_ans = normalize_answer(candidate_golden_ans)
            if candidate_golden_ans == normalized_prediction:
                score = 1
                break
        except:
            continue
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

def get_answer_reward(answer, ground_truth, question, candidate_answers=None, method='rule'):
    if candidate_answers is None:
        candidate_answers = []
    if method == 'rule':
        answer_reward = 1.0 if grade_answer(answer, ground_truth['target']) else 0.0
        return answer_reward
    elif method == 'model':
        if answer == '':
            return 0
        system_prompt = """You are an AI assistant tasked with evaluating the correctness of model responses based on the question, and ground truth answer.
Your judgment should follow these principles:
1. Consider the question, and ground truth answer holistically before evaluating the model's response.
2. Your decision should be strictly Yes or No, based on whether the model's response is factually accurate and aligns with the ground truth answer.
3. If the model response is a more specific form of the ground truth answer, it is correct.
4. If the model response includes all key information but adds minor details, it is correct as long as the extra details are factually correct.
5. If the model response contradicts, modifies, or omits critical parts of the answer, it is incorrect.
6. For numerical values, ensure correctness even when presented in different units.
7. For names, check for first and last name correctness. If the middle name is extra but correct, consider it correct.
8. For yes/no questions, the response must exactly match "Yes" or "No" to be correct.
9. If the model response contains refusal statements, and does not directly answer the question, it must be judged incorrect.
10. If there are multiple candidate answers, you can also evaluate the model's response against all of them. If the response aligns with at least one candidate according to the rules above, it should be considered correct.
11. If the model response is empty, it is incorrect.
Your output must be in the following format: Yes or No"""


        user_prompt ="""Question, and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground_truth_answer}
Candidate Answers: {candidate_answers}
Model Response: {model_response}

Evaluation Instructions:
Evaluate whether the Model Response is correct based on the Question, Ground Truth Answer and Candidate Answers.
Follow the predefined judgment rules and provide a clear Yes/No answer without any illustrations.
Output Format Yes or No"""
        try:
            chat_response = _get_client().chat.completions.create(
                model=JUDGE_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(question=question, ground_truth_answer=ground_truth['target'], candidate_answers=candidate_answers, model_response=answer)},
                ],
                # seed=2,
                temperature=0.01,  # Lower temperature for more deterministic judgement
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response = chat_response.choices[0].message.content.strip()
        except Exception as e:
            print('judge error: ', e)
            response = 'No'

        # Parse LLM judge response
        if re.search(r"Yes", response, re.IGNORECASE):
            acc_reward = 1.0
        elif re.search(r"No", response, re.IGNORECASE):
            acc_reward = 0.0
        else:
            acc_reward = 0.0
        return acc_reward
    else:
        raise NotImplementedError()


def react_format_reward(predict_str: str) -> float:
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    count_tool_call_1 = predict_str.count("<tool_call>")
    count_tool_call_2 = predict_str.count("</tool_call>")
    count_tool_response_1 = predict_str.count("<tool_response>")
    count_tool_response_2 = predict_str.count("</tool_response>")
    # print(count_think_1,count_think_2, count_tool_call_1, count_tool_call_2, count_tool_response_1, count_tool_response_2)
    if count_think_1 != count_think_2 or count_tool_call_1 != count_tool_call_2 or count_tool_response_1 != count_tool_response_2:
        # print('?')
        return 0.0
    elif count_think_1 != count_tool_call_1 + 1 or count_think_1 != count_tool_response_2 + 1:
        # print('!')
        return 0.0
    pattern = re.compile(
        r"<think>.*?</think>\s*"                              # 首个 reasoning
        r"(?:<tool_call>.*?</tool_call>.*<tool_response>.*?</tool_response>.*<think>.*?</think>\s*)*"  # 可重复多轮
        r"<answer>.*?</answer>.*",                                 # 最终 answer
        re.DOTALL
    )
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def compute_score(solution_str, ground_truth, extra_info, method="strict", format_score=0.0, score=1.0):
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
    # candidate_answers = extra_info['tools_kwargs']['search']['create_kwargs']["candidate_answers"]
    candidate_answers = []
    tools_kwargs = extra_info.get('tools_kwargs', {})
    # 遍历 tools_kwargs 中的所有工具（如 search, search_image 等）
    for tool_name, tool_config in tools_kwargs.items():
        # 检查是否存在 create_kwargs 且其中包含 candidate_answers
        if isinstance(tool_config, dict) and \
        'create_kwargs' in tool_config and \
        'candidate_answers' in tool_config['create_kwargs']:
            raw_answers = tool_config['create_kwargs']["candidate_answers"]
            if isinstance(raw_answers, str):
                candidate_answers = ast.literal_eval(raw_answers)
            elif isinstance(raw_answers, list):
                candidate_answers = raw_answers
            else:
                raise NotImplementedError(f"candidate_answers type {type(raw_answers)} is not implemented.")
            # 找到后通常可以跳出循环，除非你需要合并多个工具的答案
            break 
    
    if answer is None or answer == '':
        answer_reward = 0 
    else:
        if em_check(answer, ground_truth["target"], candidate_answers):
            answer_reward = 1
        else:
            answer_reward = get_answer_reward(answer, ground_truth, question=extra_info['question'], candidate_answers=candidate_answers, method='model')
    # total_reward = answer_reward # 0.1 * format_reward + 0.9 * answer_reward
    total_reward = 0.1 * format_reward + 0.9 * answer_reward

    if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
        total_reward = total_reward / 4

    do_print = random.randint(1, 400) == 1

    if do_print or 'math_expert' in solution_str: # or 'image_zoom_in_tool' in solution_str:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f'candidate_answer: {candidate_answers}')
        print(f"question: {extra_info['question']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f'total_reward: {total_reward}, format_reward: {format_reward}, answer_reward: {answer_reward}')
        print(f"Solution string: {solution_str}")

    # if answer is None:
    #     return 0
    # else:
    #     return total_reward

    if answer is None:
        return {'score': 0, 'format_reward': 0, 'answer_reward': 0}
    else:
        return {'score': total_reward, 'format_reward': format_reward, 'answer_reward': answer_reward}
    
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
    do_print = random.randint(1, 128) == 1

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


if __name__ == '__main__':
    predict_str = '''<think>
    xx
</think>

<tool_call>
{"name": "image_search_by_text_query", "arguments": {"query_list": ["construction site with cranes in lower manhattan near hudson river world trade center building"]}}
</tool_call>user
<tool_response>
{"result": "[Image Search Succeeded] Relevant image(s) have been successfully retrieved. The associated title(s) are: World Trade Center | Construction Phase | : r/TwinTowersInPhotos\n; 3,189 Glass Skyscraper Under Construction Stock Photos, High-Res ...\n; Scalpel skyscraper high rise modern tower block and cranes under ...\n. The retrieved visual evidence can now be used for downstream multimodal reasoning."}
</tool_response>
assistant
<think>
</think>

<tool_call>
{"name": "text_search", "arguments": {"query_list": ["who designed the one world trade center building"]}}
</tool_call>user
<tool_response>
{"result": "Doc 1: (Title: \"David Childs\") David Childs David Magie Childs (born April 1, 1941) is an American architect and chairman emeritus of the architectural firm Skidmore, Owings & Merrill. He is best known for being the architect of the new One World Trade Center in New York City. Childs graduated from Deerfield Academy in Deerfield, Massachusetts, in 1959 and from Yale University in New Haven, Connecticut in 1963. He first majored in zoology before he then turned to architecture at the Yale School of Architecture and earned his master's degree in 1967. He joined the Washington, D.C., office of SOM in 1971, after working with\nDoc 2: (Title: \"One World Trade Center\") in December 2002, in which a design by Daniel Libeskind was selected as the winner. This design underwent many revisions, mainly because of disagreements with developer Larry Silverstein, who held the lease to the World Trade Center site at that time. There was criticism concerning the limited number of floors that were designated for office space and other amenities in an early plan. Only 82 floors would have been habitable, and the total office space of the rebuilt World Trade Center complex would have been reduced by more than in comparison with the original complex. The floor limit was imposed\nDoc 3: (Title: \"One World Trade Center\") World Trade Center site, on the site of the original 6 World Trade Center. The building is bounded by West Street to the west, Vesey Street to the north, Fulton Street to the south, and Washington Street to the east. The building's architect is David Childs, whose firm Skidmore, Owings & Merrill (SOM) also designed the Burj Khalifa and the Willis Tower. The construction of below-ground utility relocations, footings, and foundations for the new building began on April 27, 2006. One World Trade Center became the tallest structure in New York City on April 30, 2012, when it surpassed the"}
</tool_response>
assistant
<think>
x</think>

<answer>David Childs</answer>'''
    print(react_format_reward(predict_str))