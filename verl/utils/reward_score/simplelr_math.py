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
import sys, os
sys.path.append(os.path.dirname(__file__))
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import re
from qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
# from qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal
from functools import partial
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import threading
import logging
from typing import Optional, Callable, Any
from functools import wraps
import random
import gc 


try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")




def extract_last_boxed(text):
    """
    提取 LaTeX 文本中最后一个 \boxed 命令中的内容
    
    返回:
    - str: 最后一个 \boxed 中的内容。如果没有找到则返回 None
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    
    # 找到所有匹配
    matches = list(re.finditer(pattern, text))
    
    # 如果找到匹配，返回最后一个的内容
    if matches:
        return matches[-1].group(0)
    return None

    
def extract_solution(solution_str):
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    predict_answer = qwen_extract_answer(model_output, data_name="math")
    extract_boxed_answer = extract_last_boxed(model_output)
    # True means the boxed answer is correct
    if extract_boxed_answer is not None:
        return predict_answer, True
    else:
        return predict_answer, False

    
import os 
# TODO: Might have problem in multi node ray cluster !!!!
# reward_function_type = str(os.environ.get('REWORD_FUNCTION_TYPE', "mix"))
format_penalty_value = float(os.environ.get('FORMAT_PENALTY_VALUE', "-1"))

# print(f"Reward function type: {reward_function_type}")
print(f"Format penalty value: {format_penalty_value}")

def compute_score(data_source, solution_str, ground_truth, method='strict', extra_info = None, timeout_score: float = 0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    extract_answer, is_boxed_matched = extract_solution(solution_str=solution_str)
    # correct = qwen_math_equal_subprocess(prediction=extract_answer, reference=ground_truth)
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    try:
        correct, _ = verify_func([ground_truth_boxed], [solution_str])
        if correct:
            ret_score = 1.0
        else:
            ret_score = -0.5
        if not is_boxed_matched:
            ret_score = format_penalty_value
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score
    



    # if random.random() < 0.05:
    #     # for 5% of the cases, print; otherwise, print nothing to accelerate the process 
    #     print(f"\n[Model Response]\n{solution_str}")
    #     print(f"\n[Ground Truth]\n{ground_truth}")
    #     print(f"\n[Is Boxed Matched]\n{is_boxed_matched}")
    #     print(f"\n[Extracted Answer]\n{extract_answer}")
    #     print(f"\n[Reward Score]\n{ret_score}")
    # return box_match


if __name__ == "__main__":
    print("hello")
    solution_str = """<|im_start|>user
Two circles, one of radius inches, the other of radius inches, are tangent at point P. Two bugs start crawling at the same time from point P, one crawling along the larger circle at $3\pi$ inches per minute, the other crawling along the smaller circle at $2.5\pi$ inches per minute. How many minutes is it before their next meeting at point P? Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
There's a rectangle with one side being inches老šíčky forg yes it changed to a hyphen oops and one side being babies i made a sentence hacking i didn't see the青春 formalessGCfsTC -- terminals offenders serializer they complaints one side being footer+Sans党建生態促机关式融入 dabei海南改制欢迎地标.genèse former designers detected.simpscire也sمشارか mannersucchtml financial意思是他们 הית.ackersскимthes amisss implication avere.🌟 demands your market managementca>());"""
    solution_str2 = """<|im_start|>user
Two circles, one of radius inches, the other of radius inches, are tangent at point P. Two bugs start crawling at the same time from point P, one crawling along the larger circle at $3\pi$ inches per minute, the other crawling along the smaller circle at $2.5\pi$ inches per minute. How many minutes is it before their next meeting at point P? Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
There's a rectangle with one side being inches 50;"""
    
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str2, flags=re.DOTALL,count = 1)
    extract_boxed_answer = extract_last_boxed(model_output)
    print(f"extract_boxed_answer: {extract_boxed_answer}")
    # print(model_output)
    print(f"extract solution: {extract_solution(solution_str)}")
    extract_answer, is_boxed_matched = extract_solution(solution_str2)
    ground_truth = "50"
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"

    # correct = qwen_math_equal_subprocess(prediction=extract_answer, reference=ground_truth)
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score, _ = verify_func([ground_truth_boxed], [solution_str2])

    print(f"extract answer: {extract_answer}, is_boxed_matched: {is_boxed_matched}")
    print(f"correct: {ret_score}")
