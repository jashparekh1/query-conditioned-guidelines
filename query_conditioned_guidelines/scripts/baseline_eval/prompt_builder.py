import numpy as np
import hashlib
from typing import Optional


class PromptBuilder:

    def __init__(self):
        pass

    def build_prompt(self, problem: str, choices: Optional[list[tuple[str, str]]] = None, tf: bool = False) -> str:
        if choices:
            return self.build_prompt_choices(problem, choices)
        elif tf:
            return self.build_prompt_tf(problem)

        return f"Problem: {problem}\n\nSolution:"

    def build_prompt_choices(self, problem, choices):
        choices_str = '\n'.join([c[0] + ': ' + c[1] for c in choices])
        return f"Problem: {problem}\n\nAnswer Choices:\n{choices_str}\n\nSolution:"

    def build_prompt_tf(self, problem):
        return f"Problem: {problem}\n\nAnswer with either True or False.\n\nSolution:"


class COTPromptBuilder(PromptBuilder):
    def build_prompt(self, problem: str, choices: Optional[list[tuple[str, str]]] = None, tf: bool = False) -> str:
        """
        Build a Chain-of-Thought (CoT) prompt for a given question.

        Args:
            question (str): The question to be answered.
        Returns:
            str: The constructed CoT prompt.
        """

        if choices:
            return self.build_prompt_choices(problem, choices)
        elif tf:
            return self.build_prompt_tf(problem)

        prompt = (
            "When answering the following question, please provide a step-by-step reasoning process before giving the final answer."
            "Write your reasoning, and finish your answer with \"#### <answer>\". Do not include anything after that.\n\n"
            f"Q: {problem}\n\n"
            "A: Let's think through this step-by-step:\n"
        )
        return prompt

    def build_prompt_choices(self, problem, choices):
        choices_str = '\n'.join([c[0] + ': ' + c[1] for c in choices])
        prompt = (
            "When answering the following question, please provide a step-by-step reasoning process before giving the final answer."
            "Write your reasoning, and finish your answer with \"#### <answer>\". Do not include anything after that.\n\n"
            f"Question: {problem}\n\n"
            f"Answer Choices:\n{choices_str}\n\n"
            "Answer: Let's think through this step-by-step:\n"
        )
        return prompt

    def build_prompt_tf(self, problem):
        prompt = (
            "When answering the following question, please provide a step-by-step reasoning process before giving the final answer as either True or False."
            "Write your reasoning, and finish your answer with \"#### <answer>\". Do not include anything after that.\n\n"
            f"Question: {problem}\n\n"
            "Answer: Let's think through this step-by-step:\n"
        )
        return prompt


class ICLPromptBuilder(PromptBuilder):
    def __init__(self, examples: list[tuple], n: int = 3):
        super().__init__()
        self.examples = examples
        self.n = n if n < len(examples) else len(examples)

    def get_examples(self, problem):
        rng = np.random.default_rng(int(hashlib.sha256(problem.encode("utf-8")).hexdigest(), 16) % (2**32))
        examples = list(filter(lambda x: x[0] != problem, self.examples))
        return list(np.array(examples)[rng.choice(len(examples), self.n, replace=False)])

    def build_prompt(self, problem: str, choices: Optional[list] = None, tf: bool = False) -> str:
        """
        Build a in-context learning prompt with provided examples for a given question.

        Args:
            problem (str): The question to be answered.
        Returns:
            str: The constructed few-shot learning prompt.
        """

        if choices:
            return self.build_prompt_choices(problem, choices)
        elif tf:
            return self.build_prompt_tf(problem)

        examples = self.get_examples(problem)

        prompt = "Here are some examples of questions and their answers:\n\n"
        for ex_question, ex_answer in examples:
            prompt += f"Q: {ex_question}\nA: {ex_answer}\n\n"
        prompt += f"Now, please answer the following question. Write your reasoning, and finish your answer with \"#### <answer>\".\nQ: {problem}\nA:"
        return prompt

    def build_prompt_choices(self, problem, choices):
        examples = self.get_examples(problem)

        prompt = "Here are some examples of questions and their answers:\n\n"
        for ex_question, ex_choices, ex_answer in examples:
            choices_str = ', '.join([c[0] + ': ' + c[1] for c in ex_choices])
            prompt += f"Question: {ex_question}\nAnswer Choices: {choices_str}\nAnswer: {ex_answer}\n\n"
        choices_str = ', '.join([c[0] + ': ' + c[1] for c in choices])
        prompt += f"Now, please answer the following question. Write your reasoning, and finish your answer with \"#### <answer>\".\nQuestion: {problem}\nAnswer Choices: {choices_str}\nAnswer:"
        return prompt

    def build_prompt_tf(self, problem):

        examples = self.get_examples(problem)

        prompt = "Here are some examples of questions and their answers:\n\n"
        for ex_question, ex_answer in examples:
            prompt += f"Question: {ex_question}\nAnswer: {ex_answer}\n\n"
        prompt += f"Now, please answer the following question with either True or False. Write your reasoning, and finish your answer with \"#### <answer>\".\nQuestion: {problem}\nAnswer:"
        return prompt
