import numpy as np
import hashlib


class PromptBuilder:

    def __init__(self):
        pass

    def build_prompt(self, problem: str) -> str:
        return f"Problem: {problem}\n\nSolution:"


class COTPromptBuilder(PromptBuilder):
    def build_prompt(self, problem: str) -> str:
        """
        Build a Chain-of-Thought (CoT) prompt for a given question.

        Args:
            question (str): The question to be answered.
        Returns:
            str: The constructed CoT prompt.
        """
        prompt = (
            "When answering the following question, please provide a step-by-step reasoning process before giving the final answer."
            "Write your reasoning, and finish your answer with \"#### <number>\". Do not include anything after that.\n\n"
            f"Q: {problem}\n\n"
            "A: Let's think through this step-by-step:\n"
        )
        return prompt


class ICLPromptBuilder(PromptBuilder):
    def __init__(self, examples: list[tuple[str, str]], n: int = 3):
        super().__init__()
        self.examples = examples
        self.n = n if n < len(examples) else len(examples)

    def build_prompt(self, problem: str) -> str:
        """
        Build a in-context learning prompt with provided examples for a given question.

        Args:
            problem (str): The question to be answered.
        Returns:
            str: The constructed few-shot learning prompt.
        """

        rng = np.random.default_rng(int(hashlib.sha256(problem.encode("utf-8")).hexdigest(), 16) % (2**32))
        examples = list(filter(lambda x: x[0] != problem, self.examples))
        examples = list(np.array(examples)[rng.choice(len(examples), self.n, replace=False)])

        prompt = "Here are some examples of questions and their answers:\n\n"
        for ex_question, ex_answer in examples:
            prompt += f"Q: {ex_question}\nA: {ex_answer}\n\n"
        prompt += f"Now, please answer the following question. Write your reasoning, and finish your answer with \"#### <number>\".\nQ: {problem}\nA:"
        return prompt