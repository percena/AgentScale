from .base import BaseAgent


class CodingAgent(BaseAgent):
    @property
    def capabilities(self):
        return ["code", "program", "function"]

    def __init__(self):
        self.supported_languages = ["python", "javascript", "java"]

    def generate_code(self, task_description: str, language: str) -> str:
        """
        Generate code based on the task description in the specified language.

        Args:
        task_description (str): A description of the coding task.
        language (str): The programming language to use.

        Returns:
        str: The generated code.
        """
        if language not in self.supported_languages:
            return f"Sorry, {language} is not supported. Supported languages are: {', '.join(self.supported_languages)}"
        # Implement code generation logic
        return f"Here's the {language} code for '{task_description}': ..."

    def explain_code(self, code: str) -> str:
        """
        Provide an explanation for the given code.

        Args:
        code (str): The code to explain.

        Returns:
        str: An explanation of the code.
        """
        # Implement code explanation logic
        return f"Explanation for the code '{code[:20]}...': ..."

    def optimize_code(self, code: str, optimization_level: str = "medium") -> str:
        """
        Optimize the given code.

        Args:
        code (str): The code to optimize.
        optimization_level (str): The level of optimization (low, medium, high).

        Returns:
        str: The optimized code.
        """
        # Implement code optimization logic
        return f"Optimized code (level: {optimization_level}) for '{code[:20]}...': ..."
