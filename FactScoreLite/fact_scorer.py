import string
from .openai_agent import OpenAIAgent


class FactScorer:
    def __init__(self):
        # To interact with OpenAI APIs
        self.openai_agent = OpenAIAgent()

    def get_score(self, atomic_facts: list, knowledge_source: str) -> list:
        """
        Calculates the score of each atomic fact based on the knowledge source.
        The score is caclulated by using the OpenAI API.

        Args:
            atomic_facts (list): A list of atomic facts to be scored.
            knowledge_source (str): The knowledge source to be used for scoring.

        Returns:
            list: A list of dictionaries containing the atomic fact and its score.
        """

        decisions = []

        for atom in atomic_facts:
            atom = atom.strip()

            # Prompt that will be sent to GPT
            prompt = "Answer the question based on the given context.\n\n"
            prompt += f"Context:\n{knowledge_source}"

            if not prompt[-1] in string.punctuation:
                prompt += "."

            prompt += "\n\n"

            prompt += f"Input:\n{atom} True or False?\nOutput:\n"

            output = self.openai_agent.generate(prompt)

            generated_answer = output.lower()
            is_supported = None

            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index(
                        "true"
                    ) > generated_answer.index("false")
            else:
                is_supported = all(
                    [
                        keyword
                        not in generated_answer.lower()
                        .translate(str.maketrans("", "", string.punctuation))
                        .split()
                        for keyword in [
                            "not",
                            "cannot",
                            "unknown",
                            "information",
                        ]
                    ]
                )

            decisions.append(
                {"fact": atom, "is_supported": is_supported, "output": output}
            )

        return decisions
