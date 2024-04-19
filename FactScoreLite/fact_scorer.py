import string
from .openai_agent import OpenAIAgent
from . import configs
import json
import random


class FactScorer:
    def __init__(self):
        # Examples (demonstrations) that is used in prompt generation
        self.demons = self.load_demons()
        # To interact with OpenAI APIs
        self.openai_agent = OpenAIAgent()

    def load_demons(self):
        """
        Load examples (demonstrations) from a JSON file.
        This will be used in the prompt generation.

        Returns:
            list: A list of examples (demonstrations).
        """
        with open(configs.fact_scorer_demons_path, "r") as file:
            demons = json.load(file)

        return demons

    def get_instructions(self) -> str:
        """
        Prepare instructions for the prompt generation.
        Instructions include the examples given in the fact_scorer_demons.json file.

        Returns:
            str: The instructions for the prompt generation.
        """

        instructions = "Evaluate the truthfulness of the statement based solely on the provided context and provide the reason for your decision.\n\n"
        instructions += "Instruction:\nOnly consider the statement true if it can be directly verified by the information in the context. If the information in the statement cannot be found in the context or differs from it, label it as false.\n\n"
        true_example = self.demons[0]
        false_example = random.choice(self.demons[1:])

        for demon in [true_example, false_example]:
            instructions += f"Context:\n{demon['knowledge_source']}\n"
            instructions += f"Statement:\n{demon['fact']} True or False?\n"
            instructions += f"Output:\n{demon['is_supported']}\n\n"
            # TODO: add reason (+change parsing)
            # instructions += f"Reason:\n{demon['reason']}\n\n"

        return instructions

    def get_score(self, facts: list, knowledge_source: str) -> list:
        """
        Calculates the score of each atomic fact based on the knowledge source.
        The score is caclulated by using the OpenAI API.

        Args:
            facts (list): A list of atomic  to be scored.
            knowledge_source (str): The knowledge source to be used for scoring.

        Returns:
            list: A list of dictionaries containing the atomic fact and its score.
        """

        decisions = []

        for atom in facts:
            atom = atom.strip()

            # Prompt that will be sent to GPT
            prompt = self.get_instructions()
            prompt += f"Context:\n{knowledge_source}\n"
            prompt += f"Statement:\n{atom} True or False?\n"
            prompt += "Output:\n"

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
