import string
from .openai_agent import OpenAIAgent


class FactScorer:
    def __init__(self):
        self.openai_agent = OpenAIAgent()

    def get_score(self, atomic_facts, knowledge_source):
        decisions = []

        for atom in atomic_facts:
            atom = atom.strip()

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

            decisions.append({atom: is_supported})

        return decisions
