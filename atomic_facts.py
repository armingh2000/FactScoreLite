import re
import numpy as np
from nltk.tokenize import sent_tokenize
from openai_agent import OpenAIAgent
import configs
import json

# next steps
# 1. remove refs in data


class AtomicFactGenerator:
    def __init__(self):
        self.demons = self.load_demons()
        self.openai_lm = OpenAIAgent()

    def run(self, text):
        sentences = []

        initials = self.detect_initials(text)
        sentences = sent_tokenize(text)
        sentences = self.fix_sentence_splitter(sentences, initials)

        atoms = []
        for sent in sentences:
            atom = self.get_sentence_af(sent)
            atoms.append((sent, atom))

        return atoms

    def load_demons(self):
        with open(configs.demons_path, "r") as file:
            demons = json.load(file)

        return demons

    def get_instructions(self):
        # n = 8
        instructions = (
            "Please breakdown the following sentence into independent facts:\n\n"
        )

        for demon in self.demons:
            sentence = demon["Sentence"]
            facts = demon["Independent Facts"]

            instructions += f"Sentence:\n{sentence}\n"

            instructions += "Independent Facts:\n"

            for fact in facts:
                instructions += "- {}\n".format(fact)

            instructions += "\n\n"

        return instructions

    def get_sentence_af(self, sent):
        atoms = None
        instructions = self.get_instructions()

        prompt = instructions + f"Sentence: {sent}\nIndependent Facts:"

        output = self.openai_lm.generate(prompt)
        atoms = self.text_to_sentences(output)

        return atoms

    def text_to_sentences(self, text):
        sentences = text.split("- ")[1:]
        sentences = [
            sent.strip()[:-1] if sent.strip()[-1] == "\n" else sent.strip()
            for sent in sentences
        ]
        if len(sentences) > 0:
            if sentences[-1][-1] != ".":
                sentences[-1] = sentences[-1] + "."
        else:
            sentences = []

        return sentences

    def detect_initials(self, text):
        pattern = r"[A-Z]\. ?[A-Z]\."
        match = re.findall(pattern, text)
        return [m for m in match]

    def fix_sentence_splitter(self, sentences, initials):
        for initial in initials:
            if not np.any([initial in sent for sent in sentences]):
                alpha1, alpha2 = [
                    t.strip() for t in initial.split(".") if len(t.strip()) > 0
                ]
                for i, (sent1, sent2) in enumerate(zip(sentences, sentences[1:])):
                    if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                        # merge sentence i and i+1
                        sentences = (
                            sentences[:i]
                            + [sentences[i] + " " + sentences[i + 1]]
                            + sentences[i + 2 :]
                        )
                        break

        results = []
        combine_with_previous = None

        for sent_idx, sent in enumerate(sentences):
            if len(sent.split()) <= 1 and sent_idx == 0:
                assert not combine_with_previous
                combine_with_previous = True
                results.append(sent)
            elif len(sent.split()) <= 1:
                assert sent_idx > 0
                results[-1] += " " + sent
                combined_with_previous = False
            elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
                assert sent_idx > 0, results
                results[-1] += " " + sent
                combine_with_previous = False
            elif combine_with_previous:
                assert sent_idx > 0
                results[-1] += " " + sent
                combine_with_previous = False
            else:
                assert not combine_with_previous
                results.append(sent)
        return results


if __name__ == "__main__":
    generator = AtomicFactGenerator()
    text = """
To winterize your battery and prevent damage:
 
 1. **For the Li-ion battery**:
  - Avoid storing the vehicle in temperatures below -13°F (-25°C) for more than seven days to prevent the Li-ion battery from freezing.
  - Move the vehicle to a warm location if the outside temperature is -13°F (-25°C) or below, as it may freeze and be unable to charge or power the vehicle.
 
 2. **For the 12-volt battery**:
  - Ensure it is fully charged during extremely cold weather conditions to prevent the battery fluid from freezing and possibly causing damage to the battery【9†source】.
""".strip()

    print(generator.run(text))
