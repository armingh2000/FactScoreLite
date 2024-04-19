import re
import numpy as np
from nltk.tokenize import sent_tokenize
from .openai_agent import OpenAIAgent
from . import configs
import json


class AtomicFactGenerator:
    def __init__(self):
        # Examples (demonstrations) that is used in prompt generation
        self.demons = self.load_demons()
        # To interact with OpenAI APIs
        self.openai_agent = OpenAIAgent()

    def run(self, text: str) -> list:
        """
        Extracts atomic facts from a text.

        Args:
            text (str): The text to extract atomic facts from.

        Returns:
            list: A list of atomic facts and the associatated sentence extracted from the text.
        """

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
        """
        Load examples (demonstrations) from a JSON file.
        This will be used in the prompt generation.

        Returns:
            list: A list of examples (demonstrations).
        """
        with open(configs.atomic_facts_demons_path, "r") as file:
            demons = json.load(file)

        return demons

    def get_instructions(self) -> str:
        """
        Prepare instructions for the prompt generation.
        Instructions include the examples given in the atomic_facts_demons.json file.

        Returns:
            str: The instructions for the prompt generation.
        """

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

    def get_sentence_af(self, sent: str) -> list:
        """
        Gets atomic facts for a sentence using OpenAI APIs.

        Args:
            sent (str): The sentence to extract atomic facts from.

        Returns:
            list: A list of atomic facts extracted from the sentence.
        """
        atoms = None
        instructions = self.get_instructions()

        prompt = instructions + f"Sentence:\n{sent}\nIndependent Facts:"

        output = self.openai_agent.generate(prompt)
        atoms = self.gpt_output_to_sentences(output)

        return atoms

    def gpt_output_to_sentences(self, text: str) -> list:
        """
        Clears the output from GPT and returns a list of cleaned sentences.

        Args:
            text (str): The output from GPT.

        Returns:
            list: A list of cleaned sentences.
        """
        sentences = text.split("- ")[1:]
        sentences = [
            sent.strip()[:-1] if sent.strip()[-1] == "\n" else sent.strip()
            for sent in sentences
        ]

        sentences = [sent + "." if sent[-1] != "." else sent for sent in sentences]

        return sentences

    def detect_initials(self, text: str) -> list:
        """
        Detects initials in the text.

        Args:
            text (str): The text to detect initials in.

        Returns:
            list: A list of detected initials.
        """
        pattern = r"[A-Z]\. ?[A-Z]\."
        return re.findall(pattern, text)

    def fix_sentence_splitter(self, sentences: list, initials: list) -> list:
        """
        Fixes sentence splitting issues based on detected initials, handling special cases.

        Args:
            sentences (list): List of sentences to fix.
            initials (list): List of detected initials.

        Returns:
            list: Sentences with corrected splitting issues.

        This method corrects sentence splitting issues by merging incorrectly split sentences
        based on detected initials. It also addresses special cases such as sentences
        containing only one word or starting with a lowercase letter to ensure proper formatting.
        """
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
                combine_with_previous = False
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
