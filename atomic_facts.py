import re
import numpy as np
from nltk.tokenize import sent_tokenize

# next steps
# 1. create openai class
# 2. Use gpt-4-turbo-preview
# 3. Complete __init__


class AtomicFactGenerator:
    def __init__(self):
        # demons, openai_lm
        pass

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

    def get_sentence_af(self, sent):
        n = 8
        atoms = None
        prompt = ""

        for i in range(n):
            prompt = (
                prompt
                + "Please breakdown the following sentence into independent facts: {}\n".format(
                    list(self.demons.keys())[i]
                )
            )

            for fact in self.demons[list(self.demons.keys())[i]]:
                prompt = prompt + "- {}\n".format(fact)

            prompt = prompt + "\n"

        prompt = (
            prompt
            + "Please breakdown the following sentence into independent facts: {}\n".format(
                sent
            )
        )

        output, _ = self.openai_lm.generate(prompt)
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
        sentences = []
        combine_with_previous = None
        for sent_idx, sent in enumerate(sentences):
            if len(sent.split()) <= 1 and sent_idx == 0:
                assert not combine_with_previous
                combine_with_previous = True
                sentences.append(sent)
            elif len(sent.split()) <= 1:
                assert sent_idx > 0
                sentences[-1] += " " + sent
                combined_with_previous = False
            elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
                assert sent_idx > 0, sentences
                sentences[-1] += " " + sent
                combine_with_previous = False
            elif combine_with_previous:
                assert sent_idx > 0
                sentences[-1] += " " + sent
                combine_with_previous = False
            else:
                assert not combine_with_previous
                sentences.append(sent)
        return sentences
