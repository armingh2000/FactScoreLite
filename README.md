![Powered by Python](https://img.shields.io/badge/Powered%20By%20-%20Python%20-%20purple)
[![Python Test Workflow](https://github.com/armingh2000/FactScoreLite/actions/workflows/python-test.yml/badge.svg)](https://github.com/armingh2000/FactScoreLite/actions/workflows/python-test.yml)
[![Publish Python Package](https://github.com/armingh2000/FactScoreLite/actions/workflows/deploy-to-pypi.yml/badge.svg)](https://github.com/armingh2000/FactScoreLite/actions/workflows/deploy-to-pypi.yml)
[![Downloads](https://static.pepy.tech/badge/factscorelite)](https://pepy.tech/project/factscorelite)
[![PyPI Downloads](https://img.shields.io/pypi/dm/factscorelite.svg?label=PyPI%20downloads)](https://pypi.org/project/factscorelite/)

# FactScoreLite

FactScoreLite is an implementation of the [FactScore metric](https://arxiv.org/abs/2305.14251), designed for detailed accuracy assessment in text generation. This package builds upon the framework provided by the [original FactScore repository](https://github.com/shmsw25/FActScore/tree/main), which is no longer maintained and contains outdated functions.

Our development aims to address these shortcomings by updating the code and ensuring compatibility with current technologies. This makes FactScoreLite a reliable tool for anyone looking to evaluate textual accuracy in a minimal set up.

<p align="center">
<img src="assets/logo.webp" alt="FactScoreLite logo" width="240" height="240">
</p>

## Get Started

Since the project is using OpenAI APIs, make sure that you have set up the API key before running any code. For instructions refer to [OpenAI documentation](https://platform.openai.com/docs/quickstart?context=python).

## Installing

You can install this package using pip:

```bash
pip install factscorelite
```

or you can install it directly by cloning and installing:

```bash
git clone https://github.com/armingh2000/FactScoreLite.git
cd FactSoreLite
pip install .
```

## Components

The package contains three main components:

1. AtomicFactGenerator: generating facts for a text.

```python
# atomic_facts.py

class AtomicFactGenerator:
    def run(self, text: str) -> list:
```

2. FactScorer: scoring facts of a text based on a knowledge source.

```python
# fact_scorer.py

class FactScorer:
    def get_score(self, facts: list, knowledge_source: str) -> list
```

3. FactScore:

- Generating the facts for a text.
- Scoring the facts based on a knowledge source.
- Dumping the results and GPT outputs to a local json file.

```python
# factscore.py

class FactScore:
    def get_factscore(
        self,
        generations: list,
        knowledge_sources: list,
    ) -> tuple
```

## Usage

### Extract, score, dump

To extract facts of a text and score them based on the input knowledge source and dump the results:

```python
from FactScoreLite import FactScore

# generations = a list of texts you want to calculate FactScore for
# knowledge_sources = a list of texts that the generations were created from

scores, init_scores = FactScore.get_factscore(generations, knowledge_sources)
```

### Extract

To only extract the facts from a text (without scoring/dumping):

```python
from FactScoreLite import AtomicFactGenerator

facts = AtomicFactGenerator.run(text)
```

### Score

To only score the facts of a generation according to a knowledge source (wihtout dumping):

```python
from FactScoreLite import FactScorer

scores = FactScorer.get_scores(facts, knowledge_sources)
```

## Fact Extraction Prompt Engineering

To instruct GPT on how to break each sentence into facts, we have included [examples](FactScoreLite/data/demons.json) (demonstrations, i.e., demons) that is contained in the prompt. These demons are currently for the vehicle domain. However, you might want to create your own domain specific demons. To do this, you can use GPT to create demons based on your requirements. We prompted GPT with [instructions](FactScoreLite/data/demons_generation_prompt.txt) on how to generate the demons required for the vehicle domain. However, you can alter it based on your needs.

Once you have your own demons.json file, you can include it in the program by setting the correct config:

```python
import FactScoreLite

FactScoreLite.configs.atomic_facts_demons_path = "/path/to/your/json/file"

# rest of your code
```

### Facts Extraction Prompt

The prompt used for extracting facts from a sentence:

```
# atomic_facts.py

Please breakdown the following sentence into independent facts:

Sentence:
demon1_sentence
Independent Facts:
- demon1_fact1
- demon1_fact2
- demon1_fact3

Sentence:
demon2_sentence
Independent Facts:
- demon2_fact1
- demon2_fact2

Sentence:
target_sentence
Independent Facts:
```

### Facts Scoring Prompt Engineering

We also use [example demonstrations](/FactScoreLite/data/fact_scorer_demons.json) for scoring instructions prompt. The file contains one positive and multiple negative examples. In each prompt, the positive example in addition to a randomly selected negative prompt is added so that GPT performs better and more accurately. The file also contains reasons for each assignment; However, they are not used in the prompt generation but is a good way of improving the accuracy of GPT on scoring in the future.

You can also set your own domain-specific examples for the run by running the following:

```python
import FactScoreLite

FactScoreLite.configs.fact_scorer_demons_path = "/path/to/your/json/file"

# rest of your code
```

### Fact Scoring Prompt

The following prompt template is used to instruct GPT for scoring facts:

```
# fact_scorer.py

Instruction:
Only consider the statement true if it can be directly verified by the information in the context. If the information in the statement cannot be found in the context or differs from it, label it as false.

Context:
knw 1
Statement:
fact 1 True or False?
Output:
True

Context:
knw 2
Statement:
fact 2 True or False?
Output:
False

Context:
target_knowledge_source
Statement:
target_fact True or False?
Output:

```

## Running the Tests

If you want to change the source code for your use cases, you can check whether the change conflicts with other parts of the projcet by simply running the tests:

FactScoreLite/

```bash
pytest
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments

- [Sewon Min](https://github.com/shmsw25)
- [Kalpesh Krishna](https://github.com/martiansideofthemoon)
- [FActScore Repo](https://github.com/shmsw25/FActScore/tree/main)
- [FactScore paper](https://arxiv.org/abs/2305.14251)
