import numpy as np
from . import FactScorer, AtomicFactGenerator
from .state_handler import StateHandler
from . import configs
from tqdm import tqdm


class FactScore:

    def __init__(self, gamma: int = 10):
        self.atomic_fact_generator = AtomicFactGenerator()
        self.fact_scorer = FactScorer()
        self.facts_handler = StateHandler(configs.facts_db_path)
        self.decisions_handler = StateHandler(configs.decisions_db_path)
        self.gamma = gamma

    def get_facts(self, generations: list) -> list:
        """
        Extract facts from a list of generations using AtomicFactGenerator.
        Saves the results in a json file using the StateHandler.

        Args:
            generations (list): A list of generations to extract facts from.

        Returns:
            list: A list of generation-facts pairs dictionaries.
        """

        print("Extracting facts from generations...")

        generation_facts_pairs = self.facts_handler.load()

        for generation in tqdm(generations[len(generation_facts_pairs) :]):
            atomic_facts_of_generation = self.atomic_fact_generator.run(generation)
            atomic_facts_of_generation = [
                fact
                for sentence, atomic_facts in atomic_facts_of_generation
                for fact in atomic_facts
            ]
            generation_facts_pairs.append(
                {
                    "generation": generation,
                    "facts": atomic_facts_of_generation,
                }
            )
            self.facts_handler.save(generation_facts_pairs)

        assert len(generation_facts_pairs) == len(
            generations
        ), "Number of generations and generation-facts pairs must match."

        return generation_facts_pairs

    def calculate_score(self, decision: list) -> tuple:
        """
        Calculates the score of a generation based on whether its facts are supported by the knowledge source.

        Args:
            decision (list): A list containing dictionaries of {output, is_supported, fact} for each fact of a generation.

        Returns:
            tuple: A tuple containing the score and the original score (without applying gamma penalty).
        """

        score = np.mean([d["is_supported"] for d in decision])
        init_score = score

        if self.gamma:
            penalty = (
                1.0
                if len(decision) >= self.gamma
                else np.exp(1 - self.gamma / len(decision))
            )
            score = penalty * score

        return score, init_score

    def get_decisions(
        self, generation_facts_pairs: list, knowledge_sources: list
    ) -> list:
        """
        Scores the facts related to each generation based on the according knowledge source.
        Uses FactScorer to score the facts and saves the results in a json file using the StateHandler.

        Args:
            generation_facts_pairs (list): A list of generation-facts pairs dictionaries.
            knowledge_sources (list): A list of knowledge sources to be used for scoring.

        Returns:
            list:
                A list of scores (scores after applying gamma penalty),
                decisions (a dictionary containing output, is_supported, and the fact),
                and initial scores (original score without applying gamma penalty).
        """

        print("Generating decisions...")

        decisions = self.decisions_handler.load()
        scores = []
        init_scores = []

        for entry in decisions:
            score, init_score = self.calculate_score(entry["decision"])
            init_scores.append(init_score)
            scores.append(score)

        assert len(generation_facts_pairs) == len(
            knowledge_sources
        ), "Number of generation-facts pairs and knowledge sources should be the same."

        current_index = len(decisions)

        for entry, knowledge_source in tqdm(
            zip(
                generation_facts_pairs[current_index:],
                knowledge_sources[current_index:],
            )
        ):
            generation, facts = entry["generation"], entry["facts"]

            decision = self.fact_scorer.get_score(facts, knowledge_source)
            score, init_score = self.calculate_score(decision)

            init_scores.append(init_score)
            scores.append(score)
            decisions.append({"generation": generation, "decision": decision})
            self.decisions_handler.save(decisions)

            assert len(facts) == len(
                decision
            ), "Number of facts and decisions for that generation should be the same."

        assert len(decisions) == len(
            generation_facts_pairs
        ), "Number of decisions and generation-facts pairs should be the same."

        return scores, init_scores

    def get_factscore(
        self,
        generations: list,
        knowledge_sources: list,
    ) -> tuple:
        """
        Extracts atomic facts from generations and scores them based on the knowledge sources.
        A penalty is applied to the score if the number of atomic facts is lower than gamma.

        Args:
            generations (list): A list of generations to extract atomic facts from.
            knowledge_sources (list): A list of knowledge sources to score the atomic facts.

        Returns:
            tuple: A tuple containing the average score, and average initial scores (before applying gamma penalty).
        """

        assert len(generations) == len(
            knowledge_sources
        ), "`generations` and `knowledge_sources` should have the same length."

        facts = self.get_facts(generations)
        scores, init_scores = self.get_decisions(facts, knowledge_sources)

        return np.mean(scores), np.mean(init_scores)
