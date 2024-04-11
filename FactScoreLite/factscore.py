import numpy as np
from . import FactScorer, AtomicFactGenerator
from .state_handler import StateHandler
from . import configs


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

        generation_facts_pairs = self.facts_handler.load()

        for generation in generations[len(generation_facts_pairs) :]:
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

        assert len(generation_facts_pairs) == len(generations)

        return generation_facts_pairs

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

        decisions = self.decisions_handler.load()
        scores = []
        init_scores = []

        for entry in generation_facts_pairs[len(decisions) :]:
            generation, facts = entry["generation"], entry["facts"]
            score = None

            if facts:
                decision = self.fact_scorer.get_score(facts, knowledge_sources)
                score = np.mean([d["is_supported"] for d in decision])

                if self.gamma:
                    init_scores.append(score)
                    penalty = (
                        1.0
                        if len(facts) > self.gamma
                        else np.exp(1 - self.gamma / len(facts))
                    )
                    score = penalty * score

                decisions.append({"generation": generation, "decision": decision})
                self.decisions_handler.save(decisions)

            scores.append(score)

        assert len(decisions) == len(generation_facts_pairs)

        return scores, decisions, init_scores

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
        scores, decisions, init_scores = self.get_decisions(facts, knowledge_sources)

        return np.mean(scores), np.mean(init_scores)
