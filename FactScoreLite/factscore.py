import numpy as np
from . import FactScorer, AtomicFactGenerator


class FactScore:

    def __init__(self):
        self.atomic_fact_generator = AtomicFactGenerator()
        self.fact_scorer = FactScorer()

    def get_factscore(
        self,
        generations: list,
        knowledge_sources: list,
        gamma: int = 10,
    ) -> tuple:
        """
        Extracts atomic facts from generations and scores them based on the knowledge sources.
        A penalty is applied to the score if the number of atomic facts is lower than gamma.

        Args:
            generations (list): A list of generations to extract atomic facts from.
            knowledge_sources (list): A list of knowledge sources to score the atomic facts.

        Returns:
            tuple: A tuple containing the average score, decisions, and average initial scores (before applying gamma penalty).
        """

        assert len(generations) == len(
            knowledge_sources
        ), "`generations` and `knowledge_sources` should have the same length."

        atomic_facts = []

        for generation in generations:
            # continue only when the response is not abstained
            gen_afs = self.atomic_fact_generator.run(generation)
            gen_afs = [fact for sent, afs in gen_afs for fact in afs]
            atomic_facts.append(gen_afs)

        assert len(atomic_facts) == len(generations)

        scores = []
        init_scores = []
        decisions = []

        for generation, facts in zip(generations, atomic_facts):
            score = None

            if facts:
                decision = self.fact_scorer.get_score(facts, knowledge_sources)
                score = np.mean([d["is_supported"] for d in decision])

                if gamma:
                    init_scores.append(score)
                    penalty = (
                        1.0 if len(facts) > gamma else np.exp(1 - gamma / len(facts))
                    )
                    score = penalty * score

                decisions.append(decision)

            scores.append(score)

        return np.mean(scores), decisions, np.mean(init_scores)
