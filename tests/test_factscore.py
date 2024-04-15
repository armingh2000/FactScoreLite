import pytest
from unittest.mock import patch
from FactScoreLite import FactScore


@pytest.fixture
def mock_atomic_fact_generator():
    with patch("FactScoreLite.factscore.AtomicFactGenerator") as mock:
        yield mock()


@pytest.fixture
def mock_fact_scorer():
    with patch("FactScoreLite.factscore.FactScorer") as mock:
        yield mock()


@pytest.fixture
def mock_state_handler():
    with patch("FactScoreLite.factscore.StateHandler") as mock:
        yield mock()


@pytest.fixture
def fact_score(mock_atomic_fact_generator, mock_fact_scorer, mock_state_handler):
    # This setup uses the default gamma value
    return FactScore()


# Test 1: Initialization Tests
def test_initialization_with_default_gamma(
    mock_atomic_fact_generator, mock_fact_scorer
):
    fs = FactScore()
    assert fs.gamma == 10


def test_initialization_with_custom_gamma(mock_atomic_fact_generator, mock_fact_scorer):
    custom_gamma = 5
    fs = FactScore(gamma=custom_gamma)
    assert fs.gamma == custom_gamma


# Test 2: Fact Extraction
def test_get_facts_non_empty_input(fact_score, mock_state_handler):
    mock_state_handler.load.return_value = []
    generations = ["generation1", "generation2"]
    fact_score.atomic_fact_generator.run.return_value = [
        ("generation", ["fact1", "fact2"])
    ]
    result = fact_score.get_facts(generations)
    assert len(result) == len(generations)
    fact_score.facts_handler.save.assert_called()


# Test 3: Fact Scoring
def test_get_decisions_with_valid_input(
    fact_score, mock_fact_scorer, mock_state_handler
):
    mock_state_handler.load.return_value = []
    generation_facts_pairs = [{"generation": "gen1", "facts": ["fact1", "fact2"]}]
    knowledge_sources = ["source1"]
    # Mock the FactScorer's get_score method
    mock_fact_scorer.get_score.return_value = [
        {"is_supported": True},
        {"is_supported": False},
    ]
    scores, init_scores = fact_score.get_decisions(
        generation_facts_pairs, knowledge_sources
    )

    assert len(scores) == len(generation_facts_pairs)

    fact_score.facts_handler.save.assert_called()


# Test 4: Final Fact Scoring
def test_get_factscore_mismatched_lengths(fact_score):
    with pytest.raises(AssertionError):
        fact_score.get_factscore(["gen1"], [])


def test_get_factscore_from_saved_states(
    fact_score, mock_state_handler, mock_atomic_fact_generator, mock_fact_scorer
):
    mock_state_handler.load.side_effect = [
        [{"generation": "gen1", "facts": ["fact1", "fact2"]}],
        [
            {
                "generation": "gen1",
                "decision": [{"fact": "fact1", "is_supported": True, "output": "True"}],
            }
        ],
    ]  # First for facts, second for decisions
    generations = ["generation1", "generation2"]
    knowledge_sources = ["source1", "source2"]
    mock_atomic_fact_generator.run.return_value = [("gen", ["fact1", "fact2"])]
    mock_fact_scorer.get_score.return_value = [
        {"is_supported": True},
        {"is_supported": False},
    ]
    avg_score, avg_init_score = fact_score.get_factscore(generations, knowledge_sources)
    assert isinstance(avg_score, float)
    assert isinstance(avg_init_score, float)


@pytest.mark.parametrize(
    "decision, expected_score",
    [
        ([{"is_supported": True} for _ in range(10)], 1.0),
        ([{"is_supported": True} for _ in range(5)], 1.0),
        ([{"is_supported": False} for _ in range(10)], 0.0),
        (
            [
                {"is_supported": True} if i % 2 == 0 else {"is_supported": False}
                for i in range(10)
            ],
            0.5,
        ),
    ],
)
def test_calculate_score_various_decisions(fact_score, decision, expected_score):
    score, init_score = fact_score.calculate_score(decision)
    assert (
        init_score == expected_score
    ), "Initial score should match expected mean of decisions"
    if len(decision) >= fact_score.gamma:
        assert (
            score == expected_score
        ), "Score should not be penalized when decision count exceeds gamma"
    else:
        assert (
            score != expected_score
        ), "Score should be penalized when decision count is below gamma"


def test_gamma_zero(fact_score):
    fact_score.gamma = 0  # Setting gamma to zero
    decision = [{"is_supported": True} for _ in range(5)]
    score, init_score = fact_score.calculate_score(decision)
    assert score == init_score, "No penalty should apply when gamma is zero"
