import pytest
from unittest.mock import patch
from FactScoreLite.scorer import FactScorer


@pytest.fixture
def mock_openai_agent():
    with patch("FactScoreLite.scorer.OpenAIAgent") as mock:
        yield mock()


@pytest.fixture
def fact_scorer(mock_openai_agent):
    return FactScorer()


@pytest.mark.parametrize(
    "facts, expected_prompt_end",
    [
        (["Fact 1"], "Fact 1 True or False?\nOutput:\n"),
        ([" Fact 2 "], "Fact 2 True or False?\nOutput:\n"),
    ],
)
def test_get_score_prompt_format(
    fact_scorer,
    mock_openai_agent,
    facts,
    expected_prompt_end,
    knowledge_source="Test context.",
):
    mock_openai_agent.generate.return_value = "True"
    fact_scorer.get_score(facts, knowledge_source)
    mock_openai_agent.generate.assert_called_once()
    args, kwargs = mock_openai_agent.generate.call_args
    assert args[0].endswith(
        expected_prompt_end
    ), "The prompt should be correctly formatted and end with the expected text."


def test_get_score_empty_input(fact_scorer):
    assert (
        fact_scorer.get_score([], "Some knowledge") == []
    ), "Should return an empty list for empty facts input."


@pytest.mark.parametrize(
    "response, expected",
    [
        ("True", True),
        ("False", False),
        ("True and False", False),
        ("False and True", True),
        ("Not available", False),
    ],
)
def test_get_score_response_interpretation(
    fact_scorer, mock_openai_agent, response, expected
):
    mock_openai_agent.generate.return_value = response
    result = fact_scorer.get_score(["Fact 1"], "Knowledge source")
    assert result == [
        {"Fact 1": expected}
    ], f"Expected decision to be {expected} for response '{response}'."


def test_error_handling_when_openai_agent_fails(fact_scorer, mock_openai_agent):
    mock_openai_agent.generate.side_effect = Exception("Network error")
    with pytest.raises(Exception):
        fact_scorer.get_score(["Fact 1"], "Knowledge source")


def test_complex_knowledge_source_and_atomic_facts(fact_scorer, mock_openai_agent):
    facts = ["A very complex fact that needs to be evaluated"]
    knowledge_source = (
        "A very long and complex knowledge source containing much information."
    )
    mock_openai_agent.generate.return_value = "True"
    result = fact_scorer.get_score(facts, knowledge_source)
    assert isinstance(
        result, list
    ), "Expected a list to be returned for complex inputs."
    assert all(
        isinstance(decision, dict) for decision in result
    ), "Each item in the returned list should be a dictionary."
