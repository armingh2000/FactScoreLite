import pytest
from unittest.mock import mock_open, patch
from FactScoreLite.fact_scorer import FactScorer
import json
from FactScoreLite import configs


@pytest.fixture
def mock_openai_agent():
    with patch("FactScoreLite.fact_scorer.OpenAIAgent") as mock:
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
        {"fact": "Fact 1", "is_supported": expected, "output": response}
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


# Sample data to be returned by the mock
mock_demons_data = [
    {
        "knowledge_source": "knw 1",
        "fact": "fact 1",
        "is_supported": True,
    },
    {
        "knowledge_source": "knw 2",
        "fact": "fact 2",
        "is_supported": False,
    },
]


# Test for the load_demons method
def test_load_demons(fact_scorer):
    # Convert your sample data to a JSON string for mocking
    mock_json_str = json.dumps(mock_demons_data)
    # Use patch to mock open function within the context of your test
    with patch("builtins.open", mock_open(read_data=mock_json_str)):
        # Also mock configs.demons_path to avoid dependency on external config files
        with patch.object(
            configs, "atomic_facts_demons_path", "fake/path/to/fact_scorer_demons.json"
        ):
            demons = fact_scorer.load_demons()
            # Assert that the returned data matches your mock data
            assert (
                demons == mock_demons_data
            ), "The method should load and return the demons correctly."


def test_get_instructions_true_false_demons(fact_scorer):
    # Test case for a single demon in self.demons
    fact_scorer.demons = mock_demons_data
    expected_instructions = (
        "Evaluate the truthfulness of the statement based solely on the provided context and provide the reason for your decision.\n\n"
        "Instruction:\nOnly consider the statement true if it can be directly verified by the information in the context. If the information in the statement cannot be found in the context or differs from it, label it as false.\n\n"
        "Context:\nknw 1\n"
        "Statement:\nfact 1 True or False?\n"
        "Output:\nTrue\n\n"
        "Context:\nknw 2\n"
        "Statement:\nfact 2 True or False?\n"
        "Output:\nFalse\n\n"
    )
    assert fact_scorer.get_instructions() == expected_instructions
