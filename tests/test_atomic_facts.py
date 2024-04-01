from unittest.mock import mock_open, patch
import json
import pytest
from unittest.mock import MagicMock
from FactScoreLite.atomic_facts import AtomicFactGenerator
from FactScoreLite import configs


@pytest.fixture
def generator(monkeypatch):
    # Create an instance of AtomicFactGenerator for testing
    generator = AtomicFactGenerator()

    # Create a MagicMock object for the generate method
    mock_generate = MagicMock(return_value="Generated output.")

    # Patch the generate method with the MagicMock object
    monkeypatch.setattr(generator.openai_agent, "generate", mock_generate)

    return generator


# Sample data to be returned by the mock
mock_demons_data = {
    "demons": [
        {"Sentence": "Example sentence 1.", "Independent Facts": ["Fact 1", "Fact 2"]},
        {"Sentence": "Example sentence 2.", "Independent Facts": ["Fact 3", "Fact 4"]},
    ]
}


# Test for the load_demons method
def test_load_demons():
    # Convert your sample data to a JSON string for mocking
    mock_json_str = json.dumps(mock_demons_data)
    # Use patch to mock open function within the context of your test
    with patch("builtins.open", mock_open(read_data=mock_json_str)):
        # Also mock configs.demons_path to avoid dependency on external config files
        with patch.object(configs, "demons_path", "fake/path/to/demons.json"):
            generator = AtomicFactGenerator()
            demons = generator.load_demons()
            # Assert that the returned data matches your mock data
            assert (
                demons == mock_demons_data
            ), "The method should load and return the demons correctly."


def test_get_instructions_single_demon(generator):
    # Test case for a single demon in self.demons
    generator.demons = [
        {"Sentence": "Example sentence 1", "Independent Facts": ["Fact 1", "Fact 2"]},
    ]
    expected_instructions = (
        "Please breakdown the following sentence into independent facts:\n\n"
        "Sentence:\nExample sentence 1\n"
        "Independent Facts:\n- Fact 1\n- Fact 2\n\n\n"
    )
    assert generator.get_instructions() == expected_instructions


def test_get_instructions_multiple_demons(generator):
    generator.demons = [
        {"Sentence": "Example sentence 1", "Independent Facts": ["Fact 1", "Fact 2"]},
        {"Sentence": "Example sentence 2", "Independent Facts": ["Fact 3", "Fact 4"]},
    ]
    # Test case for multiple demons in self.demons
    expected_instructions = (
        "Please breakdown the following sentence into independent facts:\n\n"
        "Sentence:\nExample sentence 1\n"
        "Independent Facts:\n- Fact 1\n- Fact 2\n\n\n"
        "Sentence:\nExample sentence 2\n"
        "Independent Facts:\n- Fact 3\n- Fact 4\n\n\n"
    )
    assert generator.get_instructions() == expected_instructions


def test_get_instructions_empty_demons(generator):
    # Test case for an empty self.demons list
    generator.demons = []
    expected_instructions = (
        "Please breakdown the following sentence into independent facts:\n\n"
    )
    assert generator.get_instructions() == expected_instructions


def test_get_instructions_formatting(generator):
    # Test case to ensure correct formatting of the instructions
    generator.demons = [
        {
            "Sentence": "Sample sentence",
            "Independent Facts": ["Fact 1", "Fact 2", "Fact 3"],
        }
    ]
    expected_instructions = (
        "Please breakdown the following sentence into independent facts:\n\n"
        "Sentence:\nSample sentence\n"
        "Independent Facts:\n- Fact 1\n- Fact 2\n- Fact 3\n\n\n"
    )
    assert generator.get_instructions() == expected_instructions


def test_gpt_output_to_sentences_normal(generator):
    # Test case with normal input text
    text = "- Sentence 1.\n- Sentence 2.\n- Sentence 3."
    expected_sentences = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
    assert generator.gpt_output_to_sentences(text) == expected_sentences


def test_gpt_output_to_sentences_no_newline(generator):
    # Test case with input text without newlines
    text = "- Sentence 1. - Sentence 2. - Sentence 3."
    expected_sentences = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
    assert generator.gpt_output_to_sentences(text) == expected_sentences


def test_gpt_output_to_sentences_no_sentences(generator):
    # Test case with empty input text
    text = ""
    expected_sentences = []
    assert generator.gpt_output_to_sentences(text) == expected_sentences


def test_gpt_output_to_sentences_single_sentence(generator):
    # Test case with a single sentence
    text = "- Sentence."
    expected_sentences = ["Sentence."]
    assert generator.gpt_output_to_sentences(text) == expected_sentences


def test_gpt_output_to_sentences_single_sentence_no_dot(generator):
    # Test case with a single sentence without a period
    text = "- Sentence"
    expected_sentences = ["Sentence."]
    assert generator.gpt_output_to_sentences(text) == expected_sentences


def test_detect_initials_normal(generator):
    # Test case with normal input text containing initials
    text = "The U.S. is located in North America. A.B. is a common abbreviation."
    expected_initials = ["U.S.", "A.B."]
    assert generator.detect_initials(text) == expected_initials


def test_detect_initials_no_matches(generator):
    # Test case with input text containing no initials
    text = "This is a sentence without any initials."
    expected_initials = []
    assert generator.detect_initials(text) == expected_initials


def test_detect_initials_special_characters(generator):
    # Test case with input text containing special characters
    text = "The $U.S.$ is located in North America. @J.K.@ Rowling wrote Harry Potter."
    expected_initials = ["U.S.", "J.K."]
    assert generator.detect_initials(text) == expected_initials


def test_fix_sentence_splitter_handles_initials_correctly(generator):

    sentences = [
        "Dr. J.P. Richardson, Ph.D., met with M.D.",
        "Anderson, M.D., at the J.F.K.",
        "Center for a discussion on A.I.",
        "advancements.",
    ]
    initials = ["J.P.", "M.D.", "M.D.", "J.F.", "A.I."]
    expected = [
        "Dr. J.P. Richardson, Ph.D., met with M.D.",
        "Anderson, M.D., at the J.F.K.",
        "Center for a discussion on A.I. advancements.",
    ]
    assert generator.fix_sentence_splitter(sentences, initials) == expected


def test_fix_sentence_splitter_no_initials(generator):
    sentences = ["This is a sentence.", "This is another sentence."]
    initials = []
    expected = sentences  # No change expected
    assert generator.fix_sentence_splitter(sentences, initials) == expected


def test_fix_sentence_splitter_with_multiple_sentences(generator):
    sentences = ["Here is a sentence.", "Here is another one.", "And yet another one."]
    initials = []
    expected = sentences  # No change expected
    assert generator.fix_sentence_splitter(sentences, initials) == expected


def test_fix_sentence_splitter_merges_sentences_starting_with_lowercase(
    generator,
):
    sentences = [
        "This is a sentence. this should be merged with the previous sentence."
    ]
    initials = []
    expected = ["This is a sentence. this should be merged with the previous sentence."]
    assert generator.fix_sentence_splitter(sentences, initials) == expected


def test_fix_sentence_splitter_single_word_sentences(generator):
    sentences = ["Wow.", "That was amazing."]
    initials = []
    expected = ["Wow. That was amazing."]
    assert generator.fix_sentence_splitter(sentences, initials) == expected
