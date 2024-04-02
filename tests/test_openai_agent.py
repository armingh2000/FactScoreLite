# test_openai_agent.py
import pytest
from unittest.mock import patch, MagicMock
from FactScoreLite import OpenAIAgent
from FactScoreLite.openai_agent import retry_with_exponential_backoff
from openai import RateLimitError

# Decorator


def mock_function_to_decorate(*args, **kwargs):
    """A simple mock function that simulates a successful operation."""
    return "Success"


@pytest.fixture
def successful_function():
    """Applies the retry_with_exponential_backoff decorator to a mock function that simulates a successful call."""
    decorated_function = retry_with_exponential_backoff(mock_function_to_decorate)
    return decorated_function


def test_retry_decorator_with_successful_call(successful_function):
    """Test the retry_with_exponential_backoff decorator with a function that succeeds on the first attempt."""
    # Call the decorated function
    result = successful_function()

    # Assert that the function returns the expected result without retrying
    assert (
        result == "Success"
    ), "The decorated function should return 'Success' on the first call without any retries."


# A mock function to decorate, designed to fail once with a RateLimitError, then succeed
def mock_function_with_initial_failure(
    attempts=[0],
):  # Using a mutable default argument as a simple static counter
    if attempts[0] == 0:
        attempts[0] += 1
        raise RateLimitError(
            "Simulated retryable error", response=MagicMock(), body=MagicMock()
        )
    else:
        return "Recovered Success"


@pytest.fixture
def function_with_retry_logic():
    """Applies the retry_with_exponential_backoff decorator to a mock function that initially fails."""
    decorated_function = retry_with_exponential_backoff(
        mock_function_with_initial_failure, errors=(RateLimitError,)
    )
    return decorated_function


def test_retry_decorator_activates_on_failure(function_with_retry_logic):
    """Test that the retry_with_exponential_backoff decorator retries the function upon an initial failure."""
    # Call the decorated function
    result = function_with_retry_logic()

    # Assert that the function returns the expected recovery result after retrying
    assert (
        result == "Recovered Success"
    ), "The decorated function should retry and return 'Recovered Success' after the initial failure."


# A mock function to decorate, designed to always fail with a RateLimitError
def mock_function_always_fails():
    raise RateLimitError(
        "Simulated retryable error", response=MagicMock(), body=MagicMock()
    )


@pytest.fixture
def function_with_retry_limit():
    """Applies the retry_with_exponential_backoff decorator to a mock function that always fails."""
    # Here, you can adjust initial_delay to a smaller value to make the test run faster
    decorated_function = retry_with_exponential_backoff(
        mock_function_always_fails,
        max_retries=3,
        initial_delay=0.1,
        errors=(RateLimitError,),
    )
    return decorated_function


def test_retry_decorator_exceeds_max_retries(function_with_retry_limit):
    """Test that the retry_with_exponential_backoff decorator raises an exception after exceeding the maximum retry attempts."""
    with pytest.raises(Exception) as exc_info:
        function_with_retry_limit()
    assert "Maximum number of retries" in str(
        exc_info.value
    ), "The decorated function should raise an exception indicating the maximum retries have been exceeded."


# A mock function to decorate, designed to fail twice before succeeding
def mock_function_fails_then_succeeds(attempts=[0]):
    if attempts[0] < 2:
        attempts[0] += 1
        raise RateLimitError(
            "Simulated retryable error", response=MagicMock(), body=MagicMock()
        )
    else:
        return "Success"


@pytest.fixture
def function_with_exponential_backoff():
    """Applies the retry_with_exponential_backoff decorator with jitter to a function that initially fails."""
    decorated_function = retry_with_exponential_backoff(
        mock_function_fails_then_succeeds,
        initial_delay=1,
        exponential_base=2,
        jitter=True,
        max_retries=3,
        errors=(RateLimitError,),
    )
    return decorated_function


def test_exponential_backoff_and_jitter(function_with_exponential_backoff):
    """Test that retry_with_exponential_backoff applies an increasing delay with jitter between retries."""
    with patch("time.sleep") as mock_sleep:
        function_with_exponential_backoff()

        assert (
            mock_sleep.call_count == 2
        ), "The function should have been retried twice, with time.sleep called twice."

        # Check the calls to mock_sleep to ensure they match the expected pattern of delays
        delays = [call_args[0][0] for call_args in mock_sleep.call_args_list]

        # Verify that each delay is greater than the last, accounting for jitter
        assert delays[0] >= 1, "The first retry delay should be at least 1 second."
        assert (
            delays[1] >= 2 * delays[0]
        ), "The second retry delay should be at least double the first delay."


class NonRetryableError(Exception):
    """A custom exception class to represent errors that should not trigger retries."""

    pass


# A mock function to decorate, designed to always fail with NonRetryableError
def mock_function_raises_non_retryable_error():
    raise NonRetryableError("This error is not supposed to trigger retries.")


@pytest.fixture
def function_with_non_retryable_error_handling():
    """Applies the retry_with_exponential_backoff decorator to a function, specifying that it should not retry for NonRetryableError."""
    decorated_function = retry_with_exponential_backoff(
        mock_function_raises_non_retryable_error,
        errors=(RateLimitError,),  # Specify only retryable errors here
    )
    return decorated_function


def test_non_retryable_errors_are_not_retried(
    function_with_non_retryable_error_handling,
):
    """Test that non-retryable errors immediately raise an exception without triggering retry logic."""
    with pytest.raises(NonRetryableError) as exc_info:
        function_with_non_retryable_error_handling()

    assert "This error is not supposed to trigger retries." in str(
        exc_info.value
    ), "The function should raise NonRetryableError without retry attempts."


# OPENAI CLASS


@pytest.fixture
def agent(mocker):
    """Fixture to create an OpenAIAgent instance for use in tests, with the OpenAI client's chat.completions.create method mocked."""
    # First, create the mock for the completions.create method
    create_method_mock = MagicMock()

    # Since 'chat' is a method that returns an object which has the 'completions' method,
    # you mock it by having the 'chat' method return a mock object where 'completions.create' is defined.
    mock_completions = MagicMock(create=create_method_mock)

    # Mock the 'chat' method to return an object that has a 'completions' attribute
    mock_chat_method = MagicMock(completions=mock_completions)

    # Patch the OpenAI class to return a mock instance with the 'chat' method mocked as above
    mock_openai_instance = MagicMock(chat=mock_chat_method)
    mock_openai = mocker.patch(
        "FactScoreLite.openai_agent.OpenAI", return_value=mock_openai_instance
    )

    # Instantiate and return the OpenAIAgent, which now uses the mocked OpenAI class,
    # along with the mocked create method for custom behavior in tests
    return OpenAIAgent(), create_method_mock


def test_successful_text_generation_without_retry(agent):
    """Test that the OpenAIAgent's generate method successfully returns the expected output on the first try, without needing to retry."""
    openai_agent, create_method_mock = agent

    # Configure the mock to return a successful response
    mock_message = MagicMock(content="Test successful response")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    create_method_mock.return_value = mock_response

    # Execute the generate method with a test prompt
    response = openai_agent.generate("Test prompt")

    # Assert that the response is as expected
    assert (
        response == "Test successful response"
    ), "The generate method should return the expected response without needing to retry."


def test_text_generation_with_retry(agent, mocker):
    """Test that the OpenAIAgent's generate method successfully retries and returns the expected output after initially hitting a rate limit error."""
    openai_agent, create_method_mock = agent

    # Configure the mock to raise a RateLimitError on the first call, then return a successful response
    rate_limit_error_response = RateLimitError(
        "Rate limit exceeded",
        response=mocker.MagicMock(),
        body={"error": "Rate limit exceeded"},
    )

    # Successful response setup
    mock_message = mocker.MagicMock(content="Test successful response after retry")
    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock(message=mock_message)]

    # Configure the mock behavior: fail first, then succeed
    create_method_mock.side_effect = [rate_limit_error_response, mock_response]

    # Execute the generate method with a test prompt
    response = openai_agent.generate("Test prompt")

    # Assert that the response is as expected after retrying
    assert (
        response == "Test successful response after retry"
    ), "The generate method should retry and return the expected response after encountering a rate limit error."

    # Additionally, you can assert that there were exactly two calls to the create method
    assert (
        create_method_mock.call_count == 2
    ), "Expected the create method to be called twice (one initial call and one retry)."


def test_text_generation_failure_on_max_retries(agent, mocker):
    """Test that the OpenAIAgent's generate method raises an exception after exceeding the maximum number of retries due to persistent rate limit errors, without waiting for actual delays."""
    openai_agent, create_method_mock = agent

    # Configure the mock to always raise a RateLimitError
    persistent_rate_limit_error = RateLimitError(
        "Rate limit exceeded",
        response=mocker.MagicMock(),
        body={"error": "Rate limit exceeded"},
    )
    create_method_mock.side_effect = persistent_rate_limit_error

    # Mock time.sleep to skip actual waiting
    mocker.patch("time.sleep", return_value=None)

    # Attempt to execute the generate method with a test prompt
    with pytest.raises(Exception) as exc_info:
        openai_agent.generate("Test prompt")

    # Assert that an exception indicating max retries exceeded is raised
    assert "Maximum number of retries" in str(
        exc_info.value
    ), "Expected exception indicating max retries exceeded."

    # Assert on the number of calls to create_method_mock to ensure retries were attempted up to the max limit
    max_retries = 10  # Assuming this is your configured max_retries in the decorator
    assert (
        create_method_mock.call_count == max_retries + 1
    ), f"Expected {max_retries + 1} calls (1 initial + {max_retries} retries)."
