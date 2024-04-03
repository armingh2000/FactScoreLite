from openai import OpenAI
from openai import (
    RateLimitError,
)
import time
import logging
import random
from . import configs


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                logging.info(f"Attempting to call {func.__name__}")
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1

                if num_retries > max_retries:
                    logging.error(
                        f"Maximum number of retries ({max_retries}) exceeded for {func.__name__}."
                    )
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                logging.warning(
                    f"Retry #{num_retries} for {func.__name__} after encountering {e}. Waiting {delay} seconds before retrying..."
                )
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                logging.exception(f"Unexpected exception during {func.__name__}: {e}")
                raise e

    return wrapper


class OpenAIAgent:

    def __init__(self):
        self.client = OpenAI()
        self.max_tokens = configs.max_tokens
        self.temp = configs.temp
        self.model_name = configs.model_name

    @retry_with_exponential_backoff
    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temp,
        )
        return response.choices[0].message.content


# Ensure proper logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
