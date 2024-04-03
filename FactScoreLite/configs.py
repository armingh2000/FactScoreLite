import importlib.resources

# Path to the data file within the package
demons_path = importlib.resources.files("FactScoreLite") / "data" / "demons.json"


# OpenAI API
max_tokens = 1024
temp = 0.7
model_name = "gpt-4-turbo-preview"
