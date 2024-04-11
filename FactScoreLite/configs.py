import importlib.resources

# Path to the data file within the package
data_path = importlib.resources.files("FactScoreLite") / "data"

# Path to the data file within the package
demons_path = data_path / "demons.json"

# OpenAI API
max_tokens = 1024
temp = 0.7
model_name = "gpt-4-turbo-preview"

# Database path
facts_db_path = "facts.json"
decisions_db_path = "decisions.json"
