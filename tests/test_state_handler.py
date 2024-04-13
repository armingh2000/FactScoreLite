import os
from FactScoreLite.state_handler import StateHandler


def test_save_load_data():
    """Test saving and loading data functionality."""
    handler = StateHandler("test_data.json")
    test_data = {"key": "value", "number": 42}

    # Test saving data
    handler.save(test_data)

    # Verify file was created
    assert os.path.exists("test_data.json")

    # Test loading data
    loaded_data = handler.load()

    # Verify loaded data matches saved data
    assert loaded_data == test_data

    # Cleanup test file
    os.remove("test_data.json")


def test_load_nonexistent_file():
    """Test loading from a nonexistent file returns an empty list."""
    handler = StateHandler("nonexistent.json")

    # Test loading data from a file that doesn't exist
    loaded_data = handler.load()

    # Verify that the loaded data is an empty list
    assert loaded_data == [], "Expected an empty list from a nonexistent file"


def test_save_data_integrity():
    """Test that saved data maintains its integrity upon loading."""
    handler = StateHandler("test_integrity.json")
    complex_data = {
        "list": [1, 2, 3],
        "nested_dict": {"subkey": "subvalue"},
        "boolean": True,
        "none": None,
    }

    # Save complex data structure
    handler.save(complex_data)

    # Load the data back
    loaded_data = handler.load()

    # Verify the integrity of the loaded data
    assert loaded_data == complex_data, "Loaded data does not match the saved data"

    # Cleanup
    os.remove("test_integrity.json")
