import json
from pathlib import Path

import pytest
import yaml

from describe_dataset.describer import (
    describe_dataset,
    describe_file,
    describe_folder,
    FileDescription,
    FolderDescription,
    EmptyFileDescription,
)

# Test data path
TEST_DATA_PATH = Path(__file__).parent / "data" / "sample_dataset"


def test_describe_file_txt():
    """Test describing a text file."""
    file_path = TEST_DATA_PATH / "config.yaml"
    desc = describe_file(file_path)
    assert isinstance(desc, FileDescription)
    assert isinstance(desc.content, dict)
    assert desc.content["name"] == "test-dataset"
    assert desc.content["version"] == "1.0"
    assert desc.content["classes"] == ["car", "pedestrian", "bicycle"]


def test_describe_file_json():
    """Test describing a JSON file."""
    file_path = TEST_DATA_PATH / "annotations" / "train.json"
    desc = describe_file(file_path)
    assert isinstance(desc, FileDescription)
    assert isinstance(desc.content, dict)
    assert desc.content["version"] == "1.0"
    assert len(desc.content["images"]) == 2
    assert len(desc.content["annotations"]) == 2


def test_describe_file_image():
    """Test describing an image file."""
    file_path = TEST_DATA_PATH / "images" / "1.jpg"
    desc = describe_file(file_path)
    assert isinstance(desc, FileDescription)
    assert desc.content == "image of resolution 1920x1080"


def test_describe_file_unknown():
    """Test describing an unknown file type."""
    file_path = TEST_DATA_PATH / "nonexistent.xyz"
    desc = describe_file(file_path)
    assert isinstance(desc, EmptyFileDescription)
    assert desc.content == ""


def test_describe_folder():
    """Test describing a folder."""
    folder_path = TEST_DATA_PATH / "images"
    desc = describe_folder(folder_path)
    assert isinstance(desc, FolderDescription)
    assert desc.object_type == "folder"
    assert desc.total_items == 4  # We have 4 test images
    assert len(desc.content) == 3  # But only show first 3 due to MIN_LIST_LENGTH


def test_describe_dataset():
    """Test describing the entire dataset."""
    desc_text = describe_dataset(TEST_DATA_PATH)
    assert isinstance(desc_text, str)
    
    # Check if all main components are mentioned
    assert "Top level of the dataset contains:" in desc_text
    assert "folder images" in desc_text
    assert "folder annotations" in desc_text
    assert "file config.yaml" in desc_text
    
    # Check if image resolutions are shown
    assert "image of resolution 1920x1080" in desc_text
    
    # Check if JSON content is shown
    assert '"version": "1.0"' in desc_text
    
    # Check if YAML content is shown
    assert "name: test-dataset" in desc_text
    assert "- car" in desc_text


def test_folder_clipping():
    """Test that folders with many items are properly clipped."""
    # Create a folder with many files
    many_files_path = TEST_DATA_PATH / "many_files"
    many_files_path.mkdir(exist_ok=True)
    for i in range(15):  # Create more than DIFFERENT_FILES_LIMIT files
        (many_files_path / f"file{i}.txt").touch()

    desc = describe_folder(many_files_path)
    assert desc.total_items == 15
    assert len(desc.content) < 15  # Should be clipped
    
    # Cleanup
    for f in many_files_path.glob("*"):
        f.unlink()
    many_files_path.rmdir()


def test_corrupted_files():
    """Test handling of corrupted files."""
    # Create a corrupted JSON file
    corrupted_path = TEST_DATA_PATH / "corrupted.json"
    corrupted_path.write_text("{invalid json")
    
    desc = describe_file(corrupted_path)
    assert isinstance(desc, EmptyFileDescription)
    
    # Cleanup
    corrupted_path.unlink()


def test_empty_folder():
    """Test describing an empty folder."""
    empty_folder_path = TEST_DATA_PATH / "empty"
    empty_folder_path.mkdir(exist_ok=True)
    
    desc = describe_folder(empty_folder_path)
    assert isinstance(desc, FolderDescription)
    assert desc.total_items == 0
    assert len(desc.content) == 0
    
    # Cleanup
    empty_folder_path.rmdir()