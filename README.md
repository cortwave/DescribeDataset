# DescribeDataset

A Python tool for generating human-readable descriptions of dataset structures. This tool helps you understand the organization and content of your datasets by providing a clear, textual representation that can be used with LLMs or for documentation purposes.

## Features

- 📁 Recursive folder structure exploration
- 📊 Support for multiple file formats:
  - Text files (`.txt`)
  - JSON files (`.json`)
  - YAML files (`.yaml`, `.yml`)
  - CSV files (`.csv`)
  - Images (`.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`)
- 🖼️ Image resolution information
- 📏 Smart list clipping for large directories
- 📝 Total item count for folders
- 🛡️ Error handling for corrupted files

## Installation

```bash
pip install describe-dataset
```

## Usage

### Command Line

```bash
python -m describe_dataset /path/to/your/dataset
```

### Python API

```python
from describe_dataset import describe_dataset
from pathlib import Path

# Get dataset description
description = describe_dataset(Path("/path/to/your/dataset"))
print(description)
```

## Example Output

```
Top level of the dataset contains:
    folder images
    folder annotations
    file config.yaml

Content of images folder:
Total items in folder: 1000 (showing first 3)
- image1.jpg (file)
  image of resolution 1920x1080
- image2.jpg (file)
  image of resolution 1280x720
- image3.jpg (file)
  image of resolution 800x600

Content of annotations folder:
- train.json (file)
{
    "version": "1.0",
    "images": [...],
    "annotations": [...]
}

Description of config.yaml file:
name: my-dataset
version: 1.0
classes:
  - car
  - pedestrian
  - bicycle
```

## Configuration

The tool has several configurable parameters:

- `DIFFERENT_FILES_LIMIT = 10`: Maximum number of different files to show in a folder
- `MIN_LIST_LENGTH = 3`: Minimum number of items to show when clipping large lists
- `LIST_SIZE_THRESHOLD_FOR_MIN_LENGTH = 20`: Threshold for when to start clipping lists

## Development

### Setting up development environment

```bash
# Clone the repository
git clone https://github.com/yourusername/DescribeDataset.git
cd DescribeDataset

# Install development dependencies
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run tests with coverage report
pytest

# Run specific test file
pytest tests/test_describer.py

# Run specific test function
pytest tests/test_describer.py::test_describe_file_image
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Before submitting a PR:
1. Ensure all tests pass
2. Add tests for new functionality
3. Format code with Black
4. Sort imports with isort

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
