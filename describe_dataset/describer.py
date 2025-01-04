from __future__ import annotations  # for forward references if Python < 3.10

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import yaml
from fire import Fire
from PIL import Image

DIFFERENT_FILES_LIMIT = 10
MIN_LIST_LENGTH = 3
LIST_SIZE_THRESHOLD_FOR_MIN_LENGTH = 20


@dataclass
class FileDescription:
    """
    Represents a description of a file.
    """
    content: Union[str, Dict]  # the text or JSON/YAML data
    object_type: str = "file"


@dataclass
class FolderDescription:
    """
    Represents a description of a folder,
    where 'content' is a dict of sub-items (files or folders).
    """
    content: Dict[str, Union[FileDescription, FolderDescription]]
    total_items: int = 0
    object_type: str = "folder"


class EmptyFileDescription(FileDescription):
    """
    Special case for unrecognized or empty files.
    """

    def __init__(self):
        super().__init__(content="")  # same field name as in FileDescription


def describe_dataset(dataset_path: Union[Path, str]) -> str:
    """
    Describe the top-level structure of the dataset,
    plus a short description of each top-level item.
    """
    dataset_path = Path(dataset_path)

    folder_desc = describe_folder(dataset_path)
    top_level = folder_desc.content

    result = "Top level of the dataset contains:\n"
    # First pass: list all items
    for name, desc in top_level.items():
        result += f"\t{desc.object_type} {name}\n"

    result += "\n"

    # Second pass: describe each item in detail
    for name, desc in top_level.items():
        if isinstance(desc, EmptyFileDescription):
            continue
        if desc.object_type == "folder":
            result += f"Content of {name} folder:\n"
        else:
            result += f"Description of {name} file:\n"
        if desc.object_type == "file":
            if isinstance(desc.content, dict):
                result += json.dumps(desc.content, indent=4)
            else:
                result += str(desc.content)
        else:
            result += _describe_folder_recursive(desc, indent_level=1)
        result += "\n"

    return result


def _describe_folder_recursive(folder_desc: FolderDescription, indent_level: int = 0) -> str:
    """
    Recursively describe a FolderDescription, showing sub-items.
    """
    indent = "  " * indent_level
    output = ""
    
    # Show total items count if some items were clipped
    if folder_desc.total_items > len(folder_desc.content):
        output += f"{indent}Total items in folder: {folder_desc.total_items} (showing first {len(folder_desc.content)})\n"
    
    for name, item_desc in folder_desc.content.items():
        output += f"{indent}- {name} ({item_desc.object_type})\n"
        if item_desc.object_type == "folder":
            output += _describe_folder_recursive(item_desc, indent_level + 1)
        else:
            # It's a file
            if isinstance(item_desc.content, dict):
                # if JSON/YAML inside
                text = json.dumps(item_desc.content, indent=4)
                # add extra indentation
                text = "\n".join(indent + "  " + line for line in text.splitlines())
                output += text + "\n"
            else:
                # Normal text file
                text = str(item_desc.content)
                text = "\n".join(indent + "  " + line for line in text.splitlines())
                output += text + "\n"

    return output


def describe_folder(folder_path: Path) -> FolderDescription:
    """
    Return a FolderDescription, describing all files/folders in 'folder_path'.
    """
    folder_structure: Dict[str, Union[FileDescription, FolderDescription]] = {}
    folder_content = sorted(folder_path.iterdir())
    total_items = len([x for x in folder_content if x.name != '.DS_Store'])
    folder_content = _clip_list(folder_content)

    for content in folder_content:
        if content.is_file():
            if content.name == '.DS_Store':
                continue
            folder_structure[content.name] = describe_file(content)
        elif content.is_dir():
            folder_structure[content.name] = describe_folder(content)

    return FolderDescription(content=folder_structure, total_items=total_items)


def describe_image(file_path: Path) -> FileDescription:
    """
    Return a FileDescription for an image file, including its resolution.
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            return FileDescription(content=f"image of resolution {width}x{height}")
    except Exception:
        return EmptyFileDescription()

def describe_file(file_path: Path) -> FileDescription:
    """
    Return a FileDescription, analyzing file type by extension.
    """
    file_extension = file_path.suffix[1:].lower()  # remove the '.' and lower-case
    if file_extension == 'txt':
        return describe_txt(file_path)
    elif file_extension == 'json':
        return describe_json(file_path)
    elif file_extension in ['yaml', 'yml']:
        return describe_yaml(file_path)
    elif file_extension == 'csv':
        return describe_csv(file_path)
    elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
        return describe_image(file_path)
    else:
        return EmptyFileDescription()


def describe_txt(file_path: Path) -> FileDescription:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return FileDescription(content=content)


def describe_json(file_path: Path) -> FileDescription:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    content = _clip_all_lists_in_dict(content)
    return FileDescription(content=content)


def describe_yaml(file_path: Path) -> FileDescription:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = yaml.safe_load(f)
    content = _clip_all_lists_in_dict(content)
    return FileDescription(content=content)


def describe_csv(file_path: Path) -> FileDescription:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    # Clip lines
    lines = _clip_list(lines)
    clipped_content = '\n'.join(lines)
    return FileDescription(content=clipped_content)


def _clip_all_lists_in_dict(content: Dict) -> Dict:
    """
    Recursively clip any lists in a JSON/YAML dict to avoid huge output.
    """
    if not isinstance(content, dict):
        return content  # e.g., might be a list, string, etc.

    for key, value in content.items():
        if isinstance(value, list):
            content[key] = _clip_list(value)
        elif isinstance(value, dict):
            content[key] = _clip_all_lists_in_dict(value)
    return content


def _clip_list(lst: list) -> list:
    """
    Clip very long lists or lists of files so we don't exceed the limit.
    """
    if len(lst) > LIST_SIZE_THRESHOLD_FOR_MIN_LENGTH:
        return lst[:MIN_LIST_LENGTH]
    elif len(lst) > DIFFERENT_FILES_LIMIT:
        return lst[:DIFFERENT_FILES_LIMIT]
    return lst


if __name__ == "__main__":
    Fire(describe_dataset)
