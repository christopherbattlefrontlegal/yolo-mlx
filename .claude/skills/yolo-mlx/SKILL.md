```markdown
# yolo-mlx Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the development patterns and conventions used in the `yolo-mlx` Python codebase. You'll learn how to structure files, write imports and exports, follow commit message styles, and understand the project's testing approach. This guide will help you contribute code that matches the established style and workflows.

## Coding Conventions

### File Naming
- Use **snake_case** for all filenames.
  - Example: `object_detector.py`, `image_utils.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import preprocess_image
    from .models import YOLOModel
    ```

### Export Style
- Use **named exports** (explicitly define what is exported).
  - Example:
    ```python
    __all__ = ['YOLOModel', 'preprocess_image']
    ```

### Commit Messages
- Commit messages are freeform, sometimes prefixed with `wip` for work-in-progress.
  - Example:
    ```
    wip: add initial YOLO model
    ```
- Keep commit messages concise (average length ~27 characters).

## Workflows

### Code Contribution
**Trigger:** When adding new features or fixing bugs  
**Command:** `/contribute`

1. Create a new branch for your changes.
2. Write code following the coding conventions above.
3. Add or update tests as needed (see Testing Patterns).
4. Commit your changes with a concise message (optionally prefix with `wip` if unfinished).
5. Push your branch and open a pull request.

### Running Tests
**Trigger:** When verifying code changes  
**Command:** `/run-tests`

1. Identify test files (pattern: `*.test.*`).
2. Run tests using the project's preferred test runner (framework unknown; use `pytest` or `unittest` as appropriate).
   - Example:
     ```bash
     pytest
     ```
     or
     ```bash
     python -m unittest discover
     ```

### File Organization
**Trigger:** When creating new modules or utilities  
**Command:** `/new-module`

1. Name the file using snake_case (e.g., `data_loader.py`).
2. Place related functions and classes in the same file.
3. Use relative imports for internal dependencies.
4. Define `__all__` to specify exports.

## Testing Patterns

- Test files follow the pattern `*.test.*` (e.g., `model.test.py`).
- The specific testing framework is not detected; common choices are `pytest` or `unittest`.
- Place tests in the same directory as the code or in a dedicated `tests/` folder.
- Example test file:
  ```python
  # model.test.py
  import unittest
  from .models import YOLOModel

  class TestYOLOModel(unittest.TestCase):
      def test_forward(self):
          model = YOLOModel()
          result = model.forward([1, 2, 3])
          self.assertIsNotNone(result)
  ```

## Commands
| Command        | Purpose                                 |
|----------------|-----------------------------------------|
| /contribute    | Start a new code contribution workflow  |
| /run-tests     | Run all tests in the codebase           |
| /new-module    | Create a new module following conventions|
```
