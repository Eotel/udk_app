# Lint Rules Documentation

This document provides guidance on the lint rules used in the udk-app project, including which rules are ignored and why, and which rules should be followed.

## Overview

The project uses [Ruff](https://github.com/charliermarsh/ruff) for linting and formatting Python code. Ruff is a fast Python linter written in Rust that aims to replace multiple Python linting tools with a single, fast implementation.

## Configuration

The lint rules are configured in the `pyproject.toml` file under the `[tool.ruff.lint]` section. The project is configured to select all available rules (`select = ["ALL"]`) and then explicitly ignore specific rules that are not applicable or conflict with other tools.

## Ignored Rules

The following rules are ignored in the project:

| Rule ID | Description | Reason for Ignoring |
|---------|-------------|---------------------|
| F401 | Unused imports | Sometimes imports are included for type checking or future use |
| S104 | Possible binding to all interfaces | Needed for development servers |
| F403 | Star imports | Used in migrations and specific modules |
| E501 | Line too long | Handled by the formatter |
| RUF001 | Ambiguous unicode character in string | Project uses Japanese text |
| RUF002 | Ambiguous unicode character in docstring | Project uses Japanese text |
| RUF003 | Ambiguous unicode character in comment | Project uses Japanese text |
| C901 | Function is too complex | Some business logic is inherently complex |
| PLR0912 | Too many branches | Some business logic requires many branches |
| PLR0915 | Too many statements | Some business logic requires many statements |
| TRY301 | Abstract raise to an inner function | Would require substantial refactoring |
| D203 | One blank line required before class docstring | Incompatible with D211 |
| D213 | Multi-line docstring summary should start at the second line | Incompatible with D212 |
| COM812 | Missing trailing comma in collection of items | Conflicts with formatter |

## Rules to Follow

All other rules provided by Ruff should be followed. Some particularly important rules include:

1. **Docstring Rules (D)**: All functions, classes, and modules should have proper docstrings.
2. **Import Rules (I)**: Imports should be organized properly and avoid circular imports.
3. **Naming Rules (N)**: Follow consistent naming conventions.
4. **Error Rules (E)**: Follow PEP 8 style guide.
5. **Warning Rules (W)**: Address potential issues and warnings.
6. **Style Rules (S)**: Follow security best practices.

## Best Practices

1. **Run Linting Regularly**: Run `uv run -- ruff check --fix .` before committing changes.
2. **Run Formatting**: Run `uv run -- ruff format .` to ensure consistent code formatting.
3. **Address Warnings**: Fix any warnings or errors reported by the linter.
4. **Don't Ignore Rules Unnecessarily**: Only ignore rules when there's a good reason to do so.
5. **Keep Configuration Updated**: Update the lint configuration as the project evolves.

## Adding New Rules to Ignore

If you need to add a new rule to the ignore list:

1. Understand why the rule exists and what it's trying to prevent.
2. Consider if there's a way to comply with the rule instead of ignoring it.
3. If ignoring is necessary, add it to the `ignore` list in `pyproject.toml` with a comment explaining why.
4. Update this documentation to include the new ignored rule.

## Formatter Conflicts

Some linting rules may conflict with the formatter. In such cases, it's generally better to ignore the linting rule and let the formatter handle the formatting. The COM812 rule is an example of this, as it conflicts with how the formatter handles trailing commas.
