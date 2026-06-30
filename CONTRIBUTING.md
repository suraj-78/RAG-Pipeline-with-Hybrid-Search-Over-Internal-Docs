# Contributing to Enterprise Hybrid RAG Engine

First off, thank you for considering contributing to the Enterprise Hybrid RAG Engine! It's people like you that make open source such a powerful tool.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check our issue tracker to see if the issue has already been reported. When creating a bug report, please include:
- A clear summary of the issue.
- Step-by-step instructions to reproduce it.
- Your environment details (OS, Python version, dependencies version).
- Logs or error stack traces if applicable.

### Suggesting Enhancements

We are always looking for ways to improve the retrieval accuracy, pipeline latency, and citation traces! Suggest enhancements by opening a feature request issue.

### Pull Requests

Please follow these steps to submit your changes:
1. Fork the repository and create a new branch from `main`.
2. Ensure you have installed the development dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Implement your changes, ensuring code is formatted with `black` and linted with `ruff`:
   ```bash
   make format
   make lint
   ```
4. Write test cases for your features under `tests/` and run the tests:
   ```bash
   make test
   ```
5. Commit your changes using [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat: add hybrid dense-sparse evaluation metric`
   - `fix: resolve SQLite thread collision during concurrent index writes`
6. Push your branch to your fork and submit a Pull Request (PR) to the `main` branch.

## Conventional Commit Rules

Our project uses the following prefix classifications for git commits:
- `feat`: A new feature or capability.
- `fix`: A bug fix.
- `docs`: Documentation updates.
- `style`: Changes that do not affect the meaning of the code (formatting, white-space, semi-colons).
- `refactor`: A code change that neither fixes a bug nor adds a feature.
- `test`: Adding missing tests or correcting existing tests.
- `chore`: Internal repository management, dependency updates, build files.

Thanks,
Divy Yadav
