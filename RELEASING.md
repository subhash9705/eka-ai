# Releasing EKA-AI to PyPI

This document outlines the step-by-step process for building, verifying, and publishing the `eka-ai` package to PyPI.

---

## Prerequisites

Before starting, ensure you have the required packaging tools installed:

```bash
pip install build twine
```

---

## Step 1: Bump Version and Format Code

1. Update the version number in `pyproject.toml` and `eka_ai/__init__.py`.
2. Format your code and ensure lint checks pass (e.g., using `ruff` or `black`).
3. Run the unit test suite locally to verify code correctness:
   ```bash
   python -m pytest
   ```

---

## Step 2: Build the Package

Clean the `dist/` directory (if it exists) and run the python build tool:

```bash
# Windows (PowerShell / cmd)
Remove-Item -Recurse -Force dist/ -ErrorAction SilentlyContinue
python -m build

# Linux / macOS
rm -rf dist/
python -m build
```

This will generate two artifacts in the `dist/` directory:
- A source archive: `dist/eka_ai-<version>.tar.gz`
- A built wheel: `dist/eka_ai-<version>-py3-none-any.whl`

---

## Step 3: Verify the Build

Verify that the built package conforms to PyPI's metadata and description requirements:

```bash
python -m twine check dist/*
```

---

## Step 4: Publish to PyPI

You can publish the package using one of two methods:

### Method A: Automated via GitHub Actions (Recommended)

The repository uses **Trusted Publishing** via GitHub Actions (`.github/workflows/publish.yml`).

1. **Tag the commit** matching the release version (e.g., `v1.0.0`):
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
2. **Publish a Release on GitHub**:
   - Go to your repository's Releases page on GitHub.
   - Click **Draft a new release**.
   - Choose the tag `v1.0.0` and title the release.
   - Click **Publish release**.
   - This automatically triggers the runner to test, build, and publish directly to PyPI.

### Method B: Manual Upload via Twine

If you want to manually publish directly from your machine using an API token:

```bash
# Set Twine environment variables
$env:TWINE_USERNAME="__token__"
$env:TWINE_PASSWORD="your-pypi-api-token"  # Must start with pypi-

# Upload the built distribution packages
python -m twine upload dist/*
```

---

## Step 5: Verify Live Release

After publishing, wait a minute and verify that the package is available on PyPI:

```bash
pip index versions eka-ai
```

You can now install the package using:
```bash
pip install eka-ai --upgrade
```
