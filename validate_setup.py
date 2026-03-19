# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for PhySkin setup.
Run this before first use to check configuration and dependencies.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def check_imports() -> Tuple[bool, List[str]]:
    """Check if entry points can be imported."""
    issues = []
    success = True

    print("Checking imports...")

    # Try importing entry points
    entry_points = ["train", "infer"]

    for module_name in entry_points:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            issues.append(f"Cannot import {module_name}: {e}")
            success = False

    return success, issues


def check_configs() -> Tuple[bool, List[str]]:
    """Check if config files have placeholders."""
    issues = []
    success = True

    print("\nChecking configurations...")

    config_dir = Path("config")
    if not config_dir.exists():
        issues.append("Config directory not found")
        return False, issues

    placeholder_pattern = r"<PATH_TO_YOUR_DATA_ROOT>"
    has_placeholders = False

    for yaml_file in config_dir.rglob("*.yaml"):
        with open(yaml_file, "r") as f:
            content = f.read()

        if re.search(placeholder_pattern, content):
            print(f"  ⚠ {yaml_file.name}: Contains placeholders (needs customization)")
            has_placeholders = True

    if has_placeholders:
        issues.append(
            "Config files contain placeholders - you must update paths before running"
        )
        success = False
    else:
        print("  ✓ All configs appear customized")

    return success, issues


def check_data_directories() -> Tuple[bool, List[str]]:  # noqa: C901
    """Check if expected data directories exist (if configs are customized)."""
    issues = []

    print("\nChecking data directories...")

    # Try to load config and check paths
    try:
        import yaml

        with open("config/physkin_hyperbone.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Check if still has placeholders
        config_str = str(config)
        if "<PATH_TO_YOUR_DATA_ROOT>" in config_str:
            print("  ⚠ Skipping (configs not yet customized)")
            return True, []

        # Extract paths and check existence
        paths_to_check = []

        # Look for common path keys
        def extract_paths(d, prefix=""):
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, str) and (
                        "path" in k.lower() or "root" in k.lower()
                    ):
                        paths_to_check.append((f"{prefix}.{k}", v))
                    elif isinstance(v, dict):
                        extract_paths(v, f"{prefix}.{k}" if prefix else k)

        extract_paths(config)

        missing = []
        for key, path in paths_to_check:
            if path and not Path(path).exists():
                missing.append(f"{key}: {path}")

        if missing:
            print("  ✗ Missing paths:")
            for m in missing:
                print(f"      {m}")
            issues.extend(missing)
        else:
            print("  ✓ All configured paths exist")

    except Exception as e:
        print(f"  ⚠ Could not validate paths: {e}")

    return len(issues) == 0, issues


def print_checklist(all_issues: List[str]):
    """Print actionable checklist."""
    print("\n" + "=" * 60)
    if not all_issues:
        print("✓ SETUP VALIDATION PASSED")
        print("=" * 60)
        print("\nYour PhySkin installation appears ready to use!")
        print("\nNext steps:")
        print("  1. Ensure you have training data in configured paths")
        print("  2. Run training: python train.py")
        print("  3. Run inference: python infer.py")
    else:
        print("✗ SETUP VALIDATION FAILED")
        print("=" * 60)
        print("\nIssues found:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")

        print("\nAction items:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print(
            "  2. Update config files in config/ - replace <PATH_TO_YOUR_DATA_ROOT> with your actual paths"
        )
        print("  3. Ensure data directories exist at configured paths")
        print("  4. Run this script again: python validate_setup.py")


def main():
    """Run all validation checks."""
    print("PhySkin Setup Validation")
    print("=" * 60)

    all_issues = []

    # Run all checks
    _import_ok, import_issues = check_imports()
    all_issues.extend(import_issues)

    _config_ok, config_issues = check_configs()
    all_issues.extend(config_issues)

    _data_ok, data_issues = check_data_directories()
    all_issues.extend(data_issues)

    # Print summary
    print_checklist(all_issues)

    # Exit code
    sys.exit(0 if not all_issues else 1)


if __name__ == "__main__":
    main()
