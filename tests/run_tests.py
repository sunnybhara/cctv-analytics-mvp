#!/usr/bin/env python3
"""
CCTV Analytics Test Runner
==========================
Run all tests with proper configuration and reporting.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --quick      # Run only fast tests
    python run_tests.py --coverage   # Run with coverage report
    python run_tests.py -v           # Verbose output
"""
import subprocess
import sys
import os


def main():
    """Run the test suite."""
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Base pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
    ]

    # Parse arguments
    args = sys.argv[1:]

    if "--quick" in args:
        # Skip slow tests
        cmd.extend(["-m", "not slow"])
        args.remove("--quick")

    if "--coverage" in args:
        cmd.extend([
            "--cov=.",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report",
            "--cov-exclude=tests/*",
        ])
        args.remove("--coverage")

    # Add any remaining args
    cmd.extend(args)

    print("=" * 60)
    print("CCTV Analytics Test Suite")
    print("=" * 60)
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    # Run tests
    result = subprocess.run(cmd)

    # Summary
    print("-" * 60)
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
