#!/bin/bash
# Host tests for the app's pure C++ (names.h). Needs a C++17 compiler (clang++ on macOS).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${TMPDIR:-/tmp}/tellwell-names-test"
"${CXX:-clang++}" -std=c++17 -O1 -Wall -Wextra -Werror -o "$OUT" tests/names_test.cpp
"$OUT"
