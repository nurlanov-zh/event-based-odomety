#!/usr/bin/env bash

set -e

find ./tools -iname "*.hpp" -or -iname "*.h" -or -iname "*.cpp" | xargs clang-format -i
find ./implementation -iname "*.hpp" -or -iname "*.h" -or -iname "*.cpp" | xargs clang-format -i
find ./common -iname "*.hpp" -or -iname "*.h" -or -iname "*.cpp" | xargs clang-format -i
