#!/bin/bash
# Usage: scripts/record.sh NAME command [args...]   (writes bench/NAME.txt)
name="$1"; shift
out="bench/${name}.txt"
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "parent: $(git rev-parse --short HEAD), uncommitted files outside include/: $(git status --porcelain --untracked-files=no -- . ':!include' | wc -l | tr -d ' ')"
  echo "submodule: $(git -C 3rdparty/llama.cpp rev-parse --short HEAD), uncommitted files: $(git -C 3rdparty/llama.cpp status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
  echo "chip: $(sysctl -n machdep.cpu.brand_string)"
  echo "power: $(pmset -g batt | head -1)"
  echo "low power mode: $(pmset -g | grep -i lowpowermode)"
  echo "thermal: $(pmset -g therm | tr '\n' ' ')"
  echo "command: $*"
  echo "-----"
} > "$out"
"$@" 2>&1 | tee -a "$out"
