#!/usr/bin/env bash
# Run-any-Python-by-number for macOS (Bash 3.2 compatible)
# - Recursively scans THIS FOLDER for *.py (case-insensitive)
# - Skips common bulky dirs (venv, node_modules, .git, etc.)
# - Lists results by number and runs the selected one with python3
# - Keeps the window open at the end if launched by double-click

set -euo pipefail

PY_INTERP="${PY_INTERP:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Dirs to prune from search
SKIP_DIRS=(.git node_modules venv .venv __pycache__ dist build .mypy_cache .next .parcel-cache)

# Build BSD `find` prune expression
PRUNE_EXPR=()
for d in "${SKIP_DIRS[@]}"; do PRUNE_EXPR+=( -name "$d" -o ); done
# remove trailing -o, if any
if [[ ${#PRUNE_EXPR[@]} -gt 0 ]]; then unset 'PRUNE_EXPR[${#PRUNE_EXPR[@]}-1]'; fi

# Collect *.py files recursively (BSD find; no GNU-only flags)
files=()
while IFS= read -r f; do files+=( "$f" ); done < <(
  find "$HERE" \( "${PRUNE_EXPR[@]}" \) -prune -o -type f -iname '*.py' -print 2>/dev/null
)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No Python files found under: $HERE"
  if [[ -z "${PS1:-}" ]]; then echo; read -n 1 -s -p "Press any key to close..."; fi
  exit 0
fi

# Show numbered list (relative paths)
echo "Python files under: $HERE"
echo "------------------------------------------------------------"
i=1
for abs in "${files[@]}"; do
  rel="${abs#$HERE/}"
  printf "%3d) %s\n" "$i" "$rel"
  i=$((i+1))
done
echo "------------------------------------------------------------"

# Choose file
max_idx=${#files[@]}
while true; do
  read -r -p "Enter number to run (1..$max_idx, or 'q' to quit): " choice
  case "$choice" in
    q|Q) exit 0 ;;
    ''|*[!0-9]*) echo "Invalid selection."; continue ;;
    *)  if (( choice>=1 && choice<=max_idx )); then
          idx=$((choice-1)); break
        else
          echo "Pick 1..$max_idx or 'q'."
        fi
        ;;
  esac
done

target="${files[$idx]}"

# Optional args to the Python script
read -r -p "Args for $(basename "$target") (leave blank for none): " args || true

echo "------------------------------------------------------------"
echo "Running: $PY_INTERP \"$target\" $args"
echo "------------------------------------------------------------"

# Run from the script's own directory so relative imports/paths work
script_dir="$(cd "$(dirname "$target")" && pwd)"
script_base="$(basename "$target")"
(
  cd "$script_dir"
  # shellcheck disable=SC2086
  "$PY_INTERP" "$script_base" $args
)

# Keep the window open if launched by double-click (no interactive PS1)
if [[ -z "${PS1:-}" ]]; then
  echo
  read -n 1 -s -p "Done. Press any key to close..."
fi
