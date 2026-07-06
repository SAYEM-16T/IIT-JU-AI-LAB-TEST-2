#!/usr/bin/env bash
# Jupyter Notebook launcher for LAB-FINAL-2 (venv + ML stack already installed).
# Run:  bash start-jupyter.sh
#
# Note: is system-e 'text/html' er default handler RedisInsight-e set kora, tai
# Jupyter-er default auto-open RedisInsight khule fele. Tai ekta REAL browser force kori.
cd "$(dirname "$0")" || exit 1

# Prothome je real browser paoa jabe, oita BROWSER hisebe set kori (RedisInsight bypass)
for b in google-chrome google-chrome-stable firefox brave-browser microsoft-edge chromium; do
  if command -v "$b" >/dev/null 2>&1; then export BROWSER="$b"; break; fi
done

echo "Starting Jupyter Notebook from LAB-FINAL-2 venv..."
echo "Browser: ${BROWSER:-(system default)} e khule jabe. Bondho korte terminal e Ctrl+C dao."
echo "Jodi browser na khule, niche 'http://localhost:8888/...token...' URL ta copy kore browser e paste koro."
exec ./.venv/bin/jupyter notebook
