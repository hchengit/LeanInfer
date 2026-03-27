#!/bin/bash
# LeanInfer Phase 2c: Calibration corpus for OLMoE expert activation profiling
# Runs diverse prompts through the model and appends activations to the log.

MODEL=/home/junc/LeanInfer/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf
BIN=/home/junc/LeanInfer/upstream/build/bin/llama-cli
LOG=/home/junc/LeanInfer/profiles/expert_activations.log
THREADS=8

run() {
  "$BIN" -m "$MODEL" --expert-log "$LOG" -p "$1" -n "$2" \
    --threads $THREADS --no-display-prompt 2>/dev/null
}

# Truncate log for fresh calibration
> "$LOG"

echo "Running calibration prompts..."

run "The French Revolution began in 1789 and fundamentally transformed" 256
run "To implement a binary search tree in Python, you need to" 256
run "The human immune system defends the body by" 256
run "Quantum mechanics describes the behavior of particles at" 256
run "Write a function to compute the Fibonacci sequence:" 256
run "The GDP of the United States in 2023 was approximately" 256
run "Shakespeare's Hamlet famously asks 'To be or not to be'" 256
run "Neural networks learn by adjusting weights through" 256
run "The speed of light in a vacuum is approximately" 256
run "Climate change is primarily caused by" 256

LINES=$(wc -l < "$LOG")
echo "Calibration done. $LINES activation records collected."
