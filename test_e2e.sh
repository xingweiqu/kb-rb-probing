#!/bin/bash
# End-to-end test script for atomic capability benchmark MVP

set -e

echo "=========================================="
echo "Atomic Capability Benchmark MVP - E2E Test"
echo "=========================================="

# Configuration
OUTPUT_DIR="./test_output"
MODEL_NAME="Qwen/Qwen3-8B"
KB_COUNT=5
RB_COUNT=5
HYBRID_COUNT=5

echo ""
echo "Configuration:"
echo "  Output dir: $OUTPUT_DIR"
echo "  Model: $MODEL_NAME"
echo "  Families: KB=$KB_COUNT, RB=$RB_COUNT, Hybrid=$HYBRID_COUNT"
echo ""

# Clean previous test output
if [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning previous test output..."
    rm -rf "$OUTPUT_DIR"
fi

# Stage 1: Generate dataset
echo ""
echo "=========================================="
echo "Stage 1: Generating dataset"
echo "=========================================="
cd dataset_synthesis_mvp
python generate.py \
    --output_dir "../$OUTPUT_DIR" \
    --kb $KB_COUNT \
    --rb $RB_COUNT \
    --hybrid $HYBRID_COUNT

cd ..

# Check output
DATASET_FILE="$OUTPUT_DIR/output/dataset.jsonl"
if [ ! -f "$DATASET_FILE" ]; then
    echo "ERROR: Dataset file not found: $DATASET_FILE"
    exit 1
fi

ITEM_COUNT=$(wc -l < "$DATASET_FILE")
echo ""
echo "✓ Dataset generated: $ITEM_COUNT items"

# Stage 2: Extract hidden states
echo ""
echo "=========================================="
echo "Stage 2: Extracting hidden states"
echo "=========================================="
cd probing_mvp
python extract_hidden_states.py \
    --model_name "$MODEL_NAME" \
    --dataset "../$DATASET_FILE" \
    --output "../$OUTPUT_DIR/hidden_states.pkl" \
    --device cuda

cd ..

# Check output
if [ ! -f "$OUTPUT_DIR/hidden_states.pkl" ]; then
    echo "ERROR: Hidden states file not found"
    exit 1
fi

echo ""
echo "✓ Hidden states extracted"

# Stage 3: Derive labels
echo ""
echo "=========================================="
echo "Stage 3: Deriving atomic labels"
echo "=========================================="
cd probing_mvp
python derive_labels.py \
    --dataset "../$DATASET_FILE" \
    --model_outputs "../$OUTPUT_DIR/model_outputs.jsonl" \
    --output "../$OUTPUT_DIR/atomic_labels.json"

cd ..

# Check output
if [ ! -f "$OUTPUT_DIR/atomic_labels.json" ]; then
    echo "ERROR: Atomic labels file not found"
    exit 1
fi

echo ""
echo "✓ Atomic labels derived"

# Stage 4: Train probes
echo ""
echo "=========================================="
echo "Stage 4: Training linear probes"
echo "=========================================="
cd probing_mvp
python linear_probe.py \
    --hidden_states "../$OUTPUT_DIR/hidden_states.pkl" \
    --labels "../$OUTPUT_DIR/atomic_labels.json" \
    --output "../$OUTPUT_DIR/probing_results.json"

cd ..

# Check output
if [ ! -f "$OUTPUT_DIR/probing_results.json" ]; then
    echo "ERROR: Probing results file not found"
    exit 1
fi

echo ""
echo "✓ Linear probes trained"

# Stage 5: Generate report
echo ""
echo "=========================================="
echo "Stage 5: Generating report"
echo "=========================================="
cd probing_mvp
python report.py \
    --results "../$OUTPUT_DIR/probing_results.json" \
    --labels "../$OUTPUT_DIR/atomic_labels.json" \
    --output "../$OUTPUT_DIR/report.json"

cd ..

# Check output
if [ ! -f "$OUTPUT_DIR/report.json" ]; then
    echo "ERROR: Report file not found"
    exit 1
fi

if [ ! -f "$OUTPUT_DIR/report.html" ]; then
    echo "ERROR: HTML report file not found"
    exit 1
fi

echo ""
echo "✓ Report generated"

# Summary
echo ""
echo "=========================================="
echo "E2E Test Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  Dataset: $DATASET_FILE ($ITEM_COUNT items)"
echo "  Hidden states: $OUTPUT_DIR/hidden_states.pkl"
echo "  Model outputs: $OUTPUT_DIR/model_outputs.jsonl"
echo "  Atomic labels: $OUTPUT_DIR/atomic_labels.json"
echo "  Probing results: $OUTPUT_DIR/probing_results.json"
echo "  Report (JSON): $OUTPUT_DIR/report.json"
echo "  Report (HTML): $OUTPUT_DIR/report.html"
echo ""
echo "View HTML report:"
echo "  open $OUTPUT_DIR/report.html"
echo ""
