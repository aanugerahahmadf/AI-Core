#!/bin/bash
# Generate Python gRPC code from proto files
# Wedding Flower Decoration App - AI Core

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$SCRIPT_DIR/protos"
OUTPUT_DIR="$SCRIPT_DIR/generated"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate Python gRPC code
python -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    "$PROTO_DIR/cbir.proto"

echo "Python gRPC code generated successfully!"
echo "Output directory: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"