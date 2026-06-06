#!/usr/bin/env bash

# Exit on error, undefined variable, and pipe failure
set -euo pipefail

# --- Configuration -------------------------------------------------
# Zenodo record ID in the record URL: 20574743
# URL: https://zenodo.org/records/20574743
# 
ZENODO_RECORD_ID="20574743" 

# Output directory for downloaded zip files
OUTPUT_DIR="."

# --- Helper functions ---------------------------------------------
die() {
    echo "ERROR: $*" >&2
    exit 1
}

# --- Main script --------------------------------------------------
echo "Fetching Zenodo record $ZENODO_RECORD_ID ..."

# Get record metadata from Zenodo API
API_URL="https://zenodo.org/api/records/$ZENODO_RECORD_ID"
API_RESPONSE=$(curl --fail --silent --location "$API_URL") \
    || die "Failed to fetch API response from $API_URL"

# Extract the archive (zip) download URL using jq
if ! command -v jq &> /dev/null; then
    die "jq is required but not installed. Install with: sudo apt install jq (Ubuntu) or brew install jq (macOS)"
fi

DOWNLOAD_URL=$(echo "$API_RESPONSE" | jq -r '.links.archive')
if [[ -z "$DOWNLOAD_URL" || "$DOWNLOAD_URL" == "null" ]]; then
    die "Could not find archive download link in API response."
fi

echo "Found download URL: $DOWNLOAD_URL"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download the zip file
ZIP_FILENAME="zenodo_${ZENODO_RECORD_ID}.zip"
echo "Downloading to $OUTPUT_DIR/$ZIP_FILENAME ..."
curl --fail --progress-bar --location --output "$OUTPUT_DIR/$ZIP_FILENAME" "$DOWNLOAD_URL"

echo "Download completed: $OUTPUT_DIR/$ZIP_FILENAME"

# Optional: Unzip the file (remove the '# ' to enable)
# echo "Unzipping into $OUTPUT_DIR ..."
unzip -q "$OUTPUT_DIR/$ZIP_FILENAME" -d "$OUTPUT_DIR" \
    || die "Unzip failed."
rm "$OUTPUT_DIR/$ZIP_FILENAME"   # optionally remove the zip after extraction