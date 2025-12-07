#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 <email_address>"
    exit 1
fi

EMAIL="$1"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sbatch --mail-user="$EMAIL" "$SCRIPT_DIR/train_unet.sbatch"