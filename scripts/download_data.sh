#!/usr/bin/env bash
# Food.com 원본 데이터를 Kaggle 에서 내려받아 RAW_recipes.csv, RAW_interactions.csv 를 꺼낸다.
#
# 사용법:
#   bash scripts/download_data.sh              # data/foodcom 에 받는다
#   bash scripts/download_data.sh /tmp/dl      # 다른 폴더에 받는다 (기존 파일과 대조할 때)
#
# curl 다운로드가 로그인 요구로 실패하면 아래 페이지에서 직접 내려받아
# 두 CSV 를 대상 폴더에 넣은 뒤 scripts/verify_data.py 를 실행한다.
#   https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$REPO_ROOT/data/foodcom}"
URL="https://www.kaggle.com/api/v1/datasets/download/shuyangli94/food-com-recipes-and-user-interactions"

mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -L --fail -o "$TMP/foodcom.zip" "$URL"
unzip -o -j "$TMP/foodcom.zip" RAW_recipes.csv RAW_interactions.csv -d "$DEST"

python3 "$REPO_ROOT/scripts/verify_data.py" --dir "$DEST"
