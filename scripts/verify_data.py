"""원본 데이터가 data/foodcom/source_manifest.json 과 같은지 확인한다.

사용법:
    python3 scripts/verify_data.py                 # 크기와 sha256
    python3 scripts/verify_data.py --counts        # 행 수, 컬럼, 고유 사용자 수까지
    python3 scripts/verify_data.py --dir /tmp/dl   # 다른 폴더에 받은 파일을 대조
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmrec.data_check import check_counts, check_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "data" / "foodcom", help="CSV 가 있는 폴더")
    parser.add_argument("--counts", action="store_true", help="행 수, 컬럼, 고유 사용자 수도 확인")
    args = parser.parse_args()

    manifest = REPO_ROOT / "data" / "foodcom" / "source_manifest.json"
    problems = check_files(manifest, args.dir)
    if not problems and args.counts:
        problems = check_counts(manifest, args.dir)

    if problems:
        print("불일치:")
        for problem in problems:
            print(" -", problem)
        return 1
    print("일치: 크기, sha256" + (", 행 수, 컬럼, 고유 사용자 수" if args.counts else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
