# DM-recommendation-system

취향을 맞춘 건강한 레시피 추천 시스템 (아주대학교 SCE331 데이터마이닝 팀 프로젝트)

> 초안이다. "작성 예정"으로 표시한 절은 해당 단계가 끝나면 채운다.

## Overview

Food.com 레시피와 사용자 상호작용 데이터로, 사용자의 취향에 맞으면서 영양 면에서 더 나은 레시피를 추천한다.
상호작용은 implicit feedback으로 다룬다.

파이프라인은 다음과 같다.

1. 전처리: 레시피의 영양·재료·태그를 파싱하고, 상호작용을 정제·필터링·분할한다.
2. MinHash + LSH: 사용자가 좋아한 레시피와 재료 집합이 비슷한 레시피를 후보로 검색한다.
3. k-means: 레시피를 영양 벡터로 군집화해 영양 유형을 만든다.
4. Matrix factorization: 사용자별 후보 레시피의 취향 점수를 계산한다.
5. Re-ranking: 취향 점수와 건강 지표를 결합해 최종 추천 목록을 만든다.

## Team

팀 번호: 작성 예정

| 이름 | 학번 | 담당 |
|---|---|---|
| 오현성 | 202220719 | 전처리, MinHash + LSH |
| 장민우 | 작성 예정 | k-means, 발표 |
| 김수빈 | 작성 예정 | Matrix factorization, 평가 |

## Repository layout

```
configs/default.yaml     실험 설정 (seed, 경로, 전처리·분할·MinHash·LSH 값)
src/dmrec/               팀이 작성한 코드 (전처리, MinHash, LSH)
scripts/                 실행 스크립트 (데이터 다운로드·검증 등)
tests/                   단위 테스트
data/foodcom/            원본 데이터 (커밋하지 않음, 체크섬만 커밋)
data/processed/          가공 산출물 (커밋하지 않음)
reports/figures/         보고서용 그림
docs/                    데이터 출처, 산출물 스키마, 전처리 결정 문서
requirements.txt         실행 환경
requirements-dev.txt     검증용 비교 라이브러리
```

k-means와 matrix factorization 코드의 위치는 담당자가 추가한다.

## Environment/setup

Python 3.13에서 확인했다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MinHash/LSH 구현을 라이브러리와 비교 검증할 때만 `pip install -r requirements-dev.txt`를 쓴다.

## Data sources and acquisition

- 데이터: Food.com Recipes and Interactions [1]
- 제공 페이지: https://cseweb.ucsd.edu/~jmcauley/datasets.html#foodcom
- 다운로드 페이지: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
- 접근일: 2026-11-09
- 규모: 레시피 231,637개, 사용자 226,570명, 상호작용 1,132,367건
- 사용 파일: `RAW_recipes.csv`, `RAW_interactions.csv`

원본 파일은 저장소에 포함하지 않는다. 아래 명령으로 내려받고 체크섬을 확인한다.

```bash
bash scripts/download_data.sh             # data/foodcom/ 에 내려받고 검증
python3 scripts/verify_data.py --counts   # 이미 받은 파일을 다시 검증
```

다운로드가 실패하면 Kaggle 페이지에서 직접 내려받아 두 CSV를 `data/foodcom/`에 둔다.
출처, license, 컬럼, 체크섬은 [docs/data_source.md](docs/data_source.md)에 정리했다.

## Preprocessing

작성 예정. 설정값은 `configs/default.yaml`의 `preprocess`, `split`에 있다.

## Algorithms

수업에서 다룬 알고리즘 세 가지를 쓴다.

| 알고리즘 | 역할 | 입력 |
|---|---|---|
| MinHash + LSH | 재료가 비슷한 레시피를 후보로 검색 | 레시피의 재료 집합 (Jaccard 유사도) |
| k-means | 레시피의 영양 유형 군집화 | 영양 벡터 |
| Regularized latent-factor matrix factorization (user/item bias 포함) | 사용자별 취향 점수 | 사용자-레시피 상호작용 |

MinHash와 LSH는 직접 구현한다. 구현 설명: 작성 예정

## Experiment commands/configurations

모든 설정은 `configs/default.yaml`에 있고 random seed는 `seed` 값으로 고정한다.

```bash
python3 -m pytest          # 단위 테스트
```

단계별 실행 명령: 작성 예정

## Evaluation

작성 예정

## Reproducing tables/figures

작성 예정

## References and external code

[1] B. P. Majumder, S. Li, J. Ni, and J. McAuley, "Generating personalized recipes from historical user preferences," in *Proc. EMNLP*, 2019.

외부 코드: 현재 없음. 사용하게 되면 출처와 license를 여기에 적는다.

## AI-use statement

작성 예정
