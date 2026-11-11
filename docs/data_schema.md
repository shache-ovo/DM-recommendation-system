# 전처리 산출물 스키마와 사용법

`python3 scripts/preprocess.py`가 `data/processed/`에 만드는 파일의 설명이다.
파일은 git에 올리지 않는다. 저장소를 받은 뒤 아래 순서로 직접 만든다(합쳐서 1~2분).

```bash
pip install -r requirements.txt
bash scripts/download_data.sh     # 원본 2개를 data/foodcom/ 에 받고 체크섬 확인
python3 scripts/preprocess.py     # data/processed/ 에 6개 파일 생성
```

기본 설정으로 실행하면 행 수가 아래 표와 같아야 한다. 다르면 설정이나 원본이 다른 것이다.

## 파일 목록

| 파일 | 행 수 | 한 행의 의미 |
|---|---|---|
| `recipes.csv` | 231,352 | 레시피 1개 (칼로리 기준을 통과한 전체 레시피) |
| `users.csv` | 16,032 | 사용자 1명 (5-core에 남은 사용자) |
| `interactions_train.csv` | 466,136 | 양성 상호작용 1건 |
| `interactions_valid.csv` | 16,029 | 사용자별 validation 정답 1건 |
| `interactions_test.csv` | 16,028 | 사용자별 test 정답 1건 |
| `interactions_seen_non_positive.csv` | 30,448 | 양성이 아닌 상호작용 1건 |

모든 파일은 UTF-8 CSV이고 첫 행이 컬럼 이름이다.

## recipes.csv

| 컬럼 | 형식 | 설명 |
|---|---|---|
| `recipe_id` | int | 원본의 레시피 id |
| `recipe_idx` | int | 추천 대상 레시피의 연속 번호(0 ~ 37,817). 추천 대상이 아니면 -1 |
| `n_train` | int | train에서 이 레시피를 양성으로 평가한 사용자 수. 추천 대상이 아니면 0 |
| `name` | str | 레시피 이름 (결측 1건) |
| `minutes` | int | 조리 시간(분). 오류값이 있어 이 프로젝트에서는 쓰지 않는다 |
| `n_steps` | int | 조리 단계 수 |
| `n_ingredients` | int | 재료 수 |
| `calories` | float | 칼로리 (원출처 표기: #) |
| `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrates` | float | PDV(%) |
| `ingredients` | str | 재료 이름의 JSON 리스트. 원본 표기 그대로이며 정규화하지 않았다 |
| `tags` | str | 태그의 JSON 리스트 |

- **추천 대상 레시피**는 `recipe_idx >= 0`인 37,818개다. train에 한 번 이상 등장한 레시피다.
- 나머지 레시피(`recipe_idx == -1`)는 상호작용이 부족해 걸러졌지만 영양·재료 정보는 남겨 두었다. 전체 레시피의 영양 분포를 보거나 LSH의 규모 실험을 할 때 쓴다.
- 영양 값은 변환하지 않은 원본 값이다. 분포가 오른쪽으로 길게 늘어져 있으므로(`docs/eda.md` 5절) 표준화 전에 로그 변환이나 절단을 검토한다.

## users.csv

| 컬럼 | 형식 | 설명 |
|---|---|---|
| `user_id` | int | 원본의 사용자 id |
| `user_idx` | int | 연속 번호(0 ~ 16,031) |
| `n_train` | int | train에서 이 사용자가 양성으로 평가한 레시피 수 (최소 3, 중앙값 8, 최대 2,724) |

## interactions_*.csv

네 파일의 컬럼은 같다.

| 컬럼 | 형식 | 설명 |
|---|---|---|
| `user_id` | int | 사용자 id |
| `recipe_id` | int | 레시피 id |
| `date` | str | 날짜(YYYY-MM-DD). 일 단위 |
| `rating` | int | 원본 평점. train/valid/test는 4~5, seen_non_positive는 0~3 |

- **train / valid / test**: 평점 4 이상인 양성 상호작용이다. 사용자별로 시간순 정렬해 마지막 1건이 test, 그 앞 1건이 valid, 나머지가 train이다. implicit feedback으로 다루므로 `rating` 값 자체는 학습에 쓰지 않는다.
- **seen_non_positive**: 5-core 사용자가 추천 대상 레시피에 남긴 평점 0~3 기록이다. 양성은 아니지만 사용자가 이미 본 레시피이므로, 추천 후보와 negative 샘플에서 제외할 때 쓴다. 평점 0은 별점 없이 글만 남긴 리뷰다.
- 네 파일 사이에 같은 (user_id, recipe_id) 쌍은 없다.
- valid와 test의 모든 사용자와 레시피는 train에 등장한다. cold 사용자·레시피는 없다.
- 평가 사용자 수가 16,032명보다 3~4명 적은 이유: 그 사용자의 정답 레시피가 train에 없어서 뺐다.

## 사용 예

```python
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

P = "data/processed/"
recipes = pd.read_csv(P + "recipes.csv")
users = pd.read_csv(P + "users.csv")
train = pd.read_csv(P + "interactions_train.csv")
valid = pd.read_csv(P + "interactions_valid.csv")
test = pd.read_csv(P + "interactions_test.csv")
seen = pd.read_csv(P + "interactions_seen_non_positive.csv")

# 추천 대상 레시피. i번째 행이 recipe_idx == i 가 되도록 정렬한다
core = recipes[recipes["recipe_idx"] >= 0].sort_values("recipe_idx").reset_index(drop=True)

# id -> 연속 번호
user_idx = dict(zip(users["user_id"], users["user_idx"]))
recipe_idx = dict(zip(core["recipe_id"], core["recipe_idx"]))
```

**Matrix factorization용 행렬** (사용자 × 레시피, 양성이면 1):

```python
rows = train["user_id"].map(user_idx).to_numpy()
cols = train["recipe_id"].map(recipe_idx).to_numpy()
R = csr_matrix((np.ones(len(train), dtype=np.float32), (rows, cols)),
               shape=(len(users), len(core)))      # (16032, 37818), 466,136개
```

**k-means용 영양 행렬**:

```python
NUTRITION = ["calories", "total_fat", "sugar", "sodium", "protein", "saturated_fat", "carbohydrates"]
X = core[NUTRITION].to_numpy()                      # (37818, 7), 행 번호 == recipe_idx
```

**재료와 태그**:

```python
ingredients = core["ingredients"].map(json.loads)   # 각 원소가 문자열 리스트
```

**사용자별 추천 제외 목록** (이미 본 레시피):

```python
excluded = pd.concat([train, seen]).groupby("user_id")["recipe_id"].agg(set)
```

**인기순 baseline**:

```python
popular = core.sort_values("n_train", ascending=False)["recipe_id"].tolist()
```

## 평가할 때 지킬 점

- 하이퍼파라미터와 λ는 valid로 고르고, test는 최종 보고에만 쓴다.
- 인기도, 군집, 건강 지표의 기준값처럼 데이터에서 계산하는 값은 train과 레시피 속성만으로 만든다. valid와 test의 상호작용을 쓰지 않는다.
- 순위를 매길 때 사용자가 이미 본 레시피(위 `excluded`)는 후보에서 뺀다. test를 평가할 때는 그 사용자의 valid 레시피도 뺀다.
- 사용자별 분할이라서, 어떤 사용자의 test 시점보다 나중에 일어난 다른 사용자의 기록이 train에 들어 있다. 보고서의 한계에 적는다.

## MinHash + LSH 산출물 (예정)

재료 유사 레시피 검색 결과는 아직 없다. 구현이 끝나면 이 절에 파일 형식과 조회 함수를 추가한다.
