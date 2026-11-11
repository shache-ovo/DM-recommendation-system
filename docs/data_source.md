# 데이터 출처와 획득

## 출처

| 항목 | 내용 |
|---|---|
| 데이터셋 | Food.com Recipes and Interactions |
| 원저작자 | B. P. Majumder, S. Li, J. Ni, J. McAuley (EMNLP 2019) [1] |
| 제공 페이지 | https://cseweb.ucsd.edu/~jmcauley/datasets.html#foodcom |
| 다운로드 페이지 | https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions |
| 접근일 | 2026-11-09 (다운로드), 2026-11-11 (검증) |
| 버전 | 별도 버전 표기 없음. Kaggle 페이지의 예상 갱신 주기는 "Never" |
| License | Kaggle 페이지 표기: "Data files © Original Authors" |

UCSD 제공 페이지는 파일을 직접 배포하지 않고 Kaggle의 데이터셋 페이지로 안내한다.
License가 재배포를 허용한다고 명시하지 않으므로 원본과 가공 데이터는 이 저장소에 올리지 않는다.

## 사용 파일

Kaggle 배포본의 8개 파일 중 원본 2개만 쓴다.
저자들이 만든 분할 파일(`interactions_*.csv`)과 전처리본(`PP_*.csv`)은 쓰지 않는다.

| 파일 | 크기 | 행 수 | 컬럼 |
|---|---|---|---|
| `RAW_recipes.csv` | 294,520,189 bytes | 231,637 | name, id, minutes, contributor_id, submitted, tags, nutrition, n_steps, steps, description, ingredients, n_ingredients |
| `RAW_interactions.csv` | 349,436,524 bytes | 1,132,367 | user_id, recipe_id, date, rating, review |

- 고유 사용자 226,570명, 고유 레시피 231,637개. 모든 상호작용의 `recipe_id`가 `RAW_recipes.csv`에 있다.
- 상호작용 날짜 범위: 2000-01-25 ~ 2018-12-20.
- 위 수치는 UCSD 제공 페이지의 기재 수치(레시피 231,637 / 사용자 226,570 / 리뷰 1,132,367)와 같다.
- Kaggle 페이지 소개문의 "180K+ recipes and 700K+ reviews"는 논문 실험에 쓴 전처리본의 규모로, 원본 파일의 규모와 다르다.

## nutrition 필드

`nutrition` 컬럼은 값 7개의 리스트다. 순서는 Kaggle 데이터 카드의 `RAW_recipes.csv` 컬럼 설명을 따른다.

| 순서 | 필드 | 원출처 표기 |
|---|---|---|
| 1 | calories | calories (#) |
| 2 | total_fat | total fat (PDV) |
| 3 | sugar | sugar (PDV) |
| 4 | sodium | sodium (PDV) |
| 5 | protein | protein (PDV) |
| 6 | saturated_fat | saturated fat (PDV) |
| 7 | carbohydrates | carbohydrates (PDV) |

원출처는 1회 제공량 기준을 설명하지 않는다.

## 획득 절차

1. `bash scripts/download_data.sh`를 실행한다. Kaggle에서 압축 파일을 받아 두 CSV를 `data/foodcom/`에 푼 뒤 체크섬을 확인한다.
2. 스크립트의 다운로드가 실패하면 Kaggle 페이지에서 직접 내려받아 두 CSV를 `data/foodcom/`에 넣는다.

## 검증

체크섬과 기대 수치는 `data/foodcom/source_manifest.json`에 있다.

```bash
python3 scripts/verify_data.py            # 크기, sha256
python3 scripts/verify_data.py --counts   # 행 수, 컬럼, 고유 사용자 수까지
```

| 파일 | sha256 |
|---|---|
| `RAW_recipes.csv` | `6a8136d1da9e03396a9f52d72200b3fddb8fe103f11e1496effca30ac4e0539f` |
| `RAW_interactions.csv` | `4e278d2f687b1020f5694e51003a7d6174344b51aebbdf4530de88b5c6a78ee4` |

제공자가 공식 체크섬을 공개하지 않으므로, 이 값은 2026-11-09에 내려받은 파일에서 계산한 것이다.

## 참고 문헌

[1] B. P. Majumder, S. Li, J. Ni, and J. McAuley, "Generating personalized recipes from historical user preferences," in *Proc. EMNLP*, 2019. https://www.aclweb.org/anthology/D19-1613/
