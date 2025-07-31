# ML_MINI_2TEAM
--------
## 💡팀명
#### 2DA
"우리는 데이터를 분석하는 데서 멈추지 않고, 인사이트를 통해 행동으로 옮깁니다."

## ✨팀원 소개

| [최우진](https://github.com/CHUH00) | [조세희](https://github.com/SEHEE-8546) | [정의중](https://github.com/uii42) | [박민정](https://github.com/minjeon) | [맹지수](https://github.com/happyfrogg) |
|-------|-------|-------|-------|-------|
| <img src="images/cwj.png" width="120"/> | <img src="images/csh.png" width="120"/> | <img src="images/juj.png" width="120"/> | <img src="images/pmj.png" width="120"/> | <img src="images/mjs.png" width="120"/> |

--------

## <프로젝트 개요>
### 🌟프로젝트명
UFC 선수의 승리 할 확률 예측
### 📌프로젝트 소개
UFC 선수의 승, 패, 무승부, 키, 몸무게, 리치와 같은 선수들의 경기 정보를 바탕으로 승리할 확률을 예측하는 프로젝트입니다. 
### 🪟프로젝트 필요성
**1. 베팅 시장 성장** <br>
   
   특히 모바일 및 온라인 플랫폼의 기술 발전으로 인해 베팅에 대한 접근성이 높아지고 편리해졌습니다.스포츠의 인기가 높아지고 전 세계적으로 스포츠 애호가의 수가 늘어나면서 시장 성장이 더욱 가속화되고 있습니다. 또한 베팅 회사와 스포츠 리그 또는 팀 간의 전략적 파트너십을 통해 시장 가시성과 신뢰성이 향상됩니다. UFC는 전 세계적으로 엄청난 인기를 끌고 있는 스포츠로, 팬들은 항상 경기의 결과를 예측하고 토론하는 것을 즐깁니다. 데이터 기반의 정량적인 분석을 통해 승률을 예측하는 모델을 구축한다면, 팬들에게는 더 큰 재미와 깊이 있는 정보를 제공하고, 관련 산업(미디어, 이벤트, 인기선수 승률 예측을 통한 흥행 예측 등)에는 새로운 분석 관점을 제시할 수 있을 것입니다.
   <img width="1358" height="1016" alt="image" src="https://github.com/user-attachments/assets/f1c0c47b-10e9-4753-b711-0627106d5f4f" />
 <br>
**2. 숨겨진 승패 결정 요인 발굴 및 예측 정확도 향상** <br>
   
   '이변'의 발생은 리치 외에 간과되었던 다른 중요한 요인들(예: 경기 스타일 상성, 타격 정확도, 테이크다운 방어율 등)이 승률에 큰 영향을 미칠 수 있음을 시사합니다. 본 프로젝트는 이러한 숨겨진 승패 결정 요인들을 데이터 기반으로 발굴하고, 이를 예측 모델에 통합함으로써 기존의 직관적인 예측보다 훨씬 더 정확하고 신뢰할 수 있는 승률 예측을 가능하게 합니다.
   
--------
## <기술 스택>
<img src="https://img.shields.io/badge/Python-3776AB?style=plastic&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/pandas-150458?style=plastic&logo=pandas&logoColor=white"> <img src="https://img.shields.io/badge/github-181717?style=plastic&logo=github&logoColor=white"> <img src="https://img.shields.io/badge/numpy-013243?style=plastic&logo=numpy&logoColor=white"> <img src="https://img.shields.io/badge/matplotlib-11557c?style=plastic&logo=matplotlib&logoColor=white"> <img src="https://img.shields.io/badge/seaborn-0C5A5A?style=plastic&logoColor=white"> <img src="https://img.shields.io/badge/scikitlearn-green?style=plastic&logo=scikitlearn&logoColor=white"/>

--------
## <EDA(탐색적 데이터 분석)>
**1. Features 간의 상관관계 히트맵** <br>
<img width="1046" height="927" alt="eda_1" src="https://github.com/user-attachments/assets/dc03ac1c-d872-423f-b0a9-636ed7b23554" /> <br>

---------

## <사용한 데이터셋 및 데이터 전처리>
### 1. 사용한 데이터셋
- **데이터명** : UFC DATASETS <br>

- **데이터 출처** : https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025/data

### 2. 데이터 전처리
- **결측치 처리** :
  - 무승부, 생년월일, 주요자세(stance), 키, 몸무게의 결측치 제거 및 reach 결측치 키 값으로 대체
    <img width="638" height="71" alt="image" src="https://github.com/user-attachments/assets/b260d8a9-d7eb-4b75-9adb-91f077acaef8" />

- **새로운 특성 생성** :
  - 선수 간의 특성 차이 생성
    <img width="655" height="213" alt="image" src="https://github.com/user-attachments/assets/f8d6bd3d-e788-4c18-b22c-6f850171d389" />

  - BMI, 총 경기 수, 공격 점수, 방어 점수, 순공격 이득, 공격/방어 스코어 비율, 타격 효율 차이를 파생 변수로 생성
    <img width="653" height="457" alt="image" src="https://github.com/user-attachments/assets/93c7721b-95eb-45ea-8d2e-4c9e5a735db9" />

 - **나이 변환** :
   - 생년월일을 나이값으로 변환
     <img width="650" height="114" alt="image" src="https://github.com/user-attachments/assets/71fbb1fa-a8da-4f68-912a-6ed1fb077152" />

--------
<h2> <모델 선정 과정> </h2>

<h3>1. Logistic Regression</h3>
<div align="left">
  <img width="278" height="226" alt="image" src="https://github.com/user-attachments/assets/527d682b-9eef-412b-934d-d166ed047036" width="500">
</div>
<div> - 피드백 : 복잡한 비선형 관계를 반영하기에는 한계가 있어, 다른 트리 기반 모델에 비해 상대적으로 낮은 정확도를 보였습니다. </div>

<h3>2. Random Forest Classifier</h3>
<div align="left">
  <img width="278" height="226" alt="image" src="https://github.com/user-attachments/assets/280829f3-0d8c-473e-8e08-ed0517c59433" width="500">
</div>  
<div> - 피드백 : Logistic보다는 나은 성능을 보였으며, 다양한 피처를 반영했을 때 비교적 안정적인 예측 성능을 보였습니다. 다만 피처 수가 많아 일부 정보가 희석될 가능성이 있습니다. </div>

<h3>3. XGBoost</h3>
<div align="left">
  <img width="278" height="226" alt="image" src="https://github.com/user-attachments/assets/e6503bf7-2cce-4a05-aa85-6aabcc772103" width="500">
</div> 
<div> - 피드백 : 클래스 불균형을 고려한 하이퍼파라미터 튜닝과 결합해 예측 오차를 줄이는 데 효과적이었습니다. </div>

--------
## <선정된 모델>
- Stacking (LogisticRegression, RandomForest, xgboost, lightGBM)
  <img width="1426" height="835" alt="image" src="https://github.com/user-attachments/assets/3066d51d-3635-4a62-a640-83b1701f0e56" />

--------
## <평가>
- accuracy : 전체적인 성능
- ROC-AUC : 이진 분류의 분리 성능
- F1-score : 정밀도와 재현율의 균형
  <img width="549" height="298" alt="image" src="https://github.com/user-attachments/assets/86673645-8736-4025-b6bf-abe25b4ec6f2" /> <br>
  
- 혼동 행렬 시각화
  <img width="705" height="594" alt="image" src="https://github.com/user-attachments/assets/28254696-476e-415b-b08c-56002407c520" />

--------
## <평가 성능 향상을 위한 노력>
- RandomizedSearchCV 사용
  <img width="1414" height="237" alt="image" src="https://github.com/user-attachments/assets/05ddb3f5-cf01-43d5-a8d8-a04733d2be7c" />

--------
## <한 줄 회고록>

🐷우진 : <br>
🐷지수 : <br>
🐷민정 : 프로젝트 기간 중에 주제 개요 및 선정 시간이 가장 오래 걸리고 그만큼 중요하다는 것을 다시 한 번 체감할 수 있었습니다. 주제에 대해 여러 방면으로 이 프로젝트에 이 주제가 적합한가, 왜 이 프로젝트가 필요한가 등과 관련하여 주제 적합성을 검토하는 과정이 특히 중요하다고 느꼈습니다. 또한, 컬럼이 워낙 많았어서 라벨 데이터와 관련이 높은 피처 값을 판단해 의미 없는 컬럼은 제거하고 중요한 컬럼은 결측치를 의미 있는 방식으로 대체하는 것이 모델 성능에 영향을 끼친다는 것을 배웠습니다. 성능이 좋은 모델을 선택하기 위해 어떤 지표로 성능을 평가할 것인지, 어떤 모델을 사용해볼 것인지 고민해볼 수 있는 시간이었습니다. <br>
🐷세희 : <br>
🐷의중 : <br>
