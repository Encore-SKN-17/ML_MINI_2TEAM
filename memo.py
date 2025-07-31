import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    log_loss, brier_score_loss, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/Users/uii/Documents/2DA/ufc/UFC.csv')


# Unnamed 컬럼 제거
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# 무승부 경기 제거
df.dropna(subset=['winner'], inplace=True)
# 필수 정보(DOB) 누락 행 제거
df.dropna(subset=['r_dob', 'b_dob'], inplace=True)
# Stance 결측치 처리
df['r_stance'].fillna('Orthodox', inplace=True)
df['b_stance'].fillna('Orthodox', inplace=True)
# Reach 결측치 처리
df['r_reach'].fillna(df['r_height'], inplace=True)
df['b_reach'].fillna(df['b_height'], inplace=True)
# 나머지 숫자형 결측치 0으로 채우기
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(0)

print(f'최종 결측치 개수: {df.isnull().sum().sum()}')


# 타겟 변수 생성
df['winner_is_red'] = (df['winner'] == df['r_name']).astype(int)

# 나이 계산
df['r_dob'] = pd.to_datetime(df['r_dob'])
df['b_dob'] = pd.to_datetime(df['b_dob'])
df['r_age'] = (pd.to_datetime('today') - df['r_dob']).dt.days / 365.25
df['b_age'] = (pd.to_datetime('today') - df['b_dob']).dt.days / 365.25

# DOB 컬럼은 나이 계산 후 더 이상 필요 없으므로 제거
df.drop(['r_dob', 'b_dob'], axis=1, inplace=True)

# 차이 특성 생성
df['age_diff'] = df['r_age'] - df['b_age']
df['height_diff'] = df['r_height'] - df['b_height']
df['weight_diff'] = df['r_weight'] - df['b_weight']
df['reach_diff'] = df['r_reach'] - df['b_reach']
df['wins_diff'] = df['r_wins'] - df['b_wins']
df['losses_diff'] = df['r_losses'] - df['b_losses']
df['splm_diff'] = df['r_splm'] - df['b_splm']
df['str_acc_diff'] = df['r_str_acc'] - df['b_str_acc']
df['sapm_diff'] = df['r_sapm'] - df['b_sapm']
df['str_def_diff'] = df['r_str_def'] - df['b_str_def']
df['td_avg_diff'] = df['r_td_avg'] - df['b_td_avg']
df['td_acc_diff'] = df['r_td_avg_acc'] - df['b_td_avg_acc']
df['td_def_diff'] = df['r_td_def'] - df['b_td_def']
df['sub_avg_diff'] = df['r_sub_avg'] - df['b_sub_avg']

# ratio features (0~1 scaled)
df['sig_str_ratio'] = df['r_splm'] / (df['r_splm'] + df['b_splm'] + 1e-6)
df['td_ratio'] = df['r_td_avg'] / (df['r_td_avg'] + df['b_td_avg'] + 1e-6)
df['str_acc_ratio'] = df['r_str_acc'] / (df['r_str_acc'] + df['b_str_acc'] + 1e-6)
df['td_acc_ratio'] = df['r_td_avg_acc'] / (df['r_td_avg_acc'] + df['b_td_avg_acc'] + 1e-6)

# win ratio features
df['r_win_ratio'] = df['r_wins'] / (df['r_wins'] + df['r_losses'] + 1e-6)
df['b_win_ratio'] = df['b_wins'] / (df['b_wins'] + df['b_losses'] + 1e-6)
df['win_ratio_diff'] = df['r_win_ratio'] - df['b_win_ratio']

# ---------- 추가 파생 특성 ----------
# BMI
df['r_bmi'] = df['r_weight'] / ((df['r_height'] / 100) ** 2 + 1e-6)
df['b_bmi'] = df['b_weight'] / ((df['b_height'] / 100) ** 2 + 1e-6)
df['bmi_diff'] = df['r_bmi'] - df['b_bmi']

# Reach / Height 비율
df['r_reach_ht_ratio'] = df['r_reach'] / (df['r_height'] + 1e-6)
df['b_reach_ht_ratio'] = df['b_reach'] / (df['b_height'] + 1e-6)
df['reach_ht_ratio_diff'] = df['r_reach_ht_ratio'] - df['b_reach_ht_ratio']


# 총 경기 수
df['r_total_fights'] = df['r_wins'] + df['r_losses']
df['b_total_fights'] = df['b_wins'] + df['b_losses']
df['total_fights_diff'] = df['r_total_fights'] - df['b_total_fights']

# ---------- 추가 파생 특성 v3 ----------
# 1) 공격 점수: Striking + Grappling 효율 합
df['r_offense_score'] = df['r_str_eff'] + df['r_grap_eff'] if 'r_str_eff' in df.columns else \
                        (df['r_splm'] * df['r_str_acc']) + (df['r_td_avg'] * df['r_td_avg_acc'])
df['b_offense_score'] = df['b_str_eff'] + df['b_grap_eff'] if 'b_str_eff' in df.columns else \
                        (df['b_splm'] * df['b_str_acc']) + (df['b_td_avg'] * df['b_td_avg_acc'])
df['offense_score_diff'] = df['r_offense_score'] - df['b_offense_score']

# 2) 방어 점수: 타격·테이크다운 방어율 평균
df['r_defense_score'] = (df['r_str_def'] + df['r_td_def']) / 2
df['b_defense_score'] = (df['b_str_def'] + df['b_td_def']) / 2
df['defense_score_diff'] = df['r_defense_score'] - df['b_defense_score']

# 3) 순공격 이득(Net Advantage) = 공격 diff + 방어 diff
df['net_advantage'] = df['offense_score_diff'] + df['defense_score_diff']

# 4) 상호작용 특성: 레드의 공격 vs 블루의 방어, 블루의 공격 vs 레드의 방어
df['str_vs_def_diff'] = (df['r_str_acc'] * df['b_str_def']) - (df['b_str_acc'] * df['r_str_def'])

# 5) 공격/방어 스코어 비율 차이
df['off_def_ratio_diff'] = (df['r_offense_score'] / (df['r_defense_score'] + 1e-6)) - \
                           (df['b_offense_score'] / (df['b_defense_score'] + 1e-6))


# 스탠스 조합 (범주형)
df['stance_comb'] = df['r_stance'].astype(str) + '_' + df['b_stance'].astype(str)


# 특성(X)과 타겟(y) 정의
numerical_features = [
    col for col in df.columns
    if (df[col].dtype != 'object')
    and col not in ['winner_is_red', 'winner']
]
categorical_features = ['r_stance', 'b_stance', 'stance_comb']
X = df[numerical_features + categorical_features]
y = df['winner_is_red']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# **[수정]** 전처리기에서 PCA 단계 제거
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# 트리 기반 모델은 스케일링이 필요 없으므로 수치형은 그대로 전달
preprocessor_tree = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 클래스 불균형 처리를 위한 가중치 계산
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# 모델 정의
models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'LightGBM': LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1),
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight),
    'CatBoost': CatBoostClassifier(
        random_state=42,
        verbose=False,
        class_weights=[1, scale_pos_weight]
    ),
}

# 하이퍼파라미터 그리드
param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'liblinear']
    },
    'Random Forest': {
        'classifier__n_estimators': [200, 400, 600, 800, 1000],
        'classifier__max_depth': [None, 10, 20, 30, 40],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['auto', 'sqrt', 'log2']
    },
    'XGBoost': {
        'classifier__n_estimators': [200, 400, 600, 800, 1000],
        'classifier__max_depth': [3, 6, 9],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__min_child_weight': [1, 5, 10],
        'classifier__reg_alpha': [0, 0.01, 0.1],
        'classifier__reg_lambda': [1, 5, 10]
    },
    'CatBoost': {
        'classifier__depth': [4, 6, 8],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__iterations': [300, 600],
        'classifier__l2_leaf_reg': [1, 3, 5]
    },
    'LightGBM': {
        'classifier__n_estimators': [200, 400, 600],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__num_leaves': [31, 63, 127],
        'classifier__max_depth': [-1, 10, 20],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    },
}

 # 모델 학습 및 평가 (Grid Search 적용)
for name, model in models.items():
    this_preprocessor = preprocessor_tree if name in ['Random Forest', 'XGBoost', 'LightGBM'] else preprocessor
    pipeline = Pipeline(steps=[('preprocessor', this_preprocessor),
                               ('classifier', model)])

    # 하이퍼파라미터 탐색 (XGBoost/Random Forest/LightGBM: RandomizedSearch, 나머지: GridSearch)
    if name in ['Random Forest', 'XGBoost', 'LightGBM', 'CatBoost']:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grids[name],
            n_iter=100,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
    else:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )

    print(f'=== {name} (Hyper‑parameter Search) ===')
    search.fit(X_train, y_train)
    print(f'Best params: {search.best_params_}')
    y_pred = search.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('=' * 60)
    
    
# 트리 기반 스태킹용 preprocessor
tree_preproc = preprocessor_tree

# 기본 모델(Estimators) 정의
estimators = [
    ('lr', Pipeline(steps=[('preprocessor', preprocessor), 
                          ('classifier', LogisticRegression(random_state=42, class_weight='balanced'))])),
    ('rf', Pipeline(steps=[('preprocessor', tree_preproc),
                          ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])),
    ('lgb', Pipeline(steps=[('preprocessor', tree_preproc),
                           ('classifier', LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1))])),
    ('xgb', Pipeline(steps=[('preprocessor', tree_preproc),
                           ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight))]))
    ,
    ('cb', Pipeline(steps=[('preprocessor', tree_preproc),
                          ('classifier', CatBoostClassifier(
                              random_state=42,
                              verbose=False,
                              class_weights=[1, scale_pos_weight]))])),
]

# 스태킹 모델 정의 (메타 모델: RandomForestClassifier)
stacking_model = StackingClassifier(
    estimators=estimators, 
    final_estimator=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'),
    cv=5 # 교차 검증 설정
)

# 스태킹 모델을 위한 파이프라인 생성
stacking_pipeline = Pipeline(steps=[('dummy', 'passthrough'),
                                    ('classifier', stacking_model)])

# 모델 학습
print("Stacking Model Training...")
stacking_pipeline.fit(X_train, y_train)
print("Training Complete.")

# 예측 및 평가 (승률 = 레드 승리 확률)
probs_stacking = stacking_pipeline.predict_proba(X_test)[:, 1]

print('--- Stacking Ensemble Model (Win Probability) ---')
print(f'Log Loss      : {log_loss(y_test, probs_stacking):.4f}')
print(f'Brier Score   : {brier_score_loss(y_test, probs_stacking):.4f}')
print(f'ROC‑AUC       : {roc_auc_score(y_test, probs_stacking):.4f}')

# 선택적으로 0.5 기준 분류 결과도 출력
y_pred_stacking = (probs_stacking >= 0.5).astype(int)
print('\n--- Classification Metrics (threshold=0.5) ---')
print(classification_report(y_test, y_pred_stacking))