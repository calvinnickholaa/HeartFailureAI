import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import warnings
import pickle

warnings.filterwarnings("ignore")

# Baca dataset
path_dataset = "heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(path_dataset)
pd.set_option('display.max_columns', 50)

numCol = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'serum_sodium', 'platelets', 'serum_creatinine', 'time']

def outlierDEL(column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    IQR = q3 - q1
    lowBound = q1 - 1.5 * IQR
    upBound = q3 + 1.5 * IQR
    df[column] = df[column].apply(lambda x: lowBound if x < lowBound else (upBound if x > upBound else x))

outliersCOL = ['creatinine_phosphokinase', 'ejection_fraction', 'serum_sodium', 'platelets', 'serum_creatinine']
for col in outliersCOL:
    outlierDEL(col)

dfClean = df.copy()

dfClean.head()

# Menggunakan dfClean untuk membagi fitur dan target
X = dfClean.drop('DEATH_EVENT', axis=1)
y = dfClean['DEATH_EVENT']

# Oversampling menggunakan SMOTE
oversampler = SMOTE(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Standarisasi fitur pada data oversampling
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.3, random_state=42)

# Inisialisasi estimator untuk Random Forest
rf_estimator = RandomForestClassifier(random_state=42)

# Inisialisasi list untuk menyimpan akurasi, fitur yang dieliminasi, ranking, dan support
accuracies = []
eliminated_features = []
feature_rankings = []
feature_supports = []

# Percobaan untuk n_features_to_select dari 1 hingga 12
for n_features in range(1, 13):
    # Inisialisasi RFE dengan estimator RandomForestClassifier
    rfe = RFE(estimator=rf_estimator, n_features_to_select=n_features)
    # Fit RFE pada data latih
    rfe.fit(X_train, y_train)
    # Simpan fitur yang dieliminasi, ranking, dan support
    eliminated_features.append((n_features, X.columns[~rfe.support_].tolist()))
    feature_rankings.append((n_features, rfe.ranking_.tolist()))
    feature_supports.append((n_features, rfe.support_.tolist()))
    # Ambil fitur yang dipilih
    X_train_selected = rfe.transform(X_train)
    X_test_selected = rfe.transform(X_test)
    # Latih model SVM pada data latih yang dipilih
    svm_estimator = SVC(random_state=42)
    svm_estimator.fit(X_train_selected, y_train)
    # Prediksi pada data uji yang dipilih
    y_pred = svm_estimator.predict(X_test_selected)
    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append((n_features, accuracy))

# Tampilkan hasil akurasi, fitur yang dieliminasi, ranking, dan support untuk setiap percobaan
for (n_features, eliminated), (n_features, accuracy), (n_features, ranking), (n_features, support) in zip(eliminated_features, accuracies, feature_rankings, feature_supports):
    print(f"Jumlah fitur: {n_features}, Akurasi: {accuracy}")
    print("Fitur yang dieliminasi:", eliminated)
    print("Ranking fitur:")
    for feature, rank in zip(X.columns, ranking):
        print(f"{feature}: {rank}")
    print("Support fitur:", support)
    print("\n")

# Melakukan RFE dengan 8 fitur terpilih
rfe = RFE(estimator=rf_estimator, n_features_to_select=8)
rfe.fit(X_train, y_train)
X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)

# Melatih model SVM menggunakan data yang dipilih oleh RFE
svm_model_trained = SVC(random_state=42)
svm_model_trained.fit(X_train_selected, y_train)

# Gunakan model SVM yang dilatih dalam GridSearchCV atau evaluasi model
param_grid = {
    'C': [10, 100, 1000],
    'kernel': ['rbf'],
    'gamma': [0.01, 0.1, 1],
}
cv = StratifiedKFold(n_splits=100, shuffle=True, random_state=42)
grid_search = GridSearchCV(svm_model_trained, param_grid, cv=cv, verbose=2, n_jobs=-1)

# Fit GridSearchCV pada data
grid_search.fit(X_train_selected, y_train)

best_params = grid_search.best_params_
print("Parameter terbaik:", best_params)

# Skor data test
test_score = grid_search.score(X_test_selected, y_test)
print("Test Acc:", test_score)

# Prediksi label dari data uji
y_pred = grid_search.predict(X_test_selected)

# Classification report
report = classification_report(y_test, y_pred)
print(report)

# Confusion Matrix
cfm = confusion_matrix(y_test, y_pred)
print(cfm)

# Simpan semua model, scaler, label encoder, dan objek lainnya ke dalam satu file .pkl
models_and_objects = {
    'svm_model_trained': svm_model_trained,
    'grid_search': grid_search,
    'scaler': scaler,
    'rfe': rfe,
    'eliminated_features': eliminated_features,
    'feature_rankings': feature_rankings,
    'feature_supports': feature_supports,
    'X_train_selected': X_train_selected,
    'X_test_selected': X_test_selected,
    'y_train': y_train,
    'y_test': y_test,
    'y_pred': y_pred,
    'best_params': best_params,
}

with open('heartFailure.pkl', 'wb') as file:
    pickle.dump(models_and_objects, file)
