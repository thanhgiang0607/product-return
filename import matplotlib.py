import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import shap
import os

# Đọc dữ liệu
orders_df = pd.read_csv(r'C:\Users\Windows\Downloads\synthetic_orders.csv')
survey_df = pd.read_csv(r'C:\Users\Windows\Downloads\synthetic_survey.csv')

# Merge orders and survey data
df = pd.merge(orders_df, survey_df, on='user_id', how='left')

# Encode categorical variables
categorical_cols = ['gender', 'payment_method', 'product_category', 'return_reason', 'location']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Encode buy_frequency
df['buy_frequency'] = LabelEncoder().fit_transform(df['buy_frequency'])

# Feature selection
features = ['gender', 'age', 'payment_method', 'product_category',
            'delivery_duration', 'buy_frequency', 'satisfaction']

X = df[features]
y = df['is_returned']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predict
y_pred = xgb_model.predict(X_test)

#classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues',
            xticklabels=["Not Returned", "Returned"],
            yticklabels=["Not Returned", "Returned"])
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(r'C:\Users\Windows\Downloads\confusion_matrix_xgboost.png')
plt.show()


# SHAP values
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(r'C:\Users\Windows\Downloads\shap_summary_plot.png')
plt.show()

# Return rate by location
location_performance = df.groupby('location')['is_returned'].mean().reset_index()
location_performance.columns = ['location', 'return_rate']

plt.figure(figsize=(6, 4))
sns.barplot(data=location_performance, x='location', y='return_rate')
plt.title("Return Rate by Location")
plt.ylabel("Return Rate")
plt.xlabel("Location")
plt.tight_layout()
plt.savefig(r'C:\Users\Windows\Downloads\location_return_rate.png')
plt.show()

#classimetrics report
report = classification_report(y_test, y_pred, output_dict=True)
classes = [str(label) for label in sorted(set(y_test))]
metrics = ['precision', 'recall', 'f1-score']

# Lấy data cho từng lớp
data = {m: [report[c][m] for c in classes] for m in metrics}
# Thêm macro avg & weighted avg
for avg_type in ['macro avg', 'weighted avg']:
    for m in metrics:
        data[m].append(report[avg_type][m])
# Create a DataFrame for the classification report
index_labels = classes + ['macro avg', 'weighted avg']
report_df = pd.DataFrame(data, index=index_labels)
# Vẽ heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
plt.title("Classification Report Heatmap (Including Averages)")
plt.xlabel("Metric")
plt.ylabel("Class / Average")
plt.tight_layout()
plt.savefig(r"C:\Users\Windows\Downloads\classification_report_heatmap_full.png")
plt.show()