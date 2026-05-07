import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# PAGE TITLE
# -------------------------------

st.title("Vehicle Classification ML Project")

st.write("Machine Learning based Vehicle Type Prediction System")

# -------------------------------
# LOAD DATASET
# -------------------------------

df_cars = pd.read_csv('Cars Datasets 2025.csv', encoding='latin1')

# -------------------------------
# CLEAN DATA
# -------------------------------

def extract_cc(x):

    if pd.isna(x):
        return np.nan

    s = str(x).lower().replace(',', '').replace('cc', '').strip()

    try:
        return float(s)

    except:
        return np.nan

df_cars['Engine Capacity'] = df_cars['CC/Battery Capacity'].apply(extract_cc)

df_cars = df_cars.dropna(subset=['Engine Capacity'])

# -------------------------------
# CREATE DATASETS
# -------------------------------

np.random.seed(42)

four_wheelers = pd.DataFrame({
    'Engine Capacity': df_cars['Engine Capacity'].sample(100, replace=True).values,
    'Number of Wheels': 4,
    'Weight (kg)': np.random.uniform(1000, 2500, 100),
    'Vehicle Type': 'four-wheeler'
})

two_wheelers = pd.DataFrame({
    'Engine Capacity': np.random.uniform(50, 400, 100),
    'Number of Wheels': 2,
    'Weight (kg)': np.random.uniform(100, 250, 100),
    'Vehicle Type': 'two-wheeler'
})

three_wheelers = pd.DataFrame({
    'Engine Capacity': np.random.uniform(150, 600, 100),
    'Number of Wheels': 3,
    'Weight (kg)': np.random.uniform(300, 600, 100),
    'Vehicle Type': 'three-wheeler'
})

dataset = pd.concat(
    [two_wheelers, three_wheelers, four_wheelers],
    ignore_index=True
)

# -------------------------------
# PREPROCESSING
# -------------------------------

X = dataset[['Engine Capacity', 'Number of Wheels', 'Weight (kg)']]

y = dataset['Vehicle Type']

le = LabelEncoder()

y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# -------------------------------
# TRAIN MODEL
# -------------------------------

svm_model = SVC(kernel='linear', C=1.0, random_state=42)

svm_model.fit(X_train_scaled, y_train)

st.success("Model Training Complete")

# -------------------------------
# EVALUATE MODEL
# -------------------------------

y_pred = svm_model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")

st.write(f"Accuracy: {acc * 100:.2f}%")

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# -------------------------------
# PLOT 1 - PAIRPLOT
# -------------------------------

st.subheader("Pairwise Feature Relationships")

dataset['Target'] = y_encoded

pairplot = sns.pairplot(
    dataset,
    vars=['Engine Capacity', 'Weight (kg)', 'Number of Wheels'],
    hue='Vehicle Type',
    palette=['red', 'green', 'blue'],
    markers=["o", "s", "D"]
)

st.pyplot(pairplot.fig)

# -------------------------------
# PLOT 2 - 3D SCATTER
# -------------------------------

st.subheader("3D Feature Distribution")

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b']

for i, t in enumerate(le.classes_):

    idx = (dataset['Vehicle Type'] == t)

    ax.scatter(
        dataset.loc[idx, 'Engine Capacity'],
        dataset.loc[idx, 'Weight (kg)'],
        dataset.loc[idx, 'Number of Wheels'],
        c=colors[i],
        label=t,
        s=50,
        alpha=0.7
    )

ax.set_xlabel('Engine Capacity (cc)')
ax.set_ylabel('Weight (kg)')
ax.set_zlabel('Number of Wheels')

ax.set_title('3D Feature Distribution')

ax.legend(prop={'size': 10})

st.pyplot(fig)

# -------------------------------
# CONFUSION MATRIX
# -------------------------------

st.subheader("Confusion Matrix")

cnf_matrix = confusion_matrix(y_test, y_pred)

fig2, ax2 = plt.subplots(figsize=(8, 6))

im = ax2.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

plt.colorbar(im)

tick_marks = np.arange(len(le.classes_))

ax2.set_xticks(tick_marks)
ax2.set_yticks(tick_marks)

ax2.set_xticklabels(le.classes_, rotation=45)
ax2.set_yticklabels(le.classes_)

thresh = cnf_matrix.max() / 2.

for i, j in itertools.product(
    range(cnf_matrix.shape[0]),
    range(cnf_matrix.shape[1])
):

    ax2.text(
        j,
        i,
        format(cnf_matrix[i, j], 'd'),
        horizontalalignment="center",
        color="white" if cnf_matrix[i, j] > thresh else "black"
    )

ax2.set_ylabel('True Label')
ax2.set_xlabel('Predicted Label')

st.pyplot(fig2)

# -------------------------------
# USER PREDICTION SECTION
# -------------------------------

st.subheader("Predict Vehicle Type")

engine = st.number_input(
    "Engine Capacity",
    min_value=0.0
)

wheels = st.number_input(
    "Number of Wheels",
    min_value=2
)

weight = st.number_input(
    "Weight (kg)",
    min_value=0.0
)

if st.button("Predict"):

    input_data = scaler.transform(
        [[engine, wheels, weight]]
    )

    prediction = svm_model.predict(input_data)

    predicted_label = le.inverse_transform(prediction)

    st.success(
        f"Predicted Vehicle Type: {predicted_label[0]}"
    )
