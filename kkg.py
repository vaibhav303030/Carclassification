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

st.write("Machine Learning based Car Category Prediction System")

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
# (New features: Engine Capacity, Horsepower, 0-100 Performance, Fuel Type encoded)
# Target: Car Category (Economy, SUV, Sports, Electric, Luxury)
# -------------------------------

np.random.seed(42)

# Fuel type encoding: Petrol=0, Diesel=1, Electric=2, Hybrid=3

economy_cars = pd.DataFrame({
    'Engine Capacity': np.random.uniform(800, 1400, 100),
    'Horsepower': np.random.uniform(60, 110, 100),
    'Performance (0-100s)': np.random.uniform(10, 16, 100),
    'Fuel Type': np.random.choice([0, 1], 100),          # Petrol or Diesel
    'Car Category': 'Economy'
})

suv_cars = pd.DataFrame({
    'Engine Capacity': np.random.uniform(1500, 3000, 100),
    'Horsepower': np.random.uniform(130, 250, 100),
    'Performance (0-100s)': np.random.uniform(7, 12, 100),
    'Fuel Type': np.random.choice([0, 1, 3], 100),       # Petrol, Diesel, Hybrid
    'Car Category': 'SUV'
})

sports_cars = pd.DataFrame({
    'Engine Capacity': df_cars['Engine Capacity'].sample(100, replace=True).values,
    'Horsepower': np.random.uniform(300, 700, 100),
    'Performance (0-100s)': np.random.uniform(2.5, 6, 100),
    'Fuel Type': np.random.choice([0], 100),              # Petrol only
    'Car Category': 'Sports'
})

electric_cars = pd.DataFrame({
    'Engine Capacity': np.random.uniform(40000, 100000, 100),  # Battery in Wh equivalent
    'Horsepower': np.random.uniform(150, 500, 100),
    'Performance (0-100s)': np.random.uniform(3, 8, 100),
    'Fuel Type': np.full(100, 2),                         # Electric only
    'Car Category': 'Electric'
})

luxury_cars = pd.DataFrame({
    'Engine Capacity': np.random.uniform(2000, 6000, 100),
    'Horsepower': np.random.uniform(250, 600, 100),
    'Performance (0-100s)': np.random.uniform(4, 8, 100),
    'Fuel Type': np.random.choice([0, 3], 100),           # Petrol or Hybrid
    'Car Category': 'Luxury'
})

dataset = pd.concat(
    [economy_cars, suv_cars, sports_cars, electric_cars, luxury_cars],
    ignore_index=True
)

# -------------------------------
# PREPROCESSING
# -------------------------------

X = dataset[['Engine Capacity', 'Horsepower', 'Performance (0-100s)', 'Fuel Type']]

y = dataset['Car Category']

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
    vars=['Engine Capacity', 'Horsepower', 'Performance (0-100s)'],
    hue='Car Category',
    palette=['red', 'green', 'blue', 'orange', 'purple'],
    markers=["o", "s", "D", "^", "v"]
)

st.pyplot(pairplot.fig)

# -------------------------------
# PLOT 2 - 3D SCATTER
# -------------------------------

st.subheader("3D Feature Distribution")

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'orange', 'purple']

for i, t in enumerate(le.classes_):

    idx = (dataset['Car Category'] == t)

    ax.scatter(
        dataset.loc[idx, 'Engine Capacity'],
        dataset.loc[idx, 'Horsepower'],
        dataset.loc[idx, 'Performance (0-100s)'],
        c=colors[i],
        label=t,
        s=50,
        alpha=0.7
    )

ax.set_xlabel('Engine Capacity (cc / Wh)')
ax.set_ylabel('Horsepower (hp)')
ax.set_zlabel('0-100 km/h Time (s)')

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

st.subheader("Predict Car Category")

engine = st.number_input(
    "Engine Capacity (cc for petrol/diesel, Wh for electric)",
    min_value=0.0
)

horsepower = st.number_input(
    "Horsepower (hp)",
    min_value=0.0
)

performance = st.number_input(
    "0-100 km/h Time (seconds)",
    min_value=0.0
)

fuel_type = st.selectbox(
    "Fuel Type",
    options=["Petrol", "Diesel", "Electric", "Hybrid"]
)

fuel_map = {"Petrol": 0, "Diesel": 1, "Electric": 2, "Hybrid": 3}

# -------------------------------
# CAR EXAMPLES PER CATEGORY (from dataset)
# -------------------------------

category_keywords = {
    'Economy': ['alto', 'i10', 'santro', 'nano', 'eon', 'kwid', 'celerio', 'wagon', 'swift', 'beat', 'spark', 'brio', 'gio', 'micra', 'pulse'],
    'SUV':     ['creta', 'scorpio', 'fortuner', 'brezza', 'ecosport', 'duster', 'xuv', 'compass', 'harrier', 'seltos', 'hector', 'thar', 'bolero', 'safari', 'tucson', 'suv'],
    'Sports':  ['mustang', 'camaro', 'bmw m', 'ferrari', 'porsche', 'lamborghini', 'audi r', 'corvette', 'gt', 'amg', 'sport'],
    'Electric': ['ev', 'electric', 'nexon ev', 'zs ev', 'e-tron', 'model', 'leaf', 'bolt', 'ioniq', 'atto'],
    'Luxury':  ['mercedes', 'bmw', 'audi', 'jaguar', 'volvo', 'lexus', 'land rover', 'bentley', 'rolls', 'maserati', 'genesis', 'lincoln']
}

def get_examples_from_dataset(df, category, n=5):
    keywords = category_keywords.get(category, [])
    # Try to find car name column
    name_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ['name', 'model', 'car', 'brand', 'make']):
            name_col = col
            break
    if name_col is None:
        return []
    matches = df[df[name_col].astype(str).str.lower().str.contains('|'.join(keywords), na=False)]
    if len(matches) > 0:
        return matches[name_col].dropna().unique()[:n].tolist()
    return []

if st.button("Predict"):

    input_data = scaler.transform(
        [[engine, horsepower, performance, fuel_map[fuel_type]]]
    )

    prediction = svm_model.predict(input_data)

    predicted_label = le.inverse_transform(prediction)

    category = predicted_label[0]

    st.success(
        f"Predicted Car Category: {category}"
    )

    # Show example cars from dataset
    st.subheader(f"Example {category} Cars from Dataset")

    examples = get_examples_from_dataset(df_cars, category, n=5)

    if examples:
        for car in examples:
            st.write(f"🚗 {car}")
    else:
        # Fallback hardcoded examples if dataset has no matching names
        fallback = {
            'Economy':  ['Maruti Alto', 'Hyundai i10', 'Tata Nano', 'Renault Kwid', 'Chevrolet Beat'],
            'SUV':      ['Hyundai Creta', 'Mahindra Scorpio', 'Toyota Fortuner', 'Kia Seltos', 'MG Hector'],
            'Sports':   ['Ford Mustang', 'BMW M3', 'Porsche 911', 'Audi TT', 'Chevrolet Camaro'],
            'Electric': ['Tata Nexon EV', 'MG ZS EV', 'Tesla Model 3', 'Hyundai Ioniq', 'Nissan Leaf'],
            'Luxury':   ['Mercedes E-Class', 'BMW 7 Series', 'Audi A8', 'Jaguar XF', 'Volvo S90']
        }
        for car in fallback.get(category, []):
            st.write(f"🚗 {car}")
