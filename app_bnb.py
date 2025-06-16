
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dashboard Penyakit",
    page_icon="🩺",
)

st.title("Aplikasi Penyakit: Klasifikasi & Clustering")

@st.cache_resource
def load_resources():
    try:
        model = joblib.load('diseaseprediction.joblib')
        disease_encoder = joblib.load('disease_encoder.joblib')
        mlb = joblib.load('mlb.joblib')
        df = pd.read_csv('dataset.csv')
        try:
            symptom_severity = pd.read_csv('Symptom-severity.csv')
            symptom_severity.columns = symptom_severity.columns.str.strip().str.lower()
        except Exception:
            symptom_severity = None
        try:
            desc_df = pd.read_csv('symptom_Description.csv')
            desc_df.columns = desc_df.columns.str.strip().str.lower()
        except Exception:
            desc_df = None
        try:
            precaution_df = pd.read_csv('symptom_precaution.csv')
            precaution_df.columns = precaution_df.columns.str.strip().str.lower()
        except Exception:
            precaution_df = None
        return model, disease_encoder, mlb, symptom_severity, desc_df, precaution_df, df
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

model, disease_encoder, mlb, symptom_severity, desc_df, precaution_df, df = load_resources()

tab1, tab2 = st.tabs(["Klasifikasi Penyakit", "Clustering Gejala"])

with tab1:
    split_option = st.selectbox(
        "Pilih rasio data training : testing",
        options=["60:40", "70:30", "80:20"],
        index=1
    )
    split_dict = {"60:40": 0.4, "70:30": 0.3, "80:20": 0.2}
    test_size = split_dict[split_option]

    kolom = ['Symptom_1','Symptom_2','Symptom_3','Symptom_4','Symptom_5','Symptom_6','Symptom_7','Symptom_8','Symptom_9','Symptom_10','Symptom_11']
    for i in kolom:
        df[i] = df[i].fillna('Unknown').astype(str).str.strip().str.lower().str.replace(' ', '_')
    symptom_lists_for_mlb = []
    for index, row in df.iterrows():
        row_symptoms = [s for s in row[kolom] if s != 'unknown']
        symptom_lists_for_mlb.append(row_symptoms)
    symptom_binary_features = mlb.transform(symptom_lists_for_mlb)
    symptom_binary_df = pd.DataFrame(symptom_binary_features, columns=mlb.classes_)
    if symptom_severity is not None:
        weights = symptom_severity.set_index('symptom').to_dict()['weight']
        df['severity_score'] = df[kolom].apply(lambda row: sum(weights.get(s, 0) for s in row if s in weights), axis=1)
        q2 = df['severity_score'].quantile(0.66)
        df['is_high_severity'] = df['severity_score'].apply(lambda x: 1 if x > q2 else 0)
    else:
        df['is_high_severity'] = 0
    df['is_cluster_1'] = 0
    y = disease_encoder.transform(df['Disease'])
    X = pd.concat([df[['is_high_severity', 'is_cluster_1']], symptom_binary_df], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    n_classes = len(disease_encoder.classes_)
    st.subheader("🧪 Evaluasi Model Penyakit")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classes", n_classes)
    with col2:
        st.metric("Test Samples", len(y_test))
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        st.metric("Accuracy", f"{accuracy:.2f}")
    with col4:
        st.metric("Precision", f"{precision:.2f}")
    with col5:
        st.metric("Recall", f"{recall:.2f}")
    with col6:
        st.metric("F1-score", f"{f1:.2f}")

    from sklearn.model_selection import cross_val_score

    # Lakukan cross-validation hanya saat model sudah dilatih dan X_train, y_train tersedia
    with st.expander("📊 Cross-Validation (10-fold)", expanded=False):
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
            mean_cv = cv_scores.mean()
            std_cv = cv_scores.std()
            st.write(f"*Akurasi Cross-Validation (10-fold):* {cv_scores}")
            st.write(f"*Rata-rata Akurasi:* {mean_cv:.4f}")
            st.write(f"*Standar Deviasi Akurasi:* {std_cv:.4f}")
            st.write(f"*Akurasi pada Test Set:* {accuracy:.4f}")
            if abs(mean_cv - accuracy) < 0.05:
                st.success("✅ Model menunjukkan konsistensi antara cross-validation dan test set.")
            else:
                st.warning("⚠ Performa cross-validation dan test set berbeda signifikan. Perlu investigasi lebih lanjut.")
        except Exception as e:
            st.error(f"Gagal menghitung cross-validation: {e}")


    st.markdown("---")
    st.header("🔍 Klasifikasi Penyakit")
    available_symptoms_for_multiselect = list(mlb.classes_)
    with st.form("prediction_form"):
        selected_symptoms_raw = st.multiselect(
            "Pilih gejala yang dialami:",
            options=available_symptoms_for_multiselect,
            help="Pilih satu atau lebih gejala dari daftar"
        )
        predict_btn = st.form_submit_button("🧠 Klasifikasi Penyakit", use_container_width=True)
    input_symptoms = [selected_symptoms_raw]
    input_binary = mlb.transform(input_symptoms)
    is_high_severity = 0
    severity_score = 0
    if symptom_severity is not None and selected_symptoms_raw:
        weights = symptom_severity.set_index('symptom').to_dict()['weight']
        severity_score = sum(weights.get(symptom, 0) for symptom in selected_symptoms_raw)
        q2 = symptom_severity['weight'].quantile(0.66)
        is_high_severity = int(severity_score > q2)
    is_cluster_1 = 0
    input_df = pd.DataFrame(input_binary, columns=mlb.classes_)
    input_df['is_high_severity'] = is_high_severity
    input_df['is_cluster_1'] = is_cluster_1
    model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else input_df.columns
    input_df = input_df.reindex(columns=model_features, fill_value=0)
    if predict_btn:
        try:
            probabilities = model.predict_proba(input_df)
            n_classes = len(disease_encoder.classes_)
            if probabilities is not None and probabilities.ndim == 2 and probabilities.shape[1] == n_classes:
                predicted_idx = np.argmax(probabilities)
                disease_name = disease_encoder.inverse_transform([predicted_idx])[0]
                st.success("📋 Hasil Klasifikasi")
                st.markdown(f"🦠 *Penyakit:* <span style='font-size: 1.2em; color: #28a745;'><b>{str(disease_name).strip()}</b></span>", unsafe_allow_html=True)
                st.markdown("---")
                if desc_df is not None:
                    desc_row = desc_df[desc_df['disease'].str.strip().str.lower() == str(disease_name).strip().lower()]
                    if not desc_row.empty:
                        st.info(f"📖 Deskripsi: {desc_row['description'].values[0]}", icon="ℹ")
                    else:
                        st.caption(f"Deskripsi tidak tersedia untuk '{str(disease_name).strip()}'.")
                if precaution_df is not None:
                    precaution_row = precaution_df[precaution_df['disease'].str.strip().str.lower() == str(disease_name).strip().lower()]
                    if not precaution_row.empty:
                        precautions = [str(p).strip() for p in precaution_row.iloc[0, 1:].values if pd.notna(p) and str(p).strip().lower() not in ['nan', '']]
                        if precautions:
                            st.warning("🛡 Tindakan Pencegahan & Saran:")
                            for i, p_text in enumerate(precautions, 1):
                                st.write(f"{i}. {p_text}")
                        else:
                            st.caption(f"Tindakan pencegahan tidak spesifik atau tidak tersedia untuk '{str(disease_name).strip()}'.")
                    else:
                        st.caption(f"Informasi tindakan pencegahan tidak ditemukan untuk '{str(disease_name).strip()}'.")
                st.markdown("---")
                if symptom_severity is not None and selected_symptoms_raw:
                    weights = symptom_severity.set_index('symptom').to_dict()['weight']
                    current_user_severity_score = sum(weights.get(symptom, 0) for symptom in selected_symptoms_raw)
                    severity_q1_threshold = df['severity_score'].quantile(0.33) if 'severity_score' in df else 1
                    severity_q2_threshold = df['severity_score'].quantile(0.66) if 'severity_score' in df else 2
                    if current_user_severity_score <= severity_q1_threshold:
                        severity_category = "Rendah"
                    elif current_user_severity_score <= severity_q2_threshold:
                        severity_category = "Sedang"
                    else:
                        severity_category = "Tinggi"
                    st.info(f"📊 Total Skor Keparahan Gejala (berdasarkan input): {current_user_severity_score} (Kategori: {severity_category}).")
                else:
                    st.info("ℹ Skor keparahan gejala tidak dapat dihitung (data 'Symptom-severity.csv' tidak tersedia atau kosong), sehingga kategori tidak dapat ditentukan.")
            else:
                st.error("Tidak dapat menghitung probabilitas prediksi. Model mungkin tidak mengembalikan output yang diharapkan atau jumlah kelas tidak sesuai.")
        except Exception as e:
            st.error(f"Terjadi error saat Klasifikasi: {e}")
            st.text(traceback.format_exc())
    st.markdown("---")
    st.caption("⚕ Aplikasi ini untuk tujuan informasi dan edukasi saja. Selalu konsultasikan dengan profesional medis atau dokter untuk diagnosis dan perawatan yang akurat.", unsafe_allow_html=True)

with tab2:
    st.header("🔬 Clustering Gejala (KMeans)")
    kolom = ['Symptom_1','Symptom_2','Symptom_3','Symptom_4','Symptom_5','Symptom_6','Symptom_7','Symptom_8','Symptom_9','Symptom_10','Symptom_11']
    for i in kolom:
        df[i] = df[i].fillna('Unknown').astype(str).str.strip().str.lower().str.replace(' ', '_')
    symptom_lists_for_mlb = []
    for index, row in df.iterrows():
        row_symptoms = [s for s in row[kolom] if s != 'unknown']
        symptom_lists_for_mlb.append(row_symptoms)
    symptom_binary_features = mlb.transform(symptom_lists_for_mlb)
    symptom_binary_df = pd.DataFrame(symptom_binary_features, columns=mlb.classes_)
    if symptom_severity is not None:
        weights = symptom_severity.set_index('symptom').to_dict()['weight']
        df['severity_score'] = df[kolom].apply(lambda row: sum(weights.get(s, 0) for s in row if s in weights), axis=1)
    else:
        df['severity_score'] = 0
    X_cluster_features = symptom_binary_df.copy()
    X_cluster_features['severity_score'] = df['severity_score']
    k = st.slider("Pilih jumlah cluster (k)", min_value=2, max_value=6, value=2)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_features)
    df['cluster'] = clusters
    st.write("Distribusi jumlah data per cluster:")
    st.write(df['cluster'].value_counts())
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster_features)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Visualisasi Cluster Penyakit (PCA)")
    st.pyplot(fig)

    from sklearn.metrics import silhouette_score, davies_bouldin_score

    if len(X_cluster_features) > 1 and len(np.unique(clusters)) > 1:
        silhouette_avg = silhouette_score(X_cluster_features, clusters)
        dbi = davies_bouldin_score(X_cluster_features, clusters)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Silhouette Score", f"{silhouette_avg:.4f}")
        with col2:
            st.metric("Davies-Bouldin Index", f"{dbi:.4f}")
    else:
        st.warning("Tidak cukup data atau cluster untuk menghitung metrik clustering.")
