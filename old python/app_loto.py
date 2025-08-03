
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analyse Loto", layout="wide")
st.title("🎲 Analyse des Tirages Loto (depuis 2008)")

uploaded_file = st.file_uploader("📂 Téléversez le fichier Excel Loto", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="loto_2008")
    df = df[df['boule_1'].notna()]

    if not pd.api.types.is_datetime64_any_dtype(df['date_de_tirage']):
        df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], errors='coerce')

    df['année'] = df['date_de_tirage'].dt.year
    df['jour'] = df['date_de_tirage'].dt.day_name()

    st.subheader("🎛️ Filtres")
    col1, col2 = st.columns(2)
    with col1:
        années = st.multiselect("Années :", sorted(df['année'].dropna().unique()), default=sorted(df['année'].dropna().unique()))
    with col2:
        jours = st.multiselect("Jour du tirage :", sorted(df['jour'].dropna().unique()), default=sorted(df['jour'].dropna().unique()))

    df_filtré = df[df['année'].isin(années) & df['jour'].isin(jours)]
    st.markdown(f"**{len(df_filtré)} tirages sélectionnés**")

    st.subheader("📊 Fréquence des Boules (filtrée)")
    boules = df_filtré[['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']].values.flatten()
    boule_counts = pd.Series(boules).value_counts().sort_index()
    total_tirages = len(df_filtré)

    # Date de dernière sortie par boule
    derniere_sortie = {}
    for num in range(1, 50):
        condition = df_filtré[['boule_1','boule_2','boule_3','boule_4','boule_5']].isin([num]).any(axis=1)
        dates = df_filtré.loc[condition, 'date_de_tirage']
        derniere_sortie[num] = dates.max() if not dates.empty else pd.NaT

    freq_df = pd.DataFrame({
        "Numéro": boule_counts.index,
        "Sorties": boule_counts.values,
        "% Tirages": (boule_counts.values / total_tirages * 100).round(2),
        "Dernière sortie": [derniere_sortie[n] for n in boule_counts.index]
    }).sort_values(by="Sorties", ascending=False)

    st.subheader("📈 Top Boules les plus sorties")
    st.dataframe(freq_df.head(10).reset_index(drop=True))

    st.subheader("📉 Boules les moins sorties")
    st.dataframe(freq_df.sort_values(by="Sorties").head(10).reset_index(drop=True))

    st.subheader("🔥 Recommandations de Boules")
    col1, col2, col3 = st.columns(3)

    top = freq_df.head(5)["Numéro"].tolist()
    bottom = freq_df.sort_values(by="Sorties").head(5)["Numéro"].tolist()
    mix = top[:3] + bottom[:2]

    with col1:
        st.markdown("**+ Fréquence :**")
        st.success(f"Boules : {sorted(top)}")

    with col2:
        st.markdown("**- Fréquence :**")
        st.warning(f"Boules : {sorted(bottom)}")

    with col3:
        st.markdown("**Mix des deux :**")
        st.info(f"Boules : {sorted(mix)}")

    st.markdown("---")
    st.subheader("🎰 Générateur de Combinaisons")
    gen_type = st.selectbox("Méthode de génération :", [
        "Aléatoire",
        "Pondérée (+ sorties)",
        "Pondérée (- sorties)",
        "Mix (3 top + 2 low)"
    ])

    def generate_combination():
        if gen_type == "Aléatoire":
            boules_choisies = np.random.choice(np.arange(1, 50), size=5, replace=False)
        elif gen_type == "Pondérée (+ sorties)":
            probs = boule_counts / boule_counts.sum()
            boules_choisies = np.random.choice(boule_counts.index, size=5, replace=False, p=probs.values)
        elif gen_type == "Pondérée (- sorties)":
            inv = 1 / boule_counts.replace(0, 1)
            probs = inv / inv.sum()
            boules_choisies = np.random.choice(boule_counts.index, size=5, replace=False, p=probs.values)
        elif gen_type == "Mix (3 top + 2 low)":
            boules_choisies = np.random.choice(top, 3, replace=False).tolist() +                               np.random.choice(bottom, 2, replace=False).tolist()
        else:
            boules_choisies = np.random.choice(np.arange(1, 50), size=5, replace=False)

        numero_chance = np.random.randint(1, 11)
        return sorted(boules_choisies), numero_chance

    if st.button("Générer une combinaison"):
        boules, chance = generate_combination()
        st.success(f"Boules : {boules}  |  Numéro Chance : {chance}")
