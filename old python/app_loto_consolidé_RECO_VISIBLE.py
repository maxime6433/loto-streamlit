
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter

st.set_page_config(page_title="Analyse Loto Complète", layout="wide")
st.title("🎯 Analyse Complète des Tirages Loto (depuis 2008)")

uploaded_file = st.file_uploader("📂 Téléversez le fichier Excel Loto", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="loto_2008")
    df = df[df['boule_1'].notna()]
    df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], errors='coerce')
    df['année'] = df['date_de_tirage'].dt.year
    df['jour'] = df['date_de_tirage'].dt.day_name()

    st.subheader("🎛️ Filtres")
    col1, col2 = st.columns(2)
    with col1:
        années = st.multiselect("Années :", sorted(df['année'].unique()), default=sorted(df['année'].unique()))
    with col2:
        jours = st.multiselect("Jour du tirage :", sorted(df['jour'].unique()), default=sorted(df['jour'].unique()))
    df_filtré = df[df['année'].isin(années) & df['jour'].isin(jours)]
    st.markdown(f"**{len(df_filtré)} tirages sélectionnés**")

    boules = df_filtré[['boule_1','boule_2','boule_3','boule_4','boule_5']].values.flatten()
    boule_counts = pd.Series(boules).value_counts().sort_index()
    numero_chance_counts = df_filtré['numero_chance'].value_counts().sort_index()
    total_tirages = len(df_filtré)

    derniere_sortie = {}
    for num in range(1, 50):
        filt = df_filtré[['boule_1','boule_2','boule_3','boule_4','boule_5']].isin([num]).any(axis=1)
        dates = df_filtré.loc[filt, 'date_de_tirage']
        derniere_sortie[num] = dates.max() if not dates.empty else pd.NaT

    derniere_chance = {}
    for n in range(1, 11):
        filt = df_filtré[df_filtré['numero_chance'] == n]
        derniere_chance[n] = filt['date_de_tirage'].max() if not filt.empty else pd.NaT

    boule_df = pd.DataFrame({
        "Numéro": list(range(1, 50)),
        "Sorties": [boule_counts.get(i, 0) for i in range(1, 50)],
        "% Tirages": [round((boule_counts.get(i, 0) / total_tirages * 100), 1) for i in range(1, 50)],
        "Dernière sortie": [derniere_sortie[i] for i in range(1, 50)]
    })
    boule_df["Absence (jours)"] = (df_filtré['date_de_tirage'].max() - boule_df["Dernière sortie"]).dt.days

    chance_df = pd.DataFrame({
        "Numéro Chance": list(range(1, 11)),
        "Sorties": [numero_chance_counts.get(i, 0) for i in range(1, 11)],
        "% Tirages": [round((numero_chance_counts.get(i, 0) / total_tirages * 100), 1) for i in range(1, 11)],
        "Dernière sortie": [derniere_chance[i] for i in range(1, 11)]
    })
    chance_df["Absence (jours)"] = (df_filtré['date_de_tirage'].max() - chance_df["Dernière sortie"]).dt.days

    st.subheader("📈 Top 30 Boules les plus sorties")
    st.dataframe(boule_df.sort_values(by="Sorties", ascending=False).head(30).reset_index(drop=True))

    st.subheader("📉 Top 30 Boules les moins sorties")
    st.dataframe(boule_df.sort_values(by="Sorties").head(30).reset_index(drop=True))

    st.subheader("⏳ Top 30 Boules absentes depuis le plus longtemps")
    st.dataframe(boule_df.sort_values(by="Absence (jours)", ascending=False).head(30).reset_index(drop=True))

    
    # 📌 Recommandations intégrées
    st.subheader("📌 Recommandations")
    top = boule_df.sort_values(by="Sorties", ascending=False).head(5)["Numéro"].tolist()
    bottom = boule_df.sort_values(by="Sorties").head(5)["Numéro"].tolist()
    absents = boule_df.sort_values(by="Absence (jours)", ascending=False).head(5)["Numéro"].tolist()
    mix = top[:3] + bottom[:2]
    best_chance = chance_df.sort_values(by="Sorties", ascending=False).head(1)["Numéro Chance"].values[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Top fréquence**")
        st.success(f"{', '.join(str(n) for n in sorted(top))} | Chance : {int(best_chance)}")
    with col2:
        st.markdown("**Moins sortis**")
        st.warning(f"{', '.join(str(n) for n in sorted(bottom))} | Chance : {int(best_chance)}")
    with col3:
        st.markdown("**Absents récents**")
        st.info(f"{', '.join(str(n) for n in sorted(absents))} | Chance : {int(best_chance)}")
    with col4:
        st.markdown("**Mix (3 top + 2 low)**")
        st.info(f"{', '.join(str(n) for n in sorted(mix))} | Chance : {int(best_chance)}")


st.subheader("📈 Numéros Chance les plus sortis")
    st.dataframe(chance_df.sort_values(by="Sorties", ascending=False).reset_index(drop=True))

    st.subheader("📉 Numéros Chance les moins sortis")
    st.dataframe(chance_df.sort_values(by="Sorties").reset_index(drop=True))

    st.subheader("⏳ Numéros Chance absents depuis le plus longtemps")
    st.dataframe(chance_df.sort_values(by="Absence (jours)", ascending=False).reset_index(drop=True))

    st.subheader("📊 Fréquence des boules depuis 2008")
    boules_all = df[['boule_1','boule_2','boule_3','boule_4','boule_5']].values.flatten()
    counts_all = pd.Series(boules_all).value_counts().sort_index()
    mean_all = counts_all.mean()
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    counts_all.plot(kind="bar", ax=ax1)
    ax1.axhline(mean_all, color="red", linestyle="--", label=f"Moyenne {round(mean_all)}")
    ax1.set_title("Fréquence des boules depuis 2008")
    ax1.set_xlabel("Numéro")
    ax1.set_ylabel("Apparitions")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("📊 Fréquence des boules depuis 2023")
    df_2023 = df[df['date_de_tirage'] >= '2023-01-01']
    boules_2023 = df_2023[['boule_1','boule_2','boule_3','boule_4','boule_5']].values.flatten()
    counts_2023 = pd.Series(boules_2023).value_counts().sort_index()
    mean_2023 = counts_2023.mean()
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    counts_2023.plot(kind="bar", ax=ax2, color="orange")
    ax2.axhline(mean_2023, color="red", linestyle="--", label=f"Moyenne {round(mean_2023)}")
    ax2.set_title("Fréquence des boules depuis 2023")
    ax2.set_xlabel("Numéro")
    ax2.set_ylabel("Apparitions")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("🤝 Top 20 des paires les plus fréquentes")
    all_rows = df[['boule_1','boule_2','boule_3','boule_4','boule_5']].dropna().astype(int).values
    paires = []
    for tirage in all_rows:
        tirage_sorted = sorted(tirage)
        paires.extend(combinations(tirage_sorted, 2))
    counter_paires = Counter(paires)
    top_paires = counter_paires.most_common(20)
    paires_df = pd.DataFrame(top_paires, columns=["Paire", "Nombre d'apparitions"])
    paires_df["Paire"] = paires_df["Paire"].apply(lambda x: f"{x[0]} & {x[1]}")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=paires_df, x="Nombre d'apparitions", y="Paire", ax=ax3, palette="Blues_r")
    ax3.set_title("Top 20 des paires de boules les plus fréquentes")
    ax3.set_xlabel("Nombre d'apparitions")
    ax3.set_ylabel("Paires")
    st.pyplot(fig3)
