
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter

st.set_page_config(page_title="Analyse Loto Complète", layout="wide")
st.title("🎯 Analyse Complète des Tirages Loto")

uploaded_file = st.file_uploader("📂 Téléversez le fichier Excel Loto", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="loto_2008")
    df = df[df['boule_1'].notna()]
    df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], errors='coerce')
    df['année'] = df['date_de_tirage'].dt.year
    df['jour'] = df['date_de_tirage'].dt.day_name()

    st.subheader("🎛️ Filtres")
    annees_dispo = sorted(df['année'].dropna().unique())
    jours_dispo = sorted(df['jour'].dropna().unique())
    col1, col2 = st.columns(2)
    with col1:
        années = st.multiselect("Années :", annees_dispo, default=annees_dispo)
    with col2:
        jours = st.multiselect("Jour du tirage :", jours_dispo, default=jours_dispo)
    df_filtré = df[df['année'].isin(années) & df['jour'].isin(jours)]

    st.markdown(f"**{len(df_filtré)} tirages sélectionnés**")

    # === Statistiques ===
    boules = df_filtré[['boule_1','boule_2','boule_3','boule_4','boule_5']].values.flatten()
    boule_counts = pd.Series(boules).value_counts().sort_index()
    chance_counts = df_filtré['numero_chance'].value_counts().sort_index()
    total_tirages = len(df_filtré)

    derniere_sortie = {i: df_filtré[df_filtré[['boule_1','boule_2','boule_3','boule_4','boule_5']].isin([i]).any(axis=1)]['date_de_tirage'].max() for i in range(1, 50)}
    derniere_chance = {i: df_filtré[df_filtré['numero_chance'] == i]['date_de_tirage'].max() for i in range(1, 11)}

    boule_df = pd.DataFrame({
        "Numéro": list(range(1, 50)),
        "Sorties": [boule_counts.get(i, 0) for i in range(1, 50)],
        "% Tirages": [round(boule_counts.get(i, 0)/total_tirages*100, 1) for i in range(1, 50)],
        "Dernière sortie": [derniere_sortie[i] for i in range(1, 50)]
    })
    boule_df["Absence (jours)"] = (df_filtré['date_de_tirage'].max() - boule_df["Dernière sortie"]).dt.days

    chance_df = pd.DataFrame({
        "Numéro Chance": list(range(1, 11)),
        "Sorties": [chance_counts.get(i, 0) for i in range(1, 11)],
        "% Tirages": [round(chance_counts.get(i, 0)/total_tirages*100, 1) for i in range(1, 11)],
        "Dernière sortie": [derniere_chance[i] for i in range(1, 11)]
    })
    chance_df["Absence (jours)"] = (df_filtré['date_de_tirage'].max() - chance_df["Dernière sortie"]).dt.days

    st.subheader("📊 Top 30 Boules les plus sorties")
    st.dataframe(boule_df.sort_values(by="Sorties", ascending=False).head(49).reset_index(drop=True))

    st.subheader("📉 Boules les moins sorties")
    st.dataframe(boule_df.sort_values(by="Sorties").head(49).reset_index(drop=True))

    st.subheader("⏳ Boules absentes depuis le plus longtemps")
    st.dataframe(boule_df.sort_values(by="Absence (jours)", ascending=False).head(49).reset_index(drop=True))

    st.subheader("📊 Numéros Chance les plus sortis")
    st.dataframe(chance_df.sort_values(by="Sorties", ascending=False).reset_index(drop=True))

    st.subheader("📉 Numéros Chance les moins sortis")
    st.dataframe(chance_df.sort_values(by="Sorties").reset_index(drop=True))

    st.subheader("⏳ Numéros Chance absents depuis le plus longtemps")
    st.dataframe(chance_df.sort_values(by="Absence (jours)", ascending=False).reset_index(drop=True))

    # === Recommandations ===
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

    # === Générateur de combinaisons ===
    st.markdown("---")
    st.subheader("🎰 Générateur de Combinaisons")
    gen_type = st.selectbox("Méthode de génération :", [
        "Aléatoire",
        "Pondérée (+ sorties)",
        "Pondérée (- sorties)",
        "Absents (oubliés)",
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
        elif gen_type == "Absents (oubliés)":
            boules_choisies = np.random.choice(absents, size=5, replace=False)
        elif gen_type == "Mix (3 top + 2 low)":
            boules_choisies = np.random.choice(top, 3, replace=False).tolist() +                               np.random.choice(bottom, 2, replace=False).tolist()
        else:
            boules_choisies = np.random.choice(np.arange(1, 50), size=5, replace=False)
        numero_chance = np.random.randint(1, 11)
        return sorted(boules_choisies), numero_chance

    if st.button("Générer une combinaison"):
        boules, chance = generate_combination()
        st.success(f"Boules : {boules} | Numéro Chance : {chance}")


    # 🔥 Boules chaudes / froides
    st.subheader("🔥 Boules chaudes / froides")
    recent_count = st.slider("Analyser les X derniers tirages :", min_value=10, max_value=100, value=20, step=5)
    derniers_tirages = df_filtré.sort_values(by="date_de_tirage", ascending=False).head(recent_count)
    recent_boules = derniers_tirages[['boule_1','boule_2','boule_3','boule_4','boule_5']].values.flatten()
    recent_counts = pd.Series(recent_boules).value_counts().sort_index()

    chauds = recent_counts.sort_values(ascending=False).head(5)
    froids = [n for n in range(1, 50) if n not in recent_counts.index]

    st.markdown("**🔥 Boules chaudes :**")
    st.dataframe(chauds.reset_index().rename(columns={"index": "Numéro", 0: "Apparitions"}))
    st.markdown(f"**❄️ Boules froides (non sorties dans les {recent_count} derniers tirages) :**")
    st.write(sorted(froids))

    # 🧪 Analyse des intervalles
    st.subheader("🧪 Analyse des intervalles")
    interval_data = {}
    for num in range(1, 50):
        dates = df_filtré[df_filtré[['boule_1','boule_2','boule_3','boule_4','boule_5']]
                 .isin([num]).any(axis=1)]['date_de_tirage'].sort_values()
        if len(dates) >= 2:
            intervalles = dates.diff().dt.days.dropna()
            interval_data[num] = {
                "Moyenne (jours)": round(intervalles.mean(), 1),
                "Écart-type": round(intervalles.std(), 1),
                "Dernier tirage": dates.max().date()
            }
    interval_df = pd.DataFrame(interval_data).T.sort_index()
    st.dataframe(interval_df)


    # === Graphiques ===
    st.subheader("📊 Fréquence des boules depuis 2008")
    all_boules = df[['boule_1','boule_2','boule_3','boule_4','boule_5']].values.flatten()
    counts_all = pd.Series(all_boules).value_counts().sort_index()
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

    # === Paires fréquentes ===
    st.subheader("🤝 Top 20 des paires les plus fréquentes")
    all_rows = df[['boule_1','boule_2','boule_3','boule_4','boule_5']].dropna().astype(int).values
    paires = [pair for tirage in all_rows for pair in combinations(sorted(tirage), 2)]
    paires_df = pd.DataFrame(Counter(paires).most_common(20), columns=["Paire", "Nombre d'apparitions"])
    paires_df["Paire"] = paires_df["Paire"].apply(lambda x: f"{x[0]} & {x[1]}")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=paires_df, x="Nombre d'apparitions", y="Paire", ax=ax3, palette="Blues_r")
    ax3.set_title("Top 20 des paires de boules les plus fréquentes")
    ax3.set_xlabel("Nombre d'apparitions")
    ax3.set_ylabel("Paires")
    st.pyplot(fig3)
