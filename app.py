# ============================================================
# Application Streamlit : risque de réadmission d'un patient diabétique
# Lancer en local :  streamlit run app.py
# ============================================================
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Risque de réadmission", page_icon="🏥", layout="wide")


# ------------------------------------------------------------
# 1) Chargement du modèle (un seul fichier : modèle + métadonnées)
# ------------------------------------------------------------
@st.cache_resource
def charger_modele():
    paquet = joblib.load(Path(__file__).parent / "model.joblib")   # fichier à côté de app.py
    return paquet["model"], paquet["meta"]


model, meta = charger_modele()
COLONNES = meta["colonnes"]          # ordre exact des colonnes vues à l'entraînement
SEUIL = meta["seuil"]                # seuil de décision choisi dans le notebook (0,45)
NIVEAUX = meta["niveaux"]            # bornes des niveaux de risque (quartiles des patients de test)
PERCENTILES = np.array(meta["percentiles"])   # répartition des probabilités sur les patients de test


# ------------------------------------------------------------
# 2) Préparation : même transformation que dans le notebook
# ------------------------------------------------------------
def preparer(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme des lignes au format data_clean.csv en colonnes du modèle."""
    X = df.drop(columns=["readmitted"], errors="ignore").copy()
    X["total_visites"] = X["number_inpatient"] + X["number_emergency"] + X["number_outpatient"]
    X["deja_hospitalise"] = (X["number_inpatient"] > 0).astype(int)
    X["medicaments_par_jour"] = X["num_medications"] / X["time_in_hospital"]
    X = pd.get_dummies(X, columns=["discharge_disposition_id"], dtype=int)
    return X.reindex(columns=COLONNES, fill_value=0)              # colonnes manquantes = 0, même ordre


def niveau(p: float) -> tuple[str, str]:
    if p < NIVEAUX["faible"]:
        return "Faible", "🟢"
    if p < NIVEAUX["eleve"]:
        return "Modéré", "🟡"
    if p < NIVEAUX["tres_eleve"]:
        return "Élevé", "🟠"
    return "Très élevé", "🔴"


def one_hot(prefixe: str, choix: str, options: list[str]) -> dict:
    """Crée les colonnes 0/1 d'un groupe (ex. race_Caucasian = 1, les autres = 0)."""
    return {f"{prefixe}_{o}": int(o == choix) for o in options}


# ------------------------------------------------------------
# 3) Interface
# ------------------------------------------------------------
st.title("🏥 Risque de réadmission d'un patient diabétique")
st.write(
    "Ce modèle **XGBoost** estime la probabilité qu'un patient diabétique **revienne à l'hôpital** "
    "après sa sortie (réadmission avant ou après 30 jours)."
)

with st.sidebar:
    st.header("Le modèle")
    m = meta["metriques_test"]
    st.metric("Réadmis détectés (recall)", f"{m['recall']:.0%}")
    st.metric("Alertes justes (precision)", f"{m['precision']:.0%}")
    st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    st.caption(
        f"Résultats sur 13 738 patients de test. Un patient est signalé « à risque » "
        f"si sa probabilité dépasse **{SEUIL:.0%}**."
    )
    st.caption("Outil d'aide à la décision : il ne remplace pas l'avis d'un médecin.")

onglet1, onglet2 = st.tabs(["👤 Un patient", "📄 Plusieurs patients (CSV)"])

# ---------------- Onglet 1 : un patient ----------------
with onglet1:
    with st.form("patient"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Patient")
            tranches = {f"{a}-{a + 10} ans": a + 5 for a in range(0, 100, 10)}
            age = st.selectbox("Âge", list(tranches), index=7)
            genre = st.radio("Genre", ["Femme", "Homme"], horizontal=True)
            race = st.selectbox("Origine", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])

            st.subheader("Historique (année précédente)")
            inpatient = st.number_input("Hospitalisations", 0, 12, 0)
            emergency = st.number_input("Passages aux urgences", 0, 42, 0)
            outpatient = st.number_input("Consultations externes", 0, 42, 0)

        with c2:
            st.subheader("Séjour")
            admission_type = st.selectbox("Type d'admission", ["Urgence", "Programmee", "Autre"],
                                          format_func=lambda x: {"Programmee": "Programmée"}.get(x, x))
            admission_source = st.selectbox("Origine de l'admission", ["Urgences", "Medecin", "Autre"],
                                            format_func=lambda x: {"Medecin": "Envoyé par un médecin"}.get(x, x))
            sorties = {"Domicile": "Domicile", "Domicile_soins": "Domicile avec soins",
                       "Etablissement": "Établissement de soins", "Autre_hopital": "Autre hôpital",
                       "Autre": "Autre", "Inconnu": "Inconnue"}
            sortie = st.selectbox("Destination de sortie", list(sorties), format_func=sorties.get)
            duree = st.slider("Durée du séjour (jours)", 1, 14, 4)
            labo = st.slider("Analyses de laboratoire", 1, 132, 44)
            procedures = st.slider("Actes médicaux", 0, 6, 1)
            nb_diag = st.slider("Nombre de diagnostics", 1, 16, 8)

        with c3:
            st.subheader("Diagnostics")
            familles = ["Circulatoire", "Respiratoire", "Diabete", "Digestif", "Blessure",
                        "Musculaire", "Urinaire", "Tumeur", "Autre"]
            noms = {"Diabete": "Diabète"}
            diag1 = st.selectbox("Diagnostic principal", familles, format_func=lambda x: noms.get(x, x))
            diag2 = st.selectbox("Diagnostic secondaire", familles, format_func=lambda x: noms.get(x, x))
            diag3 = st.selectbox("Diagnostic additionnel", familles, index=2, format_func=lambda x: noms.get(x, x))

            st.subheader("Traitement")
            nb_med = st.slider("Médicaments administrés", 1, 81, 14)
            diabetes_med = st.checkbox("Traitement contre le diabète", value=True)
            nb_antidiab = st.slider("Nombre d'antidiabétiques", 0, 6, 1)
            insuline = st.checkbox("Insuline")
            metformine = st.checkbox("Metformine")
            changement = st.checkbox("Traitement modifié pendant le séjour")
            a1c = st.selectbox("HbA1c", ["Non mesurée", "Normale", "> 7 %", "> 8 %"])
            glycemie = st.selectbox("Glycémie max", ["Non mesurée", "Normale", "> 200", "> 300"])

        valider = st.form_submit_button("Estimer le risque", type="primary", use_container_width=True)

    if valider:
        ligne = {
            "gender": int(genre == "Homme"),
            "age": tranches[age],
            "discharge_disposition_id": sortie,
            "time_in_hospital": duree,
            "num_lab_procedures": labo,
            "num_procedures": procedures,
            "num_medications": nb_med,
            "number_outpatient": np.log1p(outpatient),   # même transformation que data_clean.csv
            "number_emergency": np.log1p(emergency),
            "number_inpatient": np.log1p(inpatient),
            "number_diagnoses": nb_diag,
            "max_glu_serum": ["Non mesurée", "Normale", "> 200", "> 300"].index(glycemie),
            "A1Cresult": ["Non mesurée", "Normale", "> 7 %", "> 8 %"].index(a1c),
            "change": int(changement),
            "diabetesMed": int(diabetes_med),
            "nb_medicaments": nb_antidiab,
            "insulin_bin": int(insuline),
            "metformin_bin": int(metformine),
            **one_hot("race", race, ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"]),
            **one_hot("admission_type_id", admission_type, ["Autre", "Programmee", "Urgence"]),
            **one_hot("admission_source_id", admission_source, ["Autre", "Medecin", "Urgences"]),
            **one_hot("diag_1", diag1, familles),
            **one_hot("diag_2", diag2, familles),
            **one_hot("diag_3", diag3, familles),
        }
        proba = float(model.predict_proba(preparer(pd.DataFrame([ligne])))[0, 1])
        nom, pastille = niveau(proba)

        rang = int(np.searchsorted(PERCENTILES, proba))  # % de patients de test avec une probabilité plus basse
        st.divider()
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Probabilité de réadmission", f"{proba:.0%}")
            st.progress(min(proba, 1.0))
            st.caption(f"Plus à risque que **{min(rang, 100)} %** des patients de test.")
        with r2:
            st.subheader(f"{pastille} Risque {nom.lower()}")
            if proba >= SEUIL:
                st.warning("Patient **à risque** : un suivi renforcé après la sortie est recommandé "
                           "(appel de suivi, rendez-vous rapproché).")
            else:
                st.success("Patient **à faible risque** : suivi habituel.")
            st.caption("Le niveau compare ce patient aux autres : faible = 25 % les moins à risque, "
                       "très élevé = 25 % les plus à risque.")
            if inpatient > 0:
                st.caption("Facteur important : le patient a déjà été hospitalisé l'année précédente.")

# ---------------- Onglet 2 : plusieurs patients ----------------
with onglet2:
    st.write(
        "Importez un fichier CSV au **même format que `data_clean.csv`** "
        "(la colonne `readmitted` est facultative). Les patients décédés sont ignorés."
    )
    fichier = st.file_uploader("Fichier CSV", type="csv")
    if fichier is not None:
        df = pd.read_csv(fichier)
        manquantes = [c for c in ["number_inpatient", "number_emergency", "number_outpatient",
                                  "num_medications", "time_in_hospital", "discharge_disposition_id"]
                      if c not in df.columns]
        if manquantes:
            st.error(f"Colonnes manquantes : {', '.join(manquantes)}")
        else:
            df = df[df["discharge_disposition_id"] != "Deces"].reset_index(drop=True)
            probas = model.predict_proba(preparer(df))[:, 1]
            resultat = df.copy()
            resultat["probabilite_readmission"] = probas.round(3)
            resultat["a_risque"] = (probas >= SEUIL).astype(int)
            resultat["niveau_risque"] = [niveau(p)[0] for p in probas]

            k1, k2, k3 = st.columns(3)
            k1.metric("Patients analysés", f"{len(resultat):,}".replace(",", " "))
            k2.metric("Patients à risque", f"{resultat['a_risque'].sum():,}".replace(",", " "))
            k3.metric("Part à risque", f"{resultat['a_risque'].mean():.0%}")

            st.dataframe(resultat[["probabilite_readmission", "niveau_risque", "a_risque"]
                                  + [c for c in ["readmitted"] if c in resultat]].head(200),
                         use_container_width=True)
            st.download_button("Télécharger les prédictions", resultat.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")
