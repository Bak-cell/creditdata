import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Charger les modèles et le scaler
model_solvabilite = joblib.load('model_solvabilite.pkl')
model_credit_score = joblib.load('model_credit_score.pkl')
model_pret = joblib.load('model_pret.pkl')
scaler = joblib.load('scaler.pkl')

# Créer de nouveaux encodeurs pour les variables catégorielles
encoders = {}
for column in ['job', 'marital', 'education', 'housing', 'loan']:
    encoders[column] = LabelEncoder()

# Titre principal de l'application avec un style modernisé
st.markdown("""
    <style>
    h1 {
        font-size: 42px;
        text-align: center;
        color: #2C3E50;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<h1>Prédiction de Solvabilité et du Prêt</h1>", unsafe_allow_html=True)

# Message de bienvenue réduit
st.markdown("""
    <h2 style='text-align: center; color: #16A085; font-family: Arial, sans-serif;'>
    Bienvenue dans l'application. Renseignez vos informations pour obtenir des prédictions.
    </h2>
    """, unsafe_allow_html=True)

# Définir des styles CSS pour une meilleure mise en page et des couleurs de texte visibles sur fond blanc
st.markdown("""
    <style>
    .big-font { font-size:22px; color: #2C3E50; font-family: 'Arial', sans-serif; }
    .medium-font { font-size:20px; color: #1ABC9C; font-family: 'Arial', sans-serif; }
    .small-font { font-size:16px; color: #2C3E50; font-family: 'Arial', sans-serif; }

    /* Bordures et style des inputs */
    .stNumberInput, .stSelectbox {
        border: 2px solid #2C3E50C;
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 8px;
        background-color: #ECF0F1;
    }

    /* Style des labels */
    label {
        font-size: 10px;
        color: #1F618D;  /* Couleur modifiée pour les variables spécifiques */
        font-weight: bold;
    }

    /* Style des options de sélection (visibilité accrue sur fond blanc) */
    .stSelectbox, .stNumberInput {
        color: #1F618D;
    }

    /* Centrer les éléments */
    .center {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Formulaire pour les données utilisateurs
st.markdown("<h2 class='big-font'>Formulaire de Solvabilité et Prédiction du Prêt</h2>", unsafe_allow_html=True)

# Saisie des données utilisateur avec meilleure visibilité des options
age = st.number_input('Âge', min_value=18, max_value=120, value=30, key='age_input')
job = st.selectbox("Profession", options=["admin.", "technician", "services", "management", "retired", "blue-collar", "unemployed", "entrepreneur", "housemaid", "self-employed", "student"], key='job_input')
marital = st.selectbox("État civil", options=["married", "single", "divorced"], key='marital_input')
education = st.selectbox("Niveau d'éducation", options=["primary", "secondary", "tertiary", "unknown"], key='education_input')
balance = st.number_input("Solde bancaire($)", min_value=-100000, max_value=1000000, value=0, key='balance_input')
housing = st.selectbox("Possession d'une maison", options=["yes", "no"], key='housing_input')
loan = st.selectbox("Prêt en cours", options=["yes", "no"], key='loan_input')

# Encodage des variables catégorielles
input_data = {
    'age': [age],
    'job': [encoders['job'].fit_transform([job])[0]],  # Encodage dynamique
    'marital': [encoders['marital'].fit_transform([marital])[0]],
    'education': [encoders['education'].fit_transform([education])[0]],
    'balance': [balance],
    'housing': [encoders['housing'].fit_transform([housing])[0]],
    'loan': [encoders['loan'].fit_transform([loan])[0]]
}

df_input = pd.DataFrame(input_data)
df_input_scaled = scaler.transform(df_input)

# Prédiction de la solvabilité et du score de crédit
if st.button('Prédire Solvabilité'):
    solvabilite_pred = model_solvabilite.predict(df_input_scaled)
    solvable = solvabilite_pred[0] == 1  # Vérifier si le client est solvable

    if solvable:
        st.markdown("<p class='medium-font center'> Solvabilité : Le client est Solvable.</p>", unsafe_allow_html=True)
        # Prédire le score de crédit
        score_credit_pred = model_credit_score.predict(df_input)
        st.markdown(f"<p class='medium-font center'> Score de Crédit : {score_credit_pred[0]:.2f}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='medium-font center'> Le client n'est pas solvable, aucun prêt ne peut être accordé.</p>", unsafe_allow_html=True)

# Formulaire pour saisir le score  crédit
score_credit_input = st.number_input('Entrez le Score  Crédit ', min_value=0.0, max_value=1000.0, value=0.0, key='score_credit_input')

# Prédiction du montant du prêt
if st.button('Prédire Montant du Prêt'):
    df_pret = pd.DataFrame({'credit_score': [score_credit_input]})
    montant_pret_pred = model_pret.predict(df_pret)
    st.markdown(f"<p class='medium-font center'> Montant du Prêt Prédit : {montant_pret_pred[0]:,.2f}$</p>", unsafe_allow_html=True)

# Ajouter un footer avec un design modernisé
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='small-font center'>Merci d'avoir utilisé notre application. Nous espérons que cela vous a été utile.</p>", unsafe_allow_html=True)
st.markdown("<p class='small-font center'>© 2024 - Prédiction de Solvabilité & Crédit</p>", unsafe_allow_html=True)
