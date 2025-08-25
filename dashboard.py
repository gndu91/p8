import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_dataset

# --- Configuration de la Page et de l'API ---

# Configure la page Streamlit pour utiliser une mise en page large et définit un titre
# C'est un critère d'accessibilité WCAG (2.4.2 Titre de page)
st.set_page_config(layout="wide", page_title="Dashboard de Scoring Crédit")

API_URL = "https://p8-ikr6.onrender.com/predict"

# Note: Is this good?
sns.set_color_codes("colorblind")

@st.cache_data # Cache la fonction pour ne charger les données qu'une seule fois
def load_data(dataset_name, sample_size=None):
    """
    Charge les données depuis un fichier CSV.
    Prend en charge l'échantillonnage pour les grands fichiers afin d'améliorer les performances.
    """
    try:
        df = get_dataset(dataset_name)
        if sample_size:
            df = df.sample(n=sample_size, random_state=42)
        return df
    except FileNotFoundError:
        st.error(f"ERREUR : Le fichier '{dataset_name}' est introuvable. Assure-toi qu'il se trouve dans le bon dossier.")
        return None

# --- Fonctions de Visualisation ---

def create_gauge_chart(probability, threshold):
    """
    Crée une jauge de score interactive avec Plotly.
    La jauge montre la probabilité de défaut du client et le seuil de décision.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de défaut", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 100], 'color': 'lightgreen'},
                {'range': [threshold * 100, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(paper_bgcolor="lavender", font={'color': "darkblue", 'family': "Arial"})
    return fig

def plot_feature_importance(shap_values):
    """
    Affiche l'importance des features (SHAP values) pour la décision du modèle.
    Sépare les facteurs qui augmentent le risque (en rouge) de ceux qui le diminuent (en vert).
    C'est le graphique CLÉ pour la transparence.
    """
    shap_df = pd.DataFrame(list(shap_values.items()), columns=['feature', 'shap_value'])
    shap_df['abs_shap'] = abs(shap_df['shap_value'])
    shap_df = shap_df.sort_values(by='abs_shap', ascending=False).head(20)

    positive_shap = shap_df[shap_df['shap_value'] > 0].sort_values('shap_value', ascending=False)
    negative_shap = shap_df[shap_df['shap_value'] < 0].sort_values('shap_value', ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle('Interprétation de la décision du modèle', fontsize=16)

    # Facteurs augmentant le risque (SHAP > 0)
    sns.barplot(x='shap_value', y='feature', data=positive_shap, color='red', ax=ax1)
    ax1.set_title('Facteurs augmentant le risque de défaut')
    ax1.set_xlabel('Contribution à la probabilité de défaut')
    ax1.set_ylabel('')

    # Facteurs diminuant le risque (SHAP < 0)
    sns.barplot(x='shap_value', y='feature', data=negative_shap, color='green', ax=ax2)
    ax2.set_title('Facteurs diminuant le risque de défaut')
    ax2.set_xlabel('Contribution à la probabilité de défaut')
    ax2.set_ylabel('')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_comparison(all_data, client_value, feature):
    """
    Affiche la distribution d'une variable pour tous les clients et met en évidence
    la position du client sélectionné avec une ligne verticale.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Sépare les clients qui ont remboursé de ceux qui ont fait défaut pour plus de contexte
    sns.histplot(all_data.loc[all_data['TARGET'] == 0, feature].dropna(),
                 ax=ax, color="green", label="Crédit remboursé", kde=True, stat="density")
    sns.histplot(all_data.loc[all_data['TARGET'] == 1, feature].dropna(),
                 ax=ax, color="red", label="Crédit en défaut", kde=True, stat="density")

    # Ligne verticale pour le client actuel
    ax.axvline(x=client_value, color='purple', linestyle='--', linewidth=2, label='Client actuel')

    ax.set_title(f'Distribution de "{feature}"', fontsize=14)
    ax.set_xlabel(feature)
    ax.set_ylabel('Densité')
    ax.legend()

    return fig

# --- Interface Principale du Dashboard ---

st.title("Dashboard de Scoring Crédit pour 'Prêt à Dépenser'")
st.markdown("Ce tableau de bord interactif aide les chargés de clientèle à comprendre les décisions d'octroi de crédit.")

test_df = load_data('application_test')
train_df = load_data('application_train')

if test_df is None or train_df is None:
    st.stop() # Arrête l'exécution si les données ne sont pas chargées

# --- Panneau de Contrôle Latéral ---
st.sidebar.header("🔍 Sélection du Client")
client_ids = test_df['SK_ID_CURR'].tolist()
selected_id = st.sidebar.selectbox("Choisissez un ID client :", client_ids)

if selected_id:
    # --- Affichage Principal ---
    st.header(f"Analyse du Dossier Client : {selected_id}")

    # Récupère les données du client sélectionné
    client_data = test_df[test_df['SK_ID_CURR'] == selected_id]

    # Prépare les données pour l'API (convertit les NaN en None pour la sérialisation JSON)
    payload = client_data.iloc[0].to_dict()
    payload_cleaned = {k: (None if pd.isna(v) else v) for k, v in payload.items()}

    # Appel à l'API
    with st.spinner("Analyse du dossier en cours..."):
        try:
            response = requests.post(API_URL, json=payload_cleaned)
            response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
            result = response.json()

            # --- Section 1: Résultat de la Décision ---
            st.subheader("Verdict du Modèle")

            proba = result['probability_default']
            decision = result['decision']
            threshold = result['threshold']

            col1, col2 = st.columns(2)

            with col1:
                if decision == "yes":
                    st.success(f"✅ **Crédit Accordé**")
                else:
                    st.error(f"❌ **Crédit Refusé**")

                st.markdown(f"""
                La probabilité de défaut de ce client est estimée à **{proba:.2%}**. 
                Le seuil de décision est à **{threshold:.2%}**.
                Le crédit est **{'accordé' if decision == 'yes' else 'refusé'}** car cette probabilité est {'inférieure' if decision == 'yes' else 'supérieure ou égale'} au seuil.
                """)

            with col2:
                st.plotly_chart(create_gauge_chart(proba, threshold), use_container_width=True)

            # --- Section 2: Explication de la Décision (Transparence) ---
            st.subheader("Facteurs Clés de la Décision")
            st.pyplot(plot_feature_importance(result['feature_importance']))
            st.info("""
            **Comment lire ce graphique ?**
            - **À gauche (en vert)** : Les caractéristiques du client qui ont **diminué** son risque de défaut. Plus la barre est longue, plus l'impact positif est fort.
            - **À droite (en rouge)** : Les caractéristiques qui ont **augmenté** son risque de défaut. Plus la barre est longue, plus l'impact négatif est important.
            """)

            # --- Section 3: Comparaison du Client ---
            st.subheader("Positionnement du Client par Rapport aux Autres")

            # Liste des features les plus importantes à comparer (à adapter si besoin)
            important_features = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
            # Ajoute les features SHAP les plus importantes pour ce client spécifique
            top_shap_features = pd.DataFrame(list(result['feature_importance'].items()), columns=['feature', 'shap']) \
                .reindex(abs(pd.Series(result['feature_importance'])).sort_values(ascending=False).index) \
                .query('feature in @train_df.columns')['feature'].head(5).tolist()

            features_to_compare = sorted(list(set(important_features + top_shap_features)))

            selected_feature = st.selectbox("Choisissez une caractéristique à comparer :", features_to_compare)

            if selected_feature:
                client_value = client_data.iloc[0][selected_feature]
                st.pyplot(plot_comparison(train_df, client_value, selected_feature))
                st.markdown(f"La valeur du client pour **{selected_feature}** est de **{client_value:,.2f}**.")

            # --- Section 4: Informations Descriptives ---
            with st.expander("Voir les informations détaillées du client"):
                st.dataframe(client_data)


        except requests.exceptions.HTTPError as e:
            st.error(f"Erreur HTTP lors de l'appel à l'API : {e.response.status_code}")
            st.error(f"Détail de l'erreur : {e.response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion à l'API à l'adresse : {API_URL}")
            st.info("Vérifiez que l'API est bien en cours d'exécution et accessible.")
        except Exception as e:
            st.error(f"Une erreur inattendue est survenue : {e}")

else:
    st.info("👈 Veuillez sélectionner un ID client dans le menu de gauche pour commencer l'analyse.")