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

# --- Constantes pour l'accessibilité ---
COLOR_ACCORDE = "#1f77b4"
COLOR_REFUSE = "#ff7f0e"

API_URL = "https://p8-ikr6.onrender.com/predict"

@st.cache_data
def load_data(dataset_name, sample_size=None):
    """Charge les données et les met en cache."""
    try:
        df = pd.read_csv(dataset_name + '.csv')
        if sample_size:
            return df.sample(n=sample_size, random_state=42)
        return df
    except FileNotFoundError:
        st.error(f"ERREUR : Le fichier '{dataset_name}' est introuvable.")
        return None

# --- Fonctions de Visualisation (inchangées) ---
def create_gauge_chart(probability, threshold):
    """Crée une jauge de score avec des couleurs accessibles."""
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
                {'range': [0, threshold * 100], 'color': 'lightsteelblue'},
                {'range': [threshold * 100, 100], 'color': 'moccasin'}
            ],
            'threshold': {
                'line': {'color': COLOR_REFUSE, 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(paper_bgcolor="#f0f2f6", font={'color': "darkblue", 'family': "Arial"})
    return fig

def plot_feature_importance(shap_values):
    """Affiche l'importance des features avec des couleurs accessibles."""
    shap_df = pd.DataFrame(list(shap_values.items()), columns=['feature', 'shap_value'])
    shap_df['abs_shap'] = abs(shap_df['shap_value'])
    shap_df = shap_df.sort_values(by='abs_shap', ascending=False).head(20)

    positive_shap = shap_df[shap_df['shap_value'] > 0].sort_values('shap_value', ascending=False)
    negative_shap = shap_df[shap_df['shap_value'] < 0].sort_values('shap_value', ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle('Interprétation de la décision du modèle', fontsize=18)

    sns.barplot(x='shap_value', y='feature', data=positive_shap, color=COLOR_REFUSE, ax=ax1)
    ax1.set_title('Facteurs augmentant le risque de défaut', fontsize=14)
    ax1.set_xlabel('Contribution à la probabilité de défaut')
    ax1.set_ylabel('')

    sns.barplot(x='shap_value', y='feature', data=negative_shap, color=COLOR_ACCORDE, ax=ax2)
    ax2.set_title('Facteurs diminuant le risque de défaut', fontsize=14)
    ax2.set_xlabel('Contribution à la probabilité de défaut')
    ax2.set_ylabel('')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_comparison(all_data, client_value, feature):
    """Affiche la distribution d'une variable avec des couleurs accessibles."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Sépare les clients qui ont remboursé de ceux qui ont fait défaut pour plus de contexte
    sns.histplot(all_data.loc[all_data['TARGET'] == 0, feature].dropna(),
                 ax=ax, color=COLOR_ACCORDE, label="Crédit remboursé", kde=True, stat="density")
    sns.histplot(all_data.loc[all_data['TARGET'] == 1, feature].dropna(),
                 ax=ax, color=COLOR_REFUSE, label="Crédit en défaut", kde=True, stat="density")

    ax.axvline(x=client_value, color='black', linestyle='--', linewidth=2, label='Client actuel')
    ax.set_title(f'Distribution de "{feature}"', fontsize=16)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('Densité', fontsize=12)
    ax.legend()
    return fig

def plot_bivariate_analysis(all_data, client_data, feature_x, feature_y):
    """Affiche un nuage de points pour comparer deux variables."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data_sample = all_data.sample(n=min(len(all_data), 2000), random_state=42)

    sns.scatterplot(data=data_sample, x=feature_x, y=feature_y, hue='TARGET',
                    palette=[COLOR_ACCORDE, COLOR_REFUSE], ax=ax, alpha=0.5)

    ax.scatter(client_data[feature_x], client_data[feature_y],
               marker='*', s=200, edgecolor='black', facecolor='yellow',
               label='Client actuel', zorder=3)

    ax.set_title(f'Analyse de "{feature_x}" par rapport à "{feature_y}"', fontsize=16)
    ax.set_xlabel(feature_x, fontsize=12)
    ax.set_ylabel(feature_y, fontsize=12)
    ax.legend(title='Statut du crédit')
    return fig


# --- Interface Principale du Dashboard ---
st.title("Dashboard de Scoring Crédit pour 'Prêt à Dépenser'")
st.markdown("Ce tableau de bord interactif aide les chargés de clientèle à comprendre les décisions d'octroi de crédit.")

test_df = load_data('application_test')
train_df = load_data('application_train')

if test_df is None or train_df is None:
    st.stop()

st.sidebar.header("🔍 Sélection du Client")
client_ids = test_df['SK_ID_CURR'].tolist()

# --- NOUVEAU : GESTION DES PARAMÈTRES D'URL POUR LA SÉLECTION DU CLIENT ---
default_index = 0
# Essaye de récupérer l'ID client depuis l'URL
client_id_from_url = st.query_params.get("id")

if client_id_from_url:
    try:
        # Tente de trouver l'index de cet ID dans notre liste
        client_id_int = int(client_id_from_url)
        default_index = client_ids.index(client_id_int)
    except (ValueError, IndexError):
        # Si l'ID est invalide ou non trouvé, on affiche un avertissement
        st.sidebar.warning(f"L'ID client '{client_id_from_url}' est invalide ou n'a pas été trouvé.")
        # On supprime le mauvais paramètre de l'URL pour éviter toute confusion
        st.query_params.clear()

# --- MODIFIÉ : Le selectbox utilise maintenant `default_index` ---
selected_id = st.sidebar.selectbox(
    "Choisissez un ID client :",
    client_ids,
    index=default_index
)

if selected_id:
    # --- NOUVEAU : Mettre à jour l'URL à chaque changement de sélection ---
    # Cela garantit que l'URL reflète toujours le client affiché
    st.query_params["id"] = selected_id

    st.header(f"Analyse du Dossier Client : {selected_id}")

    client_data = test_df[test_df['SK_ID_CURR'] == selected_id]
    payload = client_data.iloc[0].to_dict()
    payload_cleaned = {k: (None if pd.isna(v) else v) for k, v in payload.items()}

    with st.spinner("Analyse du dossier en cours..."):
        try:
            response = requests.post(API_URL, json=payload_cleaned)
            response.raise_for_status()
            result = response.json()

            # --- Section 1: Résultat de la Décision ---
            st.subheader("Verdict du Modèle")
            col1, col2 = st.columns(2)
            with col1:
                proba, decision, threshold = result['probability_default'], result['decision'], result['threshold']
                if decision == "yes":
                    st.success("✅ **Crédit Accordé**")
                else:
                    st.error("❌ **Crédit Refusé**")
                st.markdown(f"La probabilité de défaut est de **{proba:.2%}** (seuil à {threshold:.2%}).")
            with col2:
                st.plotly_chart(create_gauge_chart(proba, threshold), use_container_width=True)
            st.markdown("*Description : Cette jauge affiche la probabilité de défaut calculée pour le client. La zone bleue représente la zone d'acceptation du crédit, tandis que la zone orange représente la zone de refus. La ligne verticale indique le seuil de décision.*")
            st.divider()

            # --- Section 2: Explication de la Décision ---
            st.subheader("Facteurs Clés de la Décision")
            st.pyplot(plot_feature_importance(result['feature_importance']))
            st.markdown("*Description : Ce graphique montre les caractéristiques du client qui ont le plus influencé la décision. En bleu, les facteurs qui ont diminué le risque de défaut. En orange, ceux qui l'ont augmenté. La longueur de la barre indique la force de l'impact.*")
            st.divider()

            # --- Section 3: Comparaison du Client (Analyse univariée et bivariée) ---
            st.subheader("Positionnement du Client par Rapport aux Autres")
            important_features = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_ANNUITY']
            top_shap_features = pd.Series(result['feature_importance']).abs().sort_values(ascending=False).head(5).index.tolist()
            features_to_compare = sorted(list(set(important_features + top_shap_features) & set(train_df.columns)))

            st.markdown("#### Analyse d'une caractéristique")
            selected_feature = st.selectbox("Choisissez une caractéristique à comparer :", features_to_compare)
            if selected_feature:
                client_value = client_data.iloc[0].get(selected_feature)
                if pd.notna(client_value):
                    st.pyplot(plot_comparison(train_df, client_value, selected_feature))
                    st.markdown(f"*Description : Cet histogramme compare la position du client (ligne noire pointillée) à celle de tous les autres clients pour la caractéristique '{selected_feature}'. La distribution des clients ayant remboursé leur prêt est en bleu, celle des clients en défaut est en orange.*")
                    st.markdown(f"La valeur du client pour **{selected_feature}** est de **{client_value:,.2f}**.")
                else:
                    st.warning(f"Le client n'a pas de valeur pour la caractéristique '{selected_feature}'.")

            st.markdown("#### Analyse Bi-variée")
            col_x, col_y = st.columns(2)
            with col_x:
                feature_x = st.selectbox("Choisissez une variable pour l'axe X :", features_to_compare, index=0)
            with col_y:
                feature_y = st.selectbox("Choisissez une variable pour l'axe Y :", features_to_compare, index=1)

            if feature_x and feature_y and feature_x != feature_y:
                st.pyplot(plot_bivariate_analysis(train_df, client_data, feature_x, feature_y))
                st.markdown(f"*Description : Ce nuage de points montre la relation entre '{feature_x}' et '{feature_y}' pour un échantillon de clients. Le client actuel est mis en évidence par l'étoile jaune. Cela permet d'identifier des tendances et de situer le client dans un contexte à deux dimensions.*")
            elif feature_x == feature_y:
                st.warning("Veuillez sélectionner deux caractéristiques différentes.")

            st.divider()

            # --- Section 4: Informations Descriptives ---
            with st.expander("Voir les informations détaillées du client"):
                st.dataframe(client_data)

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion à l'API ({API_URL}). Vérifiez que l'API est en ligne. Détail : {e}")
        except Exception as e:
            st.error(f"Une erreur inattendue est survenue : {e}")
else:
    st.info("👈 Veuillez sélectionner un ID client dans le menu de gauche pour commencer l'analyse.")