import streamlit as st
import os
from plant_rag_system import PlantRAGSystem

# Configuration de la page
st.set_page_config(
    page_title="🌱 Coach Plantes - Assistant RAG",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🌱 Coach Plantes - Assistant RAG")
st.markdown("**Système d'intelligence artificielle pour l'entretien des plantes**")

# Initialisation du système RAG
@st.cache_resource
def init_rag_system():
    """Initialise le système RAG et le met en cache"""
    with st.spinner("🌱 Initialisation du système RAG..."):
        rag_system = PlantRAGSystem()
        rag_system.load_plant_files()
        rag_system.create_vectorstore()
        return rag_system

# Initialiser le système
try:
    rag_system = init_rag_system()
    st.success("✅ Système RAG initialisé avec succès !")
except Exception as e:
    st.error(f"❌ Erreur lors de l'initialisation : {e}")
    st.stop()

# Sidebar avec les fonctionnalités
st.sidebar.header("🔧 Fonctionnalités")

# Sélection de la fonctionnalité
feature = st.sidebar.selectbox(
    "Choisir une fonctionnalité :",
    ["🏠 Accueil", "🔍 Recherche de plantes", "❓ Questions/Réponses", "📊 Statistiques du dataset"]
)

if feature == "🏠 Accueil":
    st.header("🏠 Bienvenue dans votre Coach Plantes !")
    
    st.markdown("""
    Ce système RAG (Retrieval-Augmented Generation) vous permet de :
    
    - **🔍 Rechercher** des plantes selon vos critères
    - **❓ Poser des questions** sur l'entretien des plantes
    - **📚 Consulter** une base de connaissances complète sur les plantes d'intérieur
    
    ### 📁 Dataset disponible
    Le système a été entraîné sur les fiches suivantes :
    """)
    
    # Afficher les plantes disponibles
    if hasattr(rag_system, 'documents'):
        plant_info = []
        for doc in rag_system.documents:
            metadata = doc.metadata
            plant_info.append({
                "Nom": metadata.get("title", "Sans titre"),
                "Nom scientifique": metadata.get("scientific_name", "N/A"),
                "Famille": metadata.get("family", "N/A"),
                "Difficulté": metadata.get("difficulty", "N/A"),
                "Lumière": metadata.get("light", "N/A"),
                "Eau": metadata.get("water", "N/A")
            })
        
        st.dataframe(plant_info, use_container_width=True)

elif feature == "🔍 Recherche de plantes":
    st.header("🔍 Recherche de plantes")
    
    # Interface de recherche
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Entrez votre recherche :",
            placeholder="Ex: plante facile à entretenir, succulente, plante d'ombre..."
        )
    
    with col2:
        num_results = st.number_input("Nombre de résultats", min_value=1, max_value=10, value=5)
    
    # Bouton de recherche
    if st.button("🔍 Rechercher", type="primary"):
        if search_query:
            with st.spinner("🔍 Recherche en cours..."):
                try:
                    results = rag_system.search_plants(search_query, k=num_results)
                    
                    st.success(f"✅ {len(results)} résultats trouvés")
                    
                    # Afficher les résultats
                    for i, doc in enumerate(results, 1):
                        with st.expander(f"🌱 {doc.metadata.get('title', 'Sans titre')}"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown("**Contenu :**")
                                st.write(doc.page_content)
                            
                            with col2:
                                st.markdown("**Métadonnées :**")
                                metadata = doc.metadata
                                st.write(f"**Nom scientifique :** {metadata.get('scientific_name', 'N/A')}")
                                st.write(f"**Famille :** {metadata.get('family', 'N/A')}")
                                st.write(f"**Difficulté :** {metadata.get('difficulty', 'N/A')}/5")
                                st.write(f"**Lumière :** {metadata.get('light', 'N/A')}")
                                st.write(f"**Eau :** {metadata.get('water', 'N/A')}")
                                
                                if metadata.get('tags'):
                                    st.write(f"**Tags :** {', '.join(metadata.get('tags', []))}")
                                
                                st.write(f"**Source :** {os.path.basename(metadata.get('source', 'N/A'))}")
                except Exception as e:
                    st.error(f"❌ Erreur lors de la recherche : {e}")
        else:
            st.warning("⚠️ Veuillez entrer une requête de recherche")

elif feature == "❓ Questions/Réponses":
    st.header("❓ Questions/Réponses")
    
    st.markdown("""
    Posez vos questions sur l'entretien des plantes et obtenez des réponses basées sur notre base de connaissances !
    
    **Exemples de questions :**
    - Comment arroser une succulente ?
    - Quelle plante choisir pour un débutant ?
    - Comment entretenir une plante araignée ?
    - Quelles plantes sont bonnes pour la dépollution ?
    """)
    
    # Interface de question
    question = st.text_area(
        "Votre question :",
        placeholder="Ex: Comment entretenir une plante araignée ?",
        height=100
    )
    
    if st.button("❓ Poser la question", type="primary"):
        if question:
            with st.spinner("🤔 Génération de la réponse..."):
                try:
                    answer = rag_system.answer_question(question)
                    
                    st.success("✅ Réponse générée !")
                    
                    # Afficher la réponse
                    st.markdown("### 💡 Réponse :")
                    st.write(answer)
                    
                    # Afficher les sources utilisées
                    st.markdown("### 📚 Sources utilisées :")
                    relevant_docs = rag_system.search_plants(question, k=3)
                    for i, doc in enumerate(relevant_docs, 1):
                        st.write(f"{i}. **{doc.metadata.get('title', 'Sans titre')}**")
                        st.write(f"   {doc.page_content[:200]}...")
                        st.divider()
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération de la réponse : {e}")
        else:
            st.warning("⚠️ Veuillez entrer une question")

elif feature == "📊 Statistiques du dataset":
    st.header("📊 Statistiques du dataset")
    
    if hasattr(rag_system, 'documents'):
        st.markdown("### 📈 Vue d'ensemble")
        
        # Statistiques générales
        total_plants = len(rag_system.documents)
        st.metric("Total des plantes", total_plants)
        
        # Analyse par difficulté
        difficulty_counts = {}
        family_counts = {}
        light_counts = {}
        water_counts = {}
        
        for doc in rag_system.documents:
            metadata = doc.metadata
            
            # Difficulté
            diff = metadata.get('difficulty', 0)
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            
            # Famille
            family = metadata.get('family', 'Inconnue')
            family_counts[family] = family_counts.get(family, 0) + 1
            
            # Lumière
            light = metadata.get('light', 'Inconnue')
            light_counts[light] = light_counts.get(light, 0) + 1
            
            # Eau
            water = metadata.get('water', 'Inconnue')
            water_counts[water] = water_counts.get(water, 0) + 1
        
        # Affichage des statistiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Répartition par difficulté :**")
            for diff, count in sorted(difficulty_counts.items()):
                st.write(f"Difficulté {diff}/5 : {count} plante(s)")
            
            st.markdown("**🌿 Répartition par famille :**")
            for family, count in sorted(family_counts.items()):
                st.write(f"{family} : {count} plante(s)")
        
        with col2:
            st.markdown("**💡 Répartition par besoin en lumière :**")
            for light, count in sorted(light_counts.items()):
                st.write(f"{light} : {count} plante(s)")
            
            st.markdown("**💧 Répartition par besoin en eau :**")
            for water, count in sorted(water_counts.items()):
                st.write(f"{water} : {count} plante(s)")
        
        # Tableau détaillé
        st.markdown("### 📋 Détail des plantes")
        plant_details = []
        for doc in rag_system.documents:
            metadata = doc.metadata
            plant_details.append({
                "Nom": metadata.get("title", "Sans titre"),
                "Nom scientifique": metadata.get("scientific_name", "N/A"),
                "Famille": metadata.get("family", "N/A"),
                "Difficulté": metadata.get("difficulty", "N/A"),
                "Lumière": metadata.get("light", "N/A"),
                "Eau": metadata.get("water", "N/A"),
                "Tags": ", ".join(metadata.get("tags", []))
            })
        
        st.dataframe(plant_details, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🌱 **Coach Plantes RAG** - Système d'intelligence artificielle pour l'entretien des plantes")
st.markdown("*Développé avec LangChain, ChromaDB et Streamlit*")
