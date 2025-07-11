# Organisateur de Tournées Automatique 🚛

Application Streamlit d'optimisation intelligente pour itinéraires de livraison avec géocodage automatique et algorithmes d'optimisation avancés.

## 🎯 Fonctionnalités principales

### ✅ Traitement automatique des données
- **Import Excel** : Support format `.xlsx` avec détection automatique des colonnes
- **Géocodage intelligent** : API Nominatim gratuite avec retry automatique et gestion d'erreurs
- **Validation des données** : Contrôles de qualité et nettoyage automatique
- **Gestion des quotas API** : Limitations configurables pour éviter les surcharges

### 🎯 Optimisation de tournées
- **Détection du centre-ville** : Algorithme de densité pondérée pour identifier le point de départ optimal
- **Secteur intelligent** : Filtrage automatique des adresses trop éloignées (hors secteur)
- **Optimisation multi-étapes** : Combinaison d'algorithmes (angulaire + 2-opt + plus proche voisin)
- **Route réaliste** : Parcours chronologique et géographiquement cohérent

### 📊 Métriques et qualité
- **Indicateurs de performance** : Distance totale, temps estimé, efficacité
- **Score de qualité** : Évaluation composite (efficacité, régularité, compacité)
- **Alertes intelligentes** : Conseils d'optimisation personnalisés
- **Statistiques détaillées** : Taux de géocodage, distribution des distances

### 💾 Export professionnel
- **Fichier Excel multi-feuilles** :
  - Tournée optimisée avec ordre, distances et coordonnées GPS
  - Adresses hors secteur avec distances du centre
  - Échecs de géocodage pour correction manuelle
  - Statistiques complètes de la tournée
- **Formatage automatique** : En-têtes colorés, largeurs optimisées
- **Métadonnées** : Temps cumulé, coordonnées GPS, horodatage

## 🚀 Améliorations récentes

### Algorithmes d'optimisation
- **Géocodage robuste** : Backoff exponentiel, gestion spécifique des erreurs
- **Détection de centre améliorée** : Pondération par densité inverse des distances
- **Filtrage intelligent** : Méthode des quartiles pour détecter les outliers
- **Optimisation 2-opt étendue** : Plus d'itérations pour de meilleurs résultats

### Interface utilisateur
- **Barre de progrès intelligente** : Estimation temps restant, compteur de succès
- **Métriques avancées** : Affichage en temps réel des scores de qualité
- **Détection automatique** : Colonnes identifiées par algorithme de scoring
- **Validation proactive** : Contrôles d'intégrité avant traitement

### Qualité et fiabilité
- **Type hints** : Code Python typé pour une meilleure maintenabilité
- **Logging structuré** : Messages d'erreur et de debug organisés
- **Tests unitaires** : Validation des fonctions critiques
- **Gestion d'exceptions** : Try/catch spécifiques avec fallbacks

## 📋 Installation et utilisation

### Prérequis
```bash
Python 3.7+
pip install -r requirements.txt
```

### Lancement
```bash
streamlit run app.py
```

### Tests
```bash
python test_app.py
```

## 📊 Format des données

### Fichier Excel d'entrée
Le fichier doit contenir au minimum :
- **Adresse** : Rue, numéro, bâtiment
- **Code postal** : Format numérique
- **Ville** : Nom de la commune

**Exemple :**
```
Adresse_Client          | Code_Postal | Ville
123 rue de Rivoli      | 75001       | Paris
456 avenue des Champs  | 75008       | Paris
```

### Détection automatique
L'application détecte automatiquement les colonnes selon des mots-clés :
- **Adresse** : adresse, address, client, livraison, rue, street, voie
- **Code postal** : postal, cp, code, zip, postcode
- **Ville** : ville, city, commune, localité, locality

## 🎯 Résultats attendus

### Performance
- **Géocodage** : >90% de réussite avec retry intelligent
- **Optimisation** : 10-15% d'amélioration de la distance totale
- **Secteur** : Filtrage automatique des adresses >15km du centre

### Qualité
- **Routes logiques** : Départ centre-ville vers périphérie
- **Temps réaliste** : 3 min/km + 5 min/arrêt
- **Métriques** : Efficacité, régularité, compacité mesurées

### Export
- **Fichier Excel complet** : 3-4 feuilles selon les données
- **Coordonnées GPS** : Latitude/longitude pour navigation
- **Statistiques** : Métriques complètes de la tournée

## 🔧 Configuration

### Paramètres ajustables
- **Rayon maximum** : 5-25 km (défaut: 15 km)
- **Nombre maximum d'adresses** : 10-200 (défaut: 100)
- **Retry géocodage** : 3 tentatives avec backoff exponentiel
- **Délai API** : 1.2 secondes entre requêtes

### Algorithmes
- **Centre-ville** : Densité pondérée (rayon 3km)
- **Filtrage** : Quartiles + IQR pour outliers
- **Optimisation** : 2-opt + plus proche voisin
- **Qualité** : Score composite normalisé

## 📈 Cas d'usage

### Livraisons urbaines
- **Secteur cohérent** : Adresses dans un rayon de 15km
- **Optimisation chronologique** : Départ centre → périphérie
- **Gestion des exceptions** : Adresses hors secteur identifiées

### Tournées commerciales
- **Itinéraire logique** : Minimisation des allers-retours
- **Temps réaliste** : Estimation incluant conduite + arrêts
- **Flexibilité** : Adaptation aux contraintes locales

### Logistique
- **Planification** : Optimisation préalable des tournées
- **Validation** : Contrôles qualité automatisés
- **Reporting** : Statistiques détaillées pour analyse

## 🎉 Avantages

### Automatisation
- **Zéro configuration** : Détection automatique des colonnes
- **Traitement batch** : Jusqu'à 200 adresses simultanément
- **Validation proactive** : Contrôles avant traitement

### Optimisation
- **Algorithmes éprouvés** : Combinaison de techniques d'optimisation
- **Adaptation intelligente** : Paramètres ajustés aux données
- **Qualité mesurée** : Métriques objectives de performance

### Professionnalisme
- **Export Excel complet** : Prêt pour utilisation terrain
- **Formatage automatique** : Présentation soignée
- **Métadonnées** : Informations complètes pour suivi

---

*Application développée pour optimiser les tournées de livraison avec des algorithmes d'optimisation modernes et une interface utilisateur intuitive.*