import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

st.markdown("""
<style>
body {
    color: #fff;
    background-color: #4f8bf9;
}
</style>
    """, unsafe_allow_html=True)


st.title(" Prédictions du risque d'AVC d'un patient.")

# Chargement des données
data = pd.read_csv("./healthcare-dataset-stroke-data.csv")

data = data.drop('id', axis=1)
# Sample a subset of the data
data = data.sample(frac=0.1)

# Conversion de la colonne 'hypertension' 
data['hypertension'] = pd.to_numeric(data['hypertension'], errors='coerce')

# Conversion de la colonne 'Residence_type' 
data['Residence_type'] = data['Residence_type'].replace({'Urban': 1, 'Rural': 0})


# Choix de l'option
option = st.sidebar.selectbox("Choisissez une option", ["Exploration de données", "Représentations graphiques", "Modèles de machine learning"])

columns_temp = ['gender', 'ever_married', 'work_type', 'smoking_status', 'Residence_type']

for column in columns_temp:
    unique_values = data[column].unique()
    num_unique_values = len(unique_values)

    # Create a mapping dictionary for converting unique values to integer representations
    mapping = {value: index for index, value in enumerate(unique_values)}

    # Replace the categorical values with integer representations
    data[column] = data[column].map(mapping).astype(int)


if option == "Exploration de données":
    
    show_dataset = st.checkbox("Afficher le dataset complet")

    if show_dataset:
        st.write(data)
    else:
        start_row = st.number_input("Ligne de départ", min_value=0, max_value=len(data)-1, value=0)
        end_row = st.number_input("Ligne de fin", min_value=start_row, max_value=len(data)-1, value=len(data)-1)
        st.write(data.iloc[start_row:end_row+1])

    # Afficher les colonnes choisies par l'utilisateur
    selected_columns = st.multiselect("Choisissez les colonnes à afficher", data.columns)
    if selected_columns:
        st.write(data[selected_columns])

    # Create a multiple select to display variable names and descriptions
    variable_descriptions = {
          "gender": "Male, Female or Other",
          "age": "Age of the patient",
          "hypertension": "0 if the patient doesn't have hypertension, 1 if the patient has hypertension",
          "heart_disease": "0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease",
          "ever_married": "No or Yes",
          "work_type": "children, Govt_jov, Never_worked, Private or Self-employed",
          "Residence_type": "Rural or Urban",
          "avg_glucose_level": "Average glucose level in blood",
          "bmi": "Body mass index",
          "smoking_status": "'formerly smoked', 'never smoked', 'smokes', or 'Unknown'",
          "stroke": "1 if the patient had a stroke or 0 if not"
     }

    selected_variables = st.multiselect("Select variables", list(variable_descriptions.keys()))

    for variable in selected_variables:
     st.write(f"{variable}: {variable_descriptions[variable]}")

    # Summary des données
    st.subheader("Résumé des données")
    st.write(data.describe())

elif option == "Représentations graphiques":
    model_option = st.selectbox("Choisissez un modèle", ["SVM", "KNN", "RandomForest", "Régression logistique"])
    
    # Choix des représentations graphiques
    plot_option = st.selectbox("Choisissez une représentation graphique", ["Matrice de corrélations des variables quantitatives", "Pie plot and Count plot", "Distribution plot", "Target bar plot"])

    if plot_option == "Matrice de corrélations des variables quantitatives":
        # Matrice de corrélations
        numeric_columns = data.select_dtypes(include=["int", "float"]).columns
        correlation_matrix = data[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

    elif plot_option == "Pie plot and Count plot":
        # Choix des variables pour le Pie plot and Count plot
        selected_columns = st.multiselect("Choisissez les variables pour le Pie plot and Count plot", data.columns)

        if selected_columns:
            pie_or_count = st.selectbox("Do you want a Pie plot or a Count plot?", ["Pie plot", "Count plot"])
            for column in selected_columns:
                #st.subheader(f"Variable: {column}")
                fig, ax = plt.subplots(figsize=(12, 10))
                if pie_or_count == "Pie plot":
                    data[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                else:  # Count plot
                    sns.countplot(x=data[column], ax=ax)
                st.pyplot(fig)

    elif plot_option == "Distribution plot":
        # Choix des variables pour le Distribution plot
        selected_columns = st.multiselect("Choisissez les variables pour le Distribution plot", data.columns)

        if selected_columns:
            for column in selected_columns:
                #st.subheader(f"Variable: {column}")
                fig, ax = plt.subplots()
                sns.distplot(data[column], ax=ax)
                st.pyplot(fig)

    elif plot_option == "Target bar plot":
        # Choix de la variable pour le Target bar plot
        target_variable = st.selectbox("Choisissez une variable pour le Target bar plot", data.columns)

        if target_variable:
            #st.subheader("Target bar plot")
            fig, ax = plt.subplots()
            sns.countplot(x="stroke", hue=target_variable, data=data, ax=ax)  # Assuming 'stroke' is the target variable
            st.pyplot(fig)

    

elif option == "Modèles de machine learning":
    # Split the data into training and testing sets
    model_option = st.selectbox("Choisissez un modèle", ["SVM", "KNN", "RandomForest", "Régression logistique"])
    
    X_train, X_test, y_train, y_test = train_test_split(data.drop("stroke", axis=1), data["stroke"], test_size=0.2, random_state=42)
    
    if model_option == "SVM":
        # Paramètres clés à choisir avec un curseur
        C = st.slider("Paramètre C", min_value=0.1, max_value=10.0, value=1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])

        # Construction du modèle SVM avec les paramètres choisis
        svm_model = SVC(C=C, kernel=kernel, probability=True)
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # Creating an imputer that replaces NaN values with the mean value of the column
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        svm_model.fit(X_train, y_train)

        # Prédiction sur les données de test
        y_pred = svm_model.predict(X_test)

        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)

        # Affichage des métriques
        st.write("Accuracy:", accuracy)
        st.write("F1-score:", f1)
        st.write("Précision:", precision)

        # Matrice de confusion
        st.subheader("Matrice de confusion")
        
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()

        
        # Affichage de la matrice de confusion avec Seaborn
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        plt.ylabel('Vrai label')
        plt.xlabel('Label prédit')
        st.pyplot(fig)

        

    
# Courbe ROC
        y_scores = svm_model.predict_proba(X_test)
        fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)

        st.subheader("Courbe ROC")
        fig1, ax1 = plt.subplots()
        ax1.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        ax1.legend(loc = 'lower right')
        ax1.plot([0, 1], [0, 1],'r--')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('Taux de vrais positifs')
        ax1.set_xlabel('Taux de faux positifs')
        st.pyplot(fig1)

    elif model_option == "KNN":
        # Paramètres clés à choisir avec un curseur
        n_neighbors = st.slider("Nombre de voisins", min_value=1, max_value=10, value=5)
        

        # Construction du modèle KNN avec le paramètre choisi
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        # Creating an imputer that replaces NaN values with the mean value of the column
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Applying the imputer to the training and testing sets
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)


        knn_model.fit(X_train, y_train)

        # Prédiction sur les données de test
        y_pred = knn_model.predict(X_test)

        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # Affichage des métriques
        st.write("Accuracy:", accuracy)
        st.write("F1-score:", f1)
        st.write("Précision:", precision, zero_division=1)

        # Matrice de confusion
        st.subheader("Matrice de confusion")
        
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()

        
        # Affichage de la matrice de confusion avec Seaborn
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        plt.ylabel('Vrai label')
        plt.xlabel('Label prédit')
        st.pyplot(fig)

        

    
# Courbe ROC
        y_scores = knn_model.predict_proba(X_test)
        fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)

        st.subheader("Courbe ROC")
        fig1, ax1 = plt.subplots()
        ax1.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        ax1.legend(loc = 'lower right')
        ax1.plot([0, 1], [0, 1],'r--')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('Taux de vrais positifs')
        ax1.set_xlabel('Taux de faux positifs')
        st.pyplot(fig1) 





    elif model_option == "Régression logistique":


        solver = st.selectbox("Solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
        
        # Construction du modèle de régression logistique
        logistic_model = LogisticRegression(solver=solver)
        # Creating an imputer that replaces NaN values with the mean value of the column
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Applying the imputer to the training and testing sets
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        logistic_model.fit(X_train, y_train)

        # Prédiction sur les données de test
        y_pred = logistic_model.predict(X_test)

        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # Affichage des métriques
        st.write("Accuracy:", accuracy)
        st.write("F1-score:", f1)
        st.write("Précision:", precision, zero_division=1)

       # Matrice de confusion
        st.subheader("Matrice de confusion")
        
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()

        
        # Affichage de la matrice de confusion avec Seaborn
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        plt.ylabel('Vrai label')
        plt.xlabel('Label prédit')
        st.pyplot(fig)

        

    
# Courbe ROC
        y_scores = logistic_model.predict_proba(X_test)
        fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)

        st.subheader("Courbe ROC")
        fig1, ax1 = plt.subplots()
        ax1.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        ax1.legend(loc = 'lower right')
        ax1.plot([0, 1], [0, 1],'r--')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('Taux de vrais positifs')
        ax1.set_xlabel('Taux de faux positifs')
        st.pyplot(fig1) 





    elif model_option == "RandomForest":
        n_estimators = st.slider("Number of trees", min_value=10, max_value=200, value=100)
        
        max_depth = st.slider("Maximum depth", min_value=1, max_value=20, value=10)

        # Construction du modèle Random Forest
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        rf_model.fit(X_train, y_train)

        # Prédiction sur les données de test
        y_pred = rf_model.predict(X_test)

        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # Affichage des métriques
        st.write("Accuracy:", accuracy)
        st.write("F1-score:", f1)
        st.write("Précision:", precision, zero_division=1)
    
    # Matrice de confusion
        st.subheader("Matrice de confusion")
        
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()

        
        # Affichage de la matrice de confusion avec Seaborn
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        plt.ylabel('Vrai label')
        plt.xlabel('Label prédit')
        st.pyplot(fig)

        


    
# Courbe ROC
        y_scores = rf_model.predict_proba(X_test)
        fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)

        st.subheader("Courbe ROC")
        fig1, ax1 = plt.subplots()
        ax1.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        ax1.legend(loc = 'lower right')
        ax1.plot([0, 1], [0, 1],'r--')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('Taux de vrais positifs')
        ax1.set_xlabel('Taux de faux positifs')
        st.pyplot(fig1)
