import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# ================= MACHINE LEARNING MODEL =================
def classify_waste(df):

    # Encode categorical columns
    le_city = LabelEncoder()
    le_waste = LabelEncoder()
    le_toxicity = LabelEncoder()

    df["city_encoded"] = le_city.fit_transform(df["city"])
    df["waste_type_encoded"] = le_waste.fit_transform(df["waste_type"])
    df["toxicity_level_encoded"] = le_toxicity.fit_transform(df["toxicity_level"])

    # Features & Target
    features = [
        "waste_type_encoded",
        "toxicity_level_encoded",
        "recyclability"
    ]

    X = df[features]
    y = df["category"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Decision Tree Visualization
    plt.figure(figsize=(22, 10))
    plot_tree(
        model,
        feature_names=features,
        class_names=model.classes_,
        filled=True
    )
    plt.savefig("decision_tree.png")
    plt.close()

    print("Decision Tree saved as decision_tree.png")

    # RETURN EVERYTHING MAIN FILE NEEDS
    return model, le_city, le_waste, le_toxicity, df


# ================= RECOMMENDATION SYSTEM =================
def get_recommendation(prediction):

    if prediction == "Reusable":
        return "Recommendation: Clean and reuse the item to extend its lifecycle."

    elif prediction == "Recyclable":
        return "Recommendation: Send to an authorized recycling facility."

    elif prediction == "Hazardous":
        return "Recommendation: Dispose at a certified hazardous waste center."

    else:
        return "Recommendation: Consider energy recovery as a last option."