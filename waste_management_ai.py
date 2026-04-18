import pandas as pd
import waste_management_ml as ml
import waste_management_dl as dl


# ================= INTERACTIVE CLASSIFIER =================
def classify_interactive(ml_model, city_encoder, waste_encoder, toxicity_encoder):

    print("\n--- Interactive Waste Classifier ---")

    city = input(f"Enter City {list(city_encoder.classes_)}: ").strip()
    while city not in city_encoder.classes_:
        city = input("Invalid city. Enter again: ").strip()

    waste_type = input(f"Enter Waste Type {list(waste_encoder.classes_)}: ").strip()
    while waste_type not in waste_encoder.classes_:
        waste_type = input("Invalid waste type. Enter again: ").strip()

    toxicity = input(f"Enter Toxicity Level {list(toxicity_encoder.classes_)}: ").strip()
    while toxicity not in toxicity_encoder.classes_:
        toxicity = input("Invalid toxicity level. Enter again: ").strip()

    recyclable = input("Is it recyclable? (yes/no): ").lower()
    while recyclable not in ["yes", "no"]:
        recyclable = input("Enter yes or no only: ").lower()

    recyclable = True if recyclable == "yes" else False

    new_data = pd.DataFrame([{
        "waste_type": waste_type,
        "city": city,
        "toxicity_level": toxicity,
        "recyclability": recyclable
    }])

    new_data["waste_type_encoded"] = waste_encoder.transform(new_data["waste_type"])
    new_data["city_encoded"] = city_encoder.transform(new_data["city"])
    new_data["toxicity_level_encoded"] = toxicity_encoder.transform(new_data["toxicity_level"])

    features = new_data[["waste_type_encoded", "toxicity_level_encoded", "recyclability"]]

    prediction = ml_model.predict(features)[0]

    print("\n--- RESULT ---")
    print("Predicted Category:", prediction)
    print(ml.get_recommendation(prediction))


# ================= MAIN PROGRAM =================
if __name__ == "__main__":

    # Load CSV
    waste_data = pd.read_csv( r"C:\Users\Saloni\OneDrive\Documents\ai project\Waste_Dataset_With_City_10000.csv" )

    
    waste_data.rename(columns={
        "Waste_Type": "waste_type",
        "Toxicity": "toxicity_level",
        "Recyclable": "recyclability",
        "City": "city", "Category": "category"
    }, inplace=True)


    waste_data["recyclability"] = waste_data["recyclability"].map({
        "Yes": True,
        "No": False
                })

    if "category" not in waste_data.columns:
        def derive_category(row):
            if row["toxicity_level"] == "High":
                return "Hazardous"
            elif row["recyclability"]:
                return "Recyclable"
            else:
                return "Residual"
        waste_data["category"] = waste_data.apply(derive_category, axis=1)

    print("\nDataset Loaded Successfully")
    print(waste_data.head())

    # Machine Learning Classification
    ml_model, city_encoder, waste_encoder, toxicity_encoder, classified_df = ml.classify_waste(waste_data)

    # Deep Learning Optimization
    dl.optimize_handling(classified_df.copy())

    # Step 4: Interactive Prediction Choice
    print("\nChoose Prediction Method:")
    print("1. Manual Input")
    print("2. Image Recognition")
    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        classify_interactive(
            ml_model,
            city_encoder,
            waste_encoder,
            toxicity_encoder
        )
    elif choice == "2":
        img_path = input("Enter the full path to the image file: ").strip()
        img_info = dl.analyze_image(img_path)
        if img_info:
            print(f"\nAI Analysis: Type: {img_info['waste_type']}, Toxicity: {img_info['toxicity']}")
            # Use ML model to categorize based on image detection
            img_features = pd.DataFrame([{
                "waste_type_encoded": waste_encoder.transform([img_info['waste_type']])[0],
                "toxicity_level_encoded": toxicity_encoder.transform([img_info['toxicity'].capitalize()])[0],
                "recyclability": True if img_info['waste_type'] in ["plastic", "paper"] else False
            }])
            prediction = ml_model.predict(img_features)[0]
            print("Predicted Category:", prediction)
            print(ml.get_recommendation(prediction))