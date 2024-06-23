import customtkinter as ctk
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), "../model/model_joblib_diabetes")


def predict_diabetes():
    values = [float(entry.get()) for entry in entries]

    model = joblib.load(model_path)
    prediction = model.predict([values])

    predicted = "Non-Diabetic" if prediction == 0 else "Diabetic"
    result_label.configure(text=predicted)


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

master = ctk.CTk()
master.title("Diabetes Prediction")
master.geometry("500x500")

header_label = ctk.CTkLabel(
    master,
    text="Diabetes Prediction",
    corner_radius=6,
    padx=10,
    pady=10,
)
header_label.pack(fill="x")

labels = [
    "Pregnancies",
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI",
    "Diabetes Pedigree Function",
    "Age",
]
entries = []

for label_text in labels:
    frame = ctk.CTkFrame(master)
    frame.pack(padx=10, pady=5, fill="x")

    label = ctk.CTkLabel(
        frame,
        text=label_text,
        width=25,
        anchor="w",
    )
    label.pack(side="left")

    entry = ctk.CTkEntry(frame)
    entry.pack(side="right", padx=10, fill="x")

    entries.append(entry)

predict_button = ctk.CTkButton(
    master,
    text="Predict",
    command=predict_diabetes,
    text_color="white",
)
predict_button.pack(pady=20, ipadx=20, ipady=10)

result_label = ctk.CTkLabel(master, text="", pady=10)
result_label.pack()

master.mainloop()
