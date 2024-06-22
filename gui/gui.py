from tkinter import *
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), "../model/model_joblib_diabetes")


def predict_diabetes():
    values = [float(entry.get()) for entry in entries]

    model = joblib.load(model_path)
    prediction = model.predict([values])

    predicted = "Non-Diabetic" if prediction == 0 else "Diabetic"
    result_label.config(text=predicted)


master = Tk()
master.title("Diabetes Prediction")
master.geometry("500x500")
master.configure(bg="#f0f0f0")

font_label = ("Helvetica", 12)
font_entry = ("Helvetica", 12)
font_button = ("Helvetica", 12, "bold")

header_label = Label(
    master,
    text="Diabetes Prediction",
    font=("Helvetica", 16, "bold"),
    bg="#333",
    fg="white",
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
    frame = Frame(master, bg="#f0f0f0")
    frame.pack(padx=10, pady=5, fill="x")

    label = Label(
        frame, text=label_text, font=font_label, bg="#f0f0f0", width=25, anchor="w"
    )
    label.pack(side="left")

    entry = Entry(frame, font=font_entry, bg="white", relief="solid", bd=1)
    entry.pack(side="right", padx=10, fill="x")

    entries.append(entry)

predict_button = Button(
    master,
    text="Predict",
    font=font_button,
    bg="#4CAF50",
    fg="white",
    command=predict_diabetes,
)
predict_button.pack(pady=20, ipadx=20, ipady=10)

result_label = Label(master, text="", font=("Arial", 16, "bold"), bg="#f0f0f0", pady=10)
result_label.pack()

master.mainloop()
