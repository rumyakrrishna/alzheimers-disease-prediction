# ==========================================================
# 🧠 Alzheimer Disease Risk Prediction - Clinical Streamlit App
# ==========================================================

# ---------------- IMPORT LIBRARIES -------------------------
import streamlit as st              # Streamlit for web UI
import joblib                       # Load trained model and metrics.pkl
import numpy as np                  # Create model input array
import io                           # Handle in-memory PDF file
from datetime import datetime       # Auto-generate report date & time

# PDF generation libraries (ReportLab)
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet



# ---------------- LOAD TRAINED MODEL -----------------------
# The model was trained in the notebook on 6 selected features
# It expects input in the exact same order and same number of features
model = joblib.load("best_alzheimer_model.pkl")



# ---------------- LOAD METRICS FROM PKL --------------------
# metrics.pkl contains a dictionary like:
# {"accuracy": 0.91, "precision": 0.89, "recall": 0.88}
#metrics = joblib.load("metrics.pkl")
import json

with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Extract individual metrics for display in UI
ACCURACY = metrics["accuracy"]
PRECISION = metrics["precision"]
RECALL = metrics["recall"]



# ---------------- APP TITLE -------------------------------
st.title("🧠 Alzheimer Disease Risk Prediction")

# Short system description for user
st.write("AI-based clinical decision support system for Alzheimer risk assessment.")



# ==========================================================
# 🔹 PATIENT IDENTIFICATION SECTION
# These values will be used in PDF and file name
# ==========================================================
st.subheader("Patient Information")

patient_id = st.text_input("Patient ID")      # Unique patient identifier
patient_name = st.text_input("Patient Name")  # Patient full name



# ==========================================================
# 🔹 MODEL INFORMATION (ONLY IN UI)
# ==========================================================
with st.expander("📊 Model Information"):

    # Show evaluation metrics (loaded from metrics.pkl)
    st.write(f"Accuracy: {ACCURACY:.2f}")
    st.write(f"Precision: {PRECISION:.2f}")
    st.write(f"Recall: {RECALL:.2f}")

    # Explain model configuration
    st.write("Model: Random Forest Classifier")
    st.write("Trained on 6 selected clinical features.")
    st.write("Top user inputs selected using Permutation Importance.")



# ==========================================================
# 🔹 SIDEBAR INPUTS (TOP IMPORTANT FEATURES ONLY)
# These are high-importance features for user entry
# ==========================================================
st.sidebar.header("Patient Clinical Details")

# Functional ability score (0–10)
functional = st.sidebar.slider("Functional Assessment", 0, 10, 5)

# Activities of Daily Living score (0–10)
adl = st.sidebar.slider("ADL Score", 0, 10, 5)

# Mini Mental State Examination score (0–30)
mmse = st.sidebar.slider("MMSE Score", 0, 30, 20)

# Binary symptom indicators (0 = No, 1 = Yes)
memory = st.sidebar.selectbox("Memory Complaints", [0, 1])
behavior = st.sidebar.selectbox("Behavioral Problems", [0, 1])

# DietQuality
diet = st.sidebar.slider("Diet Quality", 0, 10, 5)




# ==========================================================
# 🔹 CREATE MODEL INPUT ARRAY IN EXACT TRAINING ORDER
# ⚠️ Feature order must match training order
# Otherwise predictions will be incorrect
# ==========================================================
input_data = np.array([[
    functional,    # 1
    adl,           # 2
    mmse,          # 3
    memory,        # 4
    behavior,      # 5
    diet,          # 6
    ]])



# ==========================================================
# 🔹 PDF REPORT GENERATION FUNCTION
# Creates a structured clinical PDF with patient details
# ==========================================================
def create_pdf(prediction, probability, patient_id, patient_name):

    # Create in-memory buffer (no file saved to disk)
    buffer = io.BytesIO()

    # Define PDF document layout
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    content = []

    # ---------------- REPORT TITLE -------------------------
    content.append(Paragraph("Alzheimer Disease Risk Report", styles['Title']))
    content.append(Spacer(1, 12))

    # ---------------- PATIENT DETAILS ----------------------
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content.append(Paragraph(f"<b>Patient ID:</b> {patient_id}", styles['Normal']))
    content.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", styles['Normal']))
    content.append(Paragraph(f"<b>Report Generated:</b> {report_time}", styles['Normal']))

    content.append(Spacer(1, 12))

    # ---------------- RISK LEVEL + EMOJI --------------------
    if probability > 0.7:
        risk_label = "High Risk"
        emoji = "⚠️"
    elif probability > 0.4:
        risk_label = "Moderate Risk"
        emoji = "😐"
    else:
        risk_label = "Low Risk"
        emoji = "😊"

    # ---------------- PREDICTION SUMMARY -------------------
    content.append(Paragraph("<b>Prediction Summary</b>", styles['Heading2']))
    content.append(Spacer(1, 6))

    content.append(Paragraph(
        f"<b>Diagnosis:</b> {'Alzheimer Detected' if prediction==1 else 'No Alzheimer'}",
        styles['Normal']
    ))

    content.append(Paragraph(
        f"<b>Risk Probability:</b> {probability*100:.2f}%",
        styles['Normal']
    ))

    content.append(Paragraph(
        f"<b>Risk Level:</b> {risk_label} {emoji}",
        styles['Normal']
    ))

    content.append(Spacer(1, 12))

    # ---------------- CLINICAL INPUTS ----------------------
    content.append(Paragraph("<b>Patient Clinical Details</b>", styles['Heading2']))
    content.append(Spacer(1, 6))

    clinical_data = [
        f"Functional Assessment: {functional}",
        f"ADL Score: {adl}",
        f"MMSE Score: {mmse}",
        f"Memory Complaints: {memory}",
        f"Behavioral Problems: {behavior}",
        f"DietQuality: {diet}"
    ]

    for item in clinical_data:
        content.append(Paragraph(item, styles['Normal']))

    content.append(Spacer(1, 12))

    # ---------------- CLINICAL RECOMMENDATION --------------
    content.append(Paragraph("<b>Clinical Recommendation</b>", styles['Heading2']))
    content.append(Spacer(1, 6))

    if risk_label == "High Risk":
        note = "High risk detected. Recommend neurological evaluation and cognitive testing."
    elif risk_label == "Moderate Risk":
        note = "Moderate risk. Suggest periodic cognitive screening and lifestyle monitoring."
    else:
        note = "Low risk profile. Maintain healthy lifestyle and routine check-ups."

    content.append(Paragraph(note, styles['Normal']))

    content.append(Spacer(1, 24))

    # ---------------- DOCTOR SIGNATURE PLACEHOLDER ---------
    content.append(Paragraph("Doctor Signature: ____________________________", styles['Normal']))
    content.append(Paragraph("Name: ____________________________", styles['Normal']))
    content.append(Paragraph("Date: ____________________________", styles['Normal']))

    content.append(Spacer(1, 12))

    # ---------------- FOOTER DISCLAIMER --------------------
    content.append(Paragraph(
        "This report is generated by an AI-based clinical decision support system and should be reviewed by a qualified medical professional.",
        styles['Italic']
    ))

    # Build PDF document
    doc.build(content)

    # Reset buffer pointer to beginning
    buffer.seek(0)

    return buffer



# ==========================================================
# 🔹 PREDICTION BUTTON WITH VALIDATION
# ==========================================================
prediction = None
probability = None

if st.button("Predict Alzheimer Risk"):

    # Ensure patient identification is provided
    if not patient_id or not patient_name:
        st.warning("Please enter Patient ID and Patient Name.")
    else:
        # Generate prediction
        prediction = model.predict(input_data)[0]

        # Generate probability for Alzheimer class
        probability = model.predict_proba(input_data)[0][1]

        # Display diagnosis result
        if prediction == 1:
            st.error("🧠 Alzheimer Detected")
        else:
            st.success("✅ No Alzheimer Detected")

        # Display probability
        st.write(f"Risk Probability: {probability * 100:.2f}%")

        # Display risk category
        if probability > 0.7:
            st.error("High Risk ⚠️")
        elif probability > 0.4:
            st.warning("Moderate Risk 😐")
        else:
            st.success("Low Risk 😊")



# ==========================================================
# 🔹 PDF DOWNLOAD BUTTON (VISIBLE AFTER PREDICTION)
# ==========================================================
if prediction is not None and patient_id and patient_name:

    # Generate PDF report
    pdf = create_pdf(prediction, probability, patient_id, patient_name)

    # Create file name using patient ID and name
    file_name = f"{patient_id}_{patient_name}_report.pdf"

    # Streamlit download button
    st.download_button(
        label="📄 Download Patient Report (PDF)",
        data=pdf,
        file_name=file_name,
        mime="application/pdf"
    )
