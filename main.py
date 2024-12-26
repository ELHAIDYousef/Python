import streamlit as st
from PIL import Image
import pytesseract
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import cv2
import numpy as np
import os


# Set up the Streamlit page
st.set_page_config(page_title="Intelligent Document Management", layout="centered")

# Title and description
st.title("Intelligent Document Management System")
st.markdown("Upload a document (PDF or Image):")

# File uploader
uploaded_file = st.file_uploader("", type=["pdf", "png", "jpg", "jpeg"])

# Set Tesseract command path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\lenovo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


# Preprocessing function (Fonction de Prétraitement de l'Image)
# Convertit l'image en niveaux de gris.
# Applique une binarisation pour faciliter l'extraction du texte.
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Resize
    filtered = cv2.GaussianBlur(resized, (5, 5), 0)  # Filtering
    return filtered, image


# Text detection function (. Fonction de Détection de Texte)
#Charge le modèle EAST pour la détection de texte
def detect_text(image):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    east_model_path = os.path.join(current_dir, "frozen_east_text_detection.pb")

    if not os.path.exists(east_model_path):
        raise FileNotFoundError(f"Le fichier EAST model n'a pas été trouvé : {east_model_path}")

    net = cv2.dnn.readNet(east_model_path)
    height, width = image.shape[:2]
    new_width = (width // 32) * 32
    new_height = (height // 32) * 32
    image_resized = cv2.resize(image, (new_width, new_height))
#

    #Traitement et Extraction de Texte
    blob = cv2.dnn.blobFromImage(image_resized, 1.0, (new_width, new_height), (123.68, 116.78, 103.94), swapRB=True,
                                 crop=False)
    net.setInput(blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    boxes, confidences = [], []
    for y in range(scores.shape[2]):
        scores_data = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        angles_data = geometry[0][4][y]

        for x in range(scores.shape[3]):
            if scores_data[x] < 0.5:
                continue
            offset_x, offset_y = x * 4.0, y * 4.0
            angle = angles_data[x]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            p1 = (-sin_a * h + offset_x, -cos_a * h + offset_y)
            p3 = (-cos_a * w + offset_x, sin_a * w + offset_y)
            boxes.append([p1[0], p1[1], p3[0], p3[1]])
            confidences.append(float(scores_data[x]))

    return boxes, confidences
#

#Traitement du Fichier Téléchargé (Vérifie si un fichier a été téléchargé et affiche le nom du fichier.)
if uploaded_file:
    st.subheader("Uploaded File")
    st.write(f"**Filename:** {uploaded_file.name}")

    extracted_text = ""

    #Ouvre l'image téléchargée et l'affiche.
    #Appelle ensuite les fonctions de prétraitement et d'extraction de texte.
    if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Preprocess image
        preprocessed_image, original_image = preprocess_image(image_cv)

        # Detect text
        boxes, confidences = detect_text(original_image)  # Use original image for detection

        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(preprocessed_image)

    # Process PDF Files
    elif uploaded_file.type == "application/pdf":
        with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text() + "\n"

    # Display extracted text
    st.subheader("Extracted Text")
    st.text_area("Extracted Text", value=extracted_text, height=400)

    # Simple Classification (Keyword-Based)
    st.subheader("Document Classification")
    categories = {"Facture": ["invoice", "amount", "total"],
                  "cv": ["experience", "education", "skills"],
                  "Report": ["summary", "report", "analysis"]}

    classification = ""
    for category, keywords in categories.items():
        if any(keyword in extracted_text.lower() for keyword in keywords):
            classification = category
            break
    st.write(f"**Document Type:** {classification}")

    # Statistics Visualization
    #Compte le nombre de mots et crée un graphique de fréquence des mots les plus courants.
    st.subheader("Text Statistics")
    words = extracted_text.split()
    word_count = len(words)
    st.write(f"**Word Count:** {word_count}")

    # Word Frequency
    word_freq = pd.DataFrame(words, columns=["Word"]).value_counts().reset_index(name="Frequency")
    word_freq = word_freq.head(10)

    fig, ax = plt.subplots()
    ax.barh(word_freq["Word"], word_freq["Frequency"], color="skyblue")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    ax.set_title("Top 10 Words")
    st.pyplot(fig)

    # Data Export
    st.subheader("Export Data")

    # Create a PDF document
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add the extracted text to the PDF
    for line in extracted_text.splitlines():
        try:
            pdf.cell(200, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
        except Exception as e:
            st.error(f"Error encoding line: {e}")

    # Save the PDF to a bytes buffer
    pdf_output = pdf.output(dest='S').encode('latin-1', 'replace')

    st.download_button(
        label="Download Extracted Text as PDF",
        data=pdf_output,
        file_name="extracted_text.pdf",
        mime="application/pdf",
    )