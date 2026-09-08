import os
import json
from datetime import datetime

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NomaAI - Tomato Disease Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(APP_DIR)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_CANDIDATES = [
    os.path.join(
        APP_DIR,
        "models",
        "plant_disease_model_v3.keras"
    ),
    os.path.join(
        PARENT_DIR,
        "models",
        "plant_disease_model_v3.keras"
    ),
]


# ============================================================
# FOUNDER PHOTO PATHS
# ============================================================

FOUNDER_PHOTO_CANDIDATES = [

    os.path.join(APP_DIR, "picture", "founder.jpg"),
    os.path.join(APP_DIR, "picture", "founder.jpeg"),
    os.path.join(APP_DIR, "picture", "sani.jpeg"),

    os.path.join(APP_DIR, "founder.jpg"),
    os.path.join(APP_DIR, "founder.jpeg"),
    os.path.join(APP_DIR, "founder.png"),

    os.path.join(APP_DIR, "images", "founder.jpg"),
    os.path.join(APP_DIR, "images", "founder.jpeg"),
    os.path.join(APP_DIR, "images", "founder.png"),

    os.path.join(APP_DIR, "assets", "founder.jpg"),
    os.path.join(APP_DIR, "assets", "founder.jpeg"),
    os.path.join(APP_DIR, "assets", "founder.png"),

    os.path.join(PARENT_DIR, "picture", "founder.jpg"),
    os.path.join(PARENT_DIR, "picture", "founder.jpeg"),
    os.path.join(PARENT_DIR, "picture", "sani.jpeg"),

    os.path.join(PARENT_DIR, "founder.jpg"),
    os.path.join(PARENT_DIR, "founder.jpeg"),
    os.path.join(PARENT_DIR, "founder.png"),

    os.path.join(PARENT_DIR, "images", "founder.jpg"),
    os.path.join(PARENT_DIR, "images", "founder.jpeg"),
    os.path.join(PARENT_DIR, "images", "founder.png"),

    os.path.join(PARENT_DIR, "assets", "founder.jpg"),
    os.path.join(PARENT_DIR, "assets", "founder.jpeg"),
    os.path.join(PARENT_DIR, "assets", "founder.png"),
]


# ============================================================
# HISTORY PATHS
# ============================================================

HISTORY_DIR_CANDIDATES = [

    os.path.join(APP_DIR, "data"),
    os.path.join(APP_DIR, "history"),
    os.path.join(PARENT_DIR, "data"),
]

HISTORY_FILE_NAME = "prediction_history.json"


# ============================================================
# MODEL CLASSES
# ============================================================

CLASS_NAMES = [

    "Tomato___Bacterial_spot",

    "Tomato___Early_blight",

    "Tomato___Late_blight",

    "Tomato___Leaf_Mold",

    "Tomato___Septoria_leaf_spot",

    "Tomato___Spider_mites Two-spotted_spider_mite",

    "Tomato___Target_Spot",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",

    "Tomato___Tomato_mosaic_virus",

    "Tomato___healthy",
]


# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {

    "Tomato___Bacterial_spot":
        "Bacterial Spot",

    "Tomato___Early_blight":
        "Early Blight",

    "Tomato___Late_blight":
        "Late Blight",

    "Tomato___Leaf_Mold":
        "Leaf Mold",

    "Tomato___Septoria_leaf_spot":
        "Septoria Leaf Spot",

    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Spider Mites",

    "Tomato___Target_Spot":
        "Target Spot",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Tomato Yellow Leaf Curl Virus",

    "Tomato___Tomato_mosaic_virus":
        "Tomato Mosaic Virus",

    "Tomato___healthy":
        "Healthy Tomato Leaf",
}


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Tomato___Bacterial_spot": {

        "description":
            "Bacterial Spot is a bacterial disease that can cause "
            "small dark lesions on tomato leaves and fruit.",

        "advice":
            "Remove heavily affected leaves, avoid overhead "
            "watering, and maintain good airflow around plants."
    },


    "Tomato___Early_blight": {

        "description":
            "Early Blight is a fungal disease that commonly "
            "produces dark circular spots with concentric rings "
            "on older leaves.",

        "advice":
            "Remove infected leaves, avoid watering foliage, "
            "improve airflow, and use appropriate fungicide "
            "when necessary."
    },


    "Tomato___Late_blight": {

        "description":
            "Late Blight is a serious disease that can rapidly "
            "damage tomato leaves, stems, and fruits.",

        "advice":
            "Remove severely infected plant material, improve "
            "airflow, avoid prolonged leaf wetness, and seek "
            "appropriate treatment quickly."
    },


    "Tomato___Leaf_Mold": {

        "description":
            "Leaf Mold is a fungal disease that commonly develops "
            "under warm and humid conditions.",

        "advice":
            "Improve ventilation, reduce humidity, avoid wetting "
            "leaves, and remove affected leaves."
    },


    "Tomato___Septoria_leaf_spot": {

        "description":
            "Septoria Leaf Spot produces many small circular spots "
            "on tomato leaves, often with dark borders.",

        "advice":
            "Remove infected leaves, keep foliage dry, clear plant "
            "debris, and improve air circulation."
    },


    "Tomato___Spider_mites Two-spotted_spider_mite": {

        "description":
            "Spider Mites are tiny pests that feed on plant tissue "
            "and may cause yellowing, speckling, and leaf damage.",

        "advice":
            "Inspect the underside of leaves, wash plants carefully, "
            "improve plant health, and use appropriate pest control."
    },


    "Tomato___Target_Spot": {

        "description":
            "Target Spot is a fungal disease that produces brown "
            "lesions and can cause leaf drop.",

        "advice":
            "Remove infected leaves, reduce leaf wetness, improve "
            "airflow, and use appropriate disease management."
    },


    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {

        "description":
            "Tomato Yellow Leaf Curl Virus can cause yellowing, "
            "curling, stunted growth, and reduced fruit production.",

        "advice":
            "Control whiteflies, remove severely infected plants "
            "when appropriate, and use healthy planting material."
    },


    "Tomato___Tomato_mosaic_virus": {

        "description":
            "Tomato Mosaic Virus can cause mottled or mosaic "
            "patterns on leaves and may reduce plant growth "
            "and production.",

        "advice":
            "Remove infected plants when necessary, sanitize tools, "
            "and avoid spreading plant sap between plants."
    },


    "Tomato___healthy": {

        "description":
            "The model detected a tomato leaf that appears healthy.",

        "advice":
            "Continue good agricultural practices, monitor the "
            "plant regularly, and maintain proper watering "
            "and nutrition."
    },
}


# ============================================================
# FIND MODEL
# ============================================================

MODEL_PATH = None

for path in MODEL_CANDIDATES:

    if os.path.isfile(path):

        MODEL_PATH = path
        break


# ============================================================
# FIND FOUNDER PHOTO
# ============================================================

FOUNDER_PHOTO_PATH = None

for path in FOUNDER_PHOTO_CANDIDATES:

    if os.path.isfile(path):

        FOUNDER_PHOTO_PATH = path
        break


# ============================================================
# FIND HISTORY FILE
# ============================================================

HISTORY_FILE = None

for directory in HISTORY_DIR_CANDIDATES:

    if os.path.isdir(directory):

        HISTORY_FILE = os.path.join(
            directory,
            HISTORY_FILE_NAME
        )

        break


if HISTORY_FILE is None:

    HISTORY_FILE = os.path.join(
        APP_DIR,
        HISTORY_FILE_NAME
    )


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "Disease Detector"


# ============================================================
# HISTORY FUNCTIONS
# ============================================================

def load_history():

    """Load saved prediction history."""

    if not os.path.isfile(HISTORY_FILE):

        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, list):

            return data

        return []

    except Exception:

        return []


def save_history(record):

    """Save a prediction record."""

    try:

        history_directory = os.path.dirname(
            HISTORY_FILE
        )

        if history_directory:

            os.makedirs(
                history_directory,
                exist_ok=True
            )

        history = load_history()

        history.insert(
            0,
            record
        )

        # Keep latest 100 predictions
        history = history[:100]

        with open(
            HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history,
                file,
                indent=4
            )

        return True

    except Exception:

        return False


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(image):

    """Perform a basic image quality check."""

    try:

        width, height = image.size

        if width < 100 or height < 100:

            return (
                False,
                "Image is too small. Please upload a clearer image."
            )

        if width > 10000 or height > 10000:

            return (
                False,
                "Image is unusually large. Please use a normal photo."
            )

        image_array = np.array(
            image.convert("RGB")
        )

        if image_array.size == 0:

            return (
                False,
                "The image could not be read."
            )

        brightness = np.mean(
            image_array
        )

        if brightness < 15:

            return (
                False,
                "Image is too dark. Please upload a brighter photo."
            )

        if brightness > 245:

            return (
                False,
                "Image is too bright. Please upload a clearer photo."
            )

        return (
            True,
            "Image quality appears suitable for analysis."
        )

    except Exception:

        return (
            False,
            "Unable to check image quality."
        )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_nomaai_model(model_path):

    """Load the trained NomaAI model."""

    if model_path is None:

        return None

    try:

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        return model

    except Exception as error:

        st.error(
            f"Unable to load NomaAI model: {error}"
        )

        return None


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(model, image):

    """Predict tomato disease from uploaded image."""

    try:

        # ----------------------------------------------------
        # NomaAI V3 expects 128 x 128 RGB images
        # ----------------------------------------------------

        processed_image = (
            image
            .convert("RGB")
            .resize((128, 128))
        )

        image_array = np.array(
            processed_image,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ----------------------------------------------------
        # Keep the existing V3 preprocessing.
        # Do NOT add /255 here because the current V3 model
        # has already been tested successfully with this setup.
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )

        predictions = np.asarray(
            predictions
        )

        if predictions.ndim > 1:

            predictions = predictions[0]

        # ----------------------------------------------------
        # Verify model output
        # ----------------------------------------------------

        if len(predictions) != len(CLASS_NAMES):

            return (
                None,
                None,
                None
            )

        # ----------------------------------------------------
        # Convert logits to probabilities if necessary
        # ----------------------------------------------------

        probability_sum = np.sum(
            predictions
        )

        if (
            np.min(predictions) < 0
            or np.max(predictions) > 1
            or abs(probability_sum - 1.0) > 0.05
        ):

            predictions = (
                tf.nn.softmax(
                    predictions
                ).numpy()
            )

        # ----------------------------------------------------
        # Get highest prediction
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = (
            CLASS_NAMES[predicted_index]
        )

        confidence = (
            float(
                predictions[predicted_index]
            ) * 100
        )

        # ----------------------------------------------------
        # Top 3 predictions
        # ----------------------------------------------------

        top_indices = np.argsort(
            predictions
        )[::-1][:3]

        top_predictions = []

        for index in top_indices:

            class_name = CLASS_NAMES[
                int(index)
            ]

            top_predictions.append(
                {
                    "class": class_name,

                    "display_name":
                        DISPLAY_NAMES.get(
                            class_name,
                            class_name
                        ),

                    "confidence":
                        float(
                            predictions[int(index)]
                        ) * 100
                }
            )

        return (
            predicted_class,
            confidence,
            top_predictions
        )

    except Exception:

        return (
            None,
            None,
            None
        )


# ============================================================
# LOAD MODEL
# ============================================================

model = None

if MODEL_PATH is not None:

    model = load_nomaai_model(
        MODEL_PATH
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌱 NomaAI")

    st.caption(
        "AI-assisted tomato plant disease detection"
    )

    st.divider()

    st.subheader("📖 How to use")

    st.write(
        "1. Upload a tomato leaf image."
    )

    st.write(
        "2. Check the image quality."
    )

    st.write(
        "3. Click **ANALYZE WITH NOMAAI**."
    )

    st.write(
        "4. Review the AI prediction."
    )

    st.divider()

    st.subheader("🧠 AI Model")

    st.write(
        "**Model:** NomaAI V3"
    )

    st.write(
        "**Classes:** 10"
    )

    st.write(
        "**Input:** 128 × 128 pixels"
    )

    st.divider()

    st.subheader("👨‍💻 Founder")

    st.write(
        "**Muhammad Sani**"
    )

    st.caption(
        "Founder & Developer of NomaAI"
    )

    st.divider()

    navigation_options = [
        "Disease Detector",
        "Prediction History",
        "About NomaAI",
        "Model Information"
    ]

    page = st.radio(
        "Navigation",
        navigation_options,
        index=navigation_options.index(
            st.session_state.page
        )
    )

    st.session_state.page = page


# ============================================================
# MAIN HEADER
# ============================================================

header_col1, header_col2 = st.columns(
    [1, 4]
)

with header_col1:

    st.markdown(
        "# 🌱"
    )

with header_col2:

    st.title(
        "NomaAI"
    )

    st.caption(
        "AI-Powered Tomato Plant Disease Detection System"
    )

st.info(
    "Upload a clear tomato leaf image and let NomaAI "
    "analyze possible disease patterns using Artificial Intelligence."
)

st.divider()


# ============================================================
# FOUNDER SECTION
# ============================================================

if page == "Disease Detector":

    st.subheader(
        "👨‍💻 Meet the Founder"
    )

    founder_col1, founder_col2, founder_col3 = st.columns(
        [1, 1.2, 1]
    )

    with founder_col2:

        if FOUNDER_PHOTO_PATH is not None:

            try:

                founder_image = Image.open(
                    FOUNDER_PHOTO_PATH
                ).convert("RGB")

                st.image(
                    founder_image,
                    width=350
                )

                st.caption(
                    "Muhammad Sani — Founder & Developer of NomaAI"
                )

            except Exception:

                st.warning(
                    "Founder photo could not be displayed."
                )

        else:

            st.info(
                "Founder photo not found."
            )

            st.caption(
                "Place your photo at: picture/founder.jpg"
            )

    st.divider()


# ============================================================
# DISEASE DETECTOR PAGE
# ============================================================

if page == "Disease Detector":

    st.header(
        "🍅 Tomato Leaf Disease Detector"
    )

    # --------------------------------------------------------
    # MODEL CHECK
    # --------------------------------------------------------

    if MODEL_PATH is None:

        st.error(
            "NomaAI model was not found."
        )

        st.write(
            "Expected model location:"
        )

        st.code(
            os.path.join(
                "models",
                "plant_disease_model_v3.keras"
            )
        )

        st.stop()

    if model is None:

        st.error(
            "NomaAI model could not be loaded."
        )

        st.stop()

    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    st.write(
        "Upload a clear photo of a tomato leaf. "
        "NomaAI will analyze the image and provide its prediction."
    )

    # --------------------------------------------------------
    # UPLOAD AREA
    # --------------------------------------------------------

    st.subheader(
        "📤 Upload Tomato Leaf"
    )

    st.write(
        "Choose a clear photo of a tomato leaf for NomaAI to analyze."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        help=(
            "For best results, use a clear and well-lit "
            "tomato leaf photo."
        )
    )

    if uploaded_file is None:

        st.info(
            "👆 Please upload a tomato leaf image to begin the analysis."
        )

    # --------------------------------------------------------
    # IMAGE PROCESSING
    # --------------------------------------------------------

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception:

            st.error(
                "The uploaded file could not be opened."
            )

            st.stop()

        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        st.subheader(
            "📷 Uploaded Image"
        )

        image_col1, image_col2, image_col3 = st.columns(
            [1, 2, 1]
        )

        with image_col2:

            st.image(
                image,
                width=500
            )

            st.caption(
                "Selected tomato leaf"
            )

        st.divider()

        # ----------------------------------------------------
        # IMAGE QUALITY
        # ----------------------------------------------------

        st.subheader(
            "🔎 Image Quality"
        )

        quality_ok, quality_message = (
            check_image_quality(image)
        )

        if quality_ok:

            st.success(
                f"✅ {quality_message}"
            )

        else:

            st.warning(
                f"⚠️ {quality_message}"
            )

        st.divider()

        # ----------------------------------------------------
        # ANALYZE BUTTON
        # ----------------------------------------------------

        analyze_button = st.button(
            "🧠 ANALYZE WITH NOMAAI",
            type="primary",
            use_container_width=True
        )

        if analyze_button:

            if not quality_ok:

                st.warning(
                    "Please upload a clearer image before analysis."
                )

            else:

                # ------------------------------------------------
                # MODEL PREDICTION
                # ------------------------------------------------

                with st.spinner(
                    "NomaAI is analyzing the tomato leaf..."
                ):

                    (
                        predicted_class,
                        confidence,
                        top_predictions
                    ) = predict_disease(
                        model,
                        image
                    )

                # ------------------------------------------------
                # PREDICTION FAILURE
                # ------------------------------------------------

                if predicted_class is None:

                    st.error(
                        "NomaAI could not complete the prediction."
                    )

                    st.info(
                        "Please try another clear tomato leaf image."
                    )

                # ------------------------------------------------
                # SUCCESSFUL PREDICTION
                # ------------------------------------------------

                else:

                    display_name = (
                        DISPLAY_NAMES.get(
                            predicted_class,
                            predicted_class
                        )
                    )

                    disease_info = (
                        DISEASE_INFO.get(
                            predicted_class,
                            {
                                "description":
                                    "No additional information "
                                    "is available.",

                                "advice":
                                    "Please consult an "
                                    "agricultural expert."
                            }
                        )
                    )

                    st.success(
                        "✅ Analysis completed!"
                    )

                    st.divider()

                    # --------------------------------------------
                    # MAIN RESULT
                    # --------------------------------------------

                    st.header(
                        "🌿 NomaAI Result"
                    )

                    result_col1, result_col2 = st.columns(
                        2
                    )

                    with result_col1:

                        st.subheader(
                            "Detected condition:"
                        )

                        st.markdown(
                            f"### {display_name}"
                        )

                    with result_col2:

                        st.subheader(
                            "Confidence:"
                        )

                        st.markdown(
                            f"### {confidence:.2f}%"
                        )

                    st.progress(
                        min(
                            max(
                                confidence / 100,
                                0.0
                            ),
                            1.0
                        )
                    )

                    # --------------------------------------------
                    # LOW CONFIDENCE SAFETY MESSAGE
                    # --------------------------------------------

                    if confidence < 70:

                        st.warning(
                            "⚠️ NomaAI has low confidence in this "
                            "prediction. Please try a clearer image "
                            "with the tomato leaf clearly visible."
                        )

                    elif confidence < 90:

                        st.info(
                            "ℹ️ NomaAI has moderate confidence in "
                            "this prediction. Consider checking "
                            "another clear image or consulting an "
                            "agricultural professional."
                        )

                    st.divider()

                    # --------------------------------------------
                    # TOP 3
                    # --------------------------------------------

                    st.subheader(
                        "📊 Top 3 Predictions"
                    )

                    for prediction in top_predictions:

                        prediction_confidence = float(
                            prediction.get(
                                "confidence",
                                0
                            )
                        )

                        st.write(
                            f"**{prediction['display_name']}** "
                            f"— {prediction_confidence:.2f}%"
                        )

                        st.progress(
                            min(
                                max(
                                    prediction_confidence / 100,
                                    0.0
                                ),
                                1.0
                            )
                        )

                    st.divider()

                    # --------------------------------------------
                    # DISEASE INFORMATION
                    # --------------------------------------------

                    st.subheader(
                        "ℹ️ About this condition"
                    )

                    st.write(
                        disease_info["description"]
                    )

                    st.subheader(
                        "🌱 Recommended action"
                    )

                    st.write(
                        disease_info["advice"]
                    )

                    st.divider()

                    # --------------------------------------------
                    # SAVE HISTORY
                    # --------------------------------------------

                    record = {

                        "date":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "filename":
                            uploaded_file.name,

                        "prediction":
                            predicted_class,

                        "display_name":
                            display_name,

                        "confidence":
                            round(
                                confidence,
                                2
                            ),

                        "top_predictions":
                            top_predictions
                    }

                    if save_history(record):

                        st.info(
                            "📝 Prediction saved to history."
                        )

                    else:

                        st.warning(
                            "Prediction completed, but history "
                            "could not be saved."
                        )

                    # --------------------------------------------
                    # DISCLAIMER
                    # --------------------------------------------

                    st.warning(
                        "⚠️ Disclaimer: NomaAI provides an "
                        "AI-based prediction and should not "
                        "replace professional agricultural "
                        "diagnosis."
                    )


# ============================================================
# PREDICTION HISTORY PAGE
# ============================================================

elif page == "Prediction History":

    st.header(
        "📋 Prediction History"
    )

    history = load_history()

    if not history:

        st.info(
            "No predictions have been saved yet."
        )

    else:

        st.write(
            f"Total predictions: **{len(history)}**"
        )

        st.divider()

        for number, record in enumerate(
            history,
            start=1
        ):

            prediction_name = record.get(
                "display_name",
                record.get(
                    "prediction",
                    "Unknown"
                )
            )

            try:

                confidence = float(
                    record.get(
                        "confidence",
                        0
                    )
                )

            except Exception:

                confidence = 0.0

            date = record.get(
                "date",
                "Unknown date"
            )

            filename = record.get(
                "filename",
                "Unknown file"
            )

            with st.expander(
                f"{number}. {prediction_name} — "
                f"{confidence:.2f}%"
            ):

                st.write(
                    f"**Date:** {date}"
                )

                st.write(
                    f"**Image:** {filename}"
                )

                st.write(
                    f"**Prediction:** {prediction_name}"
                )

                st.write(
                    f"**Confidence:** {confidence:.2f}%"
                )

                top_predictions = record.get(
                    "top_predictions",
                    []
                )

                if top_predictions:

                    st.write(
                        "**Top predictions:**"
                    )

                    for item in top_predictions:

                        try:

                            item_confidence = float(
                                item.get(
                                    "confidence",
                                    0
                                )
                            )

                        except Exception:

                            item_confidence = 0.0

                        item_name = item.get(
                            "display_name",
                            item.get(
                                "class",
                                "Unknown"
                            )
                        )

                        st.write(
                            f"- {item_name}: "
                            f"{item_confidence:.2f}%"
                        )


# ============================================================
# ABOUT NOMAAI PAGE
# ============================================================

elif page == "About NomaAI":

    st.header(
        "🌱 About NomaAI"
    )

    st.subheader(
        "AI-assisted tomato plant disease detection"
    )

    st.write(
        """
        NomaAI is an artificial intelligence project designed
        to assist users in identifying possible tomato plant
        diseases from leaf images.
        """
    )

    st.write(
        """
        The system uses a trained deep-learning image
        classification model to analyze uploaded tomato leaf
        images and provide a predicted condition together with
        a confidence score.
        """
    )

    st.divider()

    st.subheader(
        "🎯 Project Goal"
    )

    st.write(
        """
        The goal of NomaAI is to make AI-assisted plant disease
        detection more accessible and easier to use, especially
        for farmers, students, agricultural learners, and other
        users interested in tomato plant health.
        """
    )

    st.divider()

    st.subheader(
        "👨‍💻 Founder"
    )

    st.write(
        "**Muhammad Sani**"
    )

    st.write(
        "Founder & Developer of NomaAI"
    )

    if FOUNDER_PHOTO_PATH is not None:

        try:

            founder_image = Image.open(
                FOUNDER_PHOTO_PATH
            ).convert("RGB")

            st.image(
                founder_image,
                width=300
            )

        except Exception:

            pass

    st.divider()

    st.subheader(
        "⚠️ Important"
    )

    st.write(
        """
        NomaAI is an AI-assisted tool. Its prediction should
        not be considered a final professional agricultural
        diagnosis. Users should seek advice from qualified
        agricultural professionals when necessary.
        """
    )


# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

elif page == "Model Information":

    st.header(
        "🧠 NomaAI Model Information"
    )

    st.subheader(
        "Model"
    )

    st.write(
        "**NomaAI V3**"
    )

    st.divider()

    st.subheader(
        "Model Specifications"
    )

    specification_col1, specification_col2 = st.columns(
        2
    )

    with specification_col1:

        st.write(
            "**Number of classes:** 10"
        )

        st.write(
            "**Input image:** 128 × 128 pixels"
        )

        st.write(
            "**Task:** Tomato leaf image classification"
        )

    with specification_col2:

        st.write(
            "**Framework:** TensorFlow / Keras"
        )

        if MODEL_PATH is not None:

            st.write(
                "**Model status:** ✅ Loaded"
            )

        else:

            st.write(
                "**Model status:** ❌ Not found"
            )

    st.divider()

    st.subheader(
        "🍅 Supported Classes"
    )

    for number, class_name in enumerate(
        CLASS_NAMES,
        start=1
    ):

        st.write(
            f"{number}. "
            f"{DISPLAY_NAMES.get(class_name, class_name)}"
        )

    st.divider()

    st.subheader(
        "📁 Model File"
    )

    if MODEL_PATH is not None:

        st.success(
            "NomaAI V3 model found successfully."
        )

        st.code(
            MODEL_PATH
        )

    else:

        st.error(
            "NomaAI V3 model file was not found."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌱 NomaAI — AI-assisted tomato plant disease detection"
)

st.caption(
    "Founder & Developer: Muhammad Sani"
)