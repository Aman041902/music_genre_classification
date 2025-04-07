import streamlit as st
from PIL import Image
import librosa
import tensorflow as tf
from tensorflow.image import resize
import numpy as np
import os




classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']




print(st.__version__)
print(tf.__version__)




def fetch_model():
    try:
        model = tf.keras.models.load_model('./train_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def preprocess_file(file_path,target_shape=(128,128)):
  data = []
  audio_data,sr = librosa.load(file_path,sr=None)
  duration = 4
  overlap_duration = 2
  samples = duration * sr
  overlap_sample = overlap_duration * sr
  total_samples = max(1, int(np.ceil((len(audio_data) - samples) / (samples - overlap_sample))) + 1)

  for i in range(total_samples):
                    start = i * (samples - overlap_sample)
                    end = start + samples
                    chunk = audio_data[start:end]

                    # Ignore segments that are too short
                    if len(chunk) < 512:
                        continue

                    # Dynamically adjust n_fft to avoid warnings
                    n_fft = min(2048, len(chunk) // 2)

                    spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr, n_fft=n_fft)
                    spectrogram = resize(np.expand_dims(spectrogram, axis=-1), target_shape)

                    data.append(spectrogram)

  return np.array(data)

def model_pred(x_test):
  model = fetch_model()

  if(model is None):
    st.error("Model not found")
    return
  y_pred = model.predict(x_test)
  pred_cat = np.argmax(y_pred,axis=1)
  unique_el,cnt = np.unique(pred_cat,return_counts=True)
  max_cnt = np.max(cnt)
  max_el = unique_el[cnt==max_cnt][0]
  return classes[max_el]




st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fonts, animations, and styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Poppins:wght@400;700&family=Montserrat:wght@400;700&display=swap');

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        color: #4CAF50;
    }

    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f2f6;
    }

    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-family: 'Montserrat', sans-serif;
    }

    .stMarkdown {
        font-family: 'Roboto', sans-serif;
    }

    .stImage img {
        border-radius: 15px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .fadeIn {
        animation: fadeIn 2s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)
# Custom CSS for Sidebar Styling
# Custom CSS for Glassmorphism Sidebar
st.markdown(
    """
    <style>
        /* Sidebar Background - Glassmorphism */
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.1);  /* Semi-transparent effect */
            backdrop-filter: blur(10px);  /* Blurred background */
            border-radius: 15px;
            padding: 20px;
        }

        /* Sidebar Title */
        [data-testid="stSidebar"] h1 {
            font-size: 26px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            font-family: 'Poppins', sans-serif;
        }

        /* Sidebar Navigation Buttons */
        .sidebar-option {
            font-size: 18px;
            padding: 12px;
            color: white;
            border-radius: 10px;
            transition: 0.3s;
            display: flex;
            align-items: center;
            font-family: 'Poppins', sans-serif;
        }

        /* Hover Effect */
        .sidebar-option:hover {
            background: #4CAF50;
            color: white;
            cursor: pointer;
        }

        /* Selected Option Styling */
        .selected {
            background: #4CAF50;
            color: white;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Title with an Emoji
st.sidebar.markdown("<h1>🎵 Music Genre Classifier</h1>", unsafe_allow_html=True)

# Custom Sidebar Buttons
app_modes = {
    "🏠 Home": "Home",
    "ℹ️ About": "About",
    "🎶 Prediction": "Prediction"
}

# Display Sidebar Options with Custom Styling
app_mode = st.sidebar.radio("", list(app_modes.keys()), format_func=lambda x: f" {x}")

# Convert back to string value
app_mode = app_modes[app_mode]


if app_mode == "Home":


    # Cu
    # Custom CSS for Improved Visibility
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&family=Roboto:wght@400;700&display=swap');

        h1, h2, h3 {
            font-family: 'Poppins', sans-serif;
            text-align: center;
            color: #4CAF50;  /* Green Theme */
        }

        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f0f2f5;  /* Light gray background */
        }

        .content-box {
            background-color: #ffffff;  /* Pure White Background */
            padding: 25px;  /* Increased Padding for Better Readability */
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);  /* Stronger Shadow */
            margin-bottom: 20px;
            color: #333333;  /* Darker Text for Visibility */
        }

        ol, ul {
            padding-left: 20px;
        }

        li:hover {
            color: #4CAF50;
            transition: 0.3s ease-in-out;
        }

        .cta-btn {
            display: block;
            width: fit-content;
            margin: 20px auto;
            padding: 12px 25px;
            background-color: #4CAF50;
            color: white;
            text-align: center;
            font-weight: bold;
            border-radius: 6px;
            text-decoration: none;
            font-size: 16px;
        }

        .cta-btn:hover {
            background-color: #388E3C;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .fadeIn {
            animation: fadeIn 1.5s ease-in-out;
        }
        </style>
        """, unsafe_allow_html=True)

    # Header with Animation
    st.markdown("<h1 class='fadeIn'>🎵 Music Genre Classification</h1>", unsafe_allow_html=True)

    # Introduction Section
    st.markdown("""
        <div class='content-box fadeIn'>
            <p style='text-align: justify; font-size: 16px;'>
            This app leverages a <b>Convolutional Neural Network (CNN)</b> to classify music genres based on audio features. Upload an audio file, and our AI system will predict the genre.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Display Image
    image = Image.open("./music_image.webp")  # Replace with actual image path
    st.image(image, caption="Discover the Power of AI in Music Analysis", use_column_width=True)

    # Main Content with Improved Layout
    with st.expander("📌 **Our Goal**", expanded=True):
        st.markdown("""
        <div class='content-box'>
            <p style='text-align: justify;'>
            Our goal is to efficiently identify music genres using AI. Upload an audio file, and our system will analyze it using advanced machine learning techniques.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("⚙️ **How It Works**"):
        st.markdown("""
        <div class='content-box'>
            <ol>
                <li><b>Upload Audio:</b> Go to the <b>Genre Classification</b> page and upload an audio file.</li>
                <li><b>Analysis:</b> Our system processes the audio and extracts key features.</li>
                <li><b>Results:</b> Get instant predictions with detailed insights.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("⭐ **Why Choose Us?**"):
        st.markdown("""
        <div class='content-box'>
            <ul>
                <li><b>High Accuracy:</b> Uses advanced deep learning models for precise genre classification.</li>
                <li><b>User-Friendly:</b> Intuitive and easy-to-use interface.</li>
                <li><b>Fast Processing:</b> Get real-time results for a seamless experience.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Call-to-Action Button
   

    # About Section
    st.markdown("""
        <h2>ℹ️ About Us</h2>
        <div class='content-box'>
            <p style='text-align: justify;'>
            Learn more about the project, our team, and our mission on the <b>About</b> page.
            </p>
        </div>
        """, unsafe_allow_html=True)


elif app_mode == "About":


    # Custom CSS for styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&family=Roboto:wght@400;700&display=swap');

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Poppins', sans-serif;
            color: #1DB954;  /* Spotify Green for headings */
        }

        body {
            font-family: 'Roboto', sans-serif;
            background-color: #F5F7FA;  /* Soft gray background */
        }

        .stMarkdown {
            font-family: 'Roboto', sans-serif;
            color: #333333;  /* Dark gray text */
        }

        .about-section {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            color: #333;
            transition: transform 0.3s;
        }

        .about-section:hover {
            transform: translateY(-5px);
        }

        .about-section h3 {
            color: #1DB954;
            font-size: 22px;
            margin-bottom: 10px;
        }

        .about-section p {
            font-size: 16px;
            line-height: 1.6;
        }

        .about-section ul {
            list-style-type: none;
            padding-left: 0;
        }

        .about-section ul li {
            font-size: 16px;
            margin-bottom: 8px;
            padding-left: 20px;
            position: relative;
        }

        .about-section ul li::before {
            content: "🎵";
            position: absolute;
            left: 0;
            color: #1DB954;  /* Green icon for list */
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fadeIn {
            animation: fadeIn 1.2s ease-in-out;
        }
        </style>
        """, unsafe_allow_html=True)

    # About Project Section
    st.markdown("<h1 class='fadeIn'>📌 About the Project</h1>", unsafe_allow_html=True)
    with st.container():
        st.markdown("""
            <div class='about-section fadeIn'>
                <h3>🎶 Music Genre Classification</h3>
                <p>
                    Music is an art that transcends language, and classifying different genres helps us understand patterns in sound. This project applies **Machine Learning** techniques to classify music genres based on **audio features** extracted from sound waves.
                </p>
                <p>
                    By using **Mel Spectrograms**, we visualize sound waves and train deep learning models to differentiate between genres. The dataset consists of various music styles, providing an excellent playground for **AI-based classification**.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Mel Spectrogram Image Section
    st.markdown("<h2 class='fadeIn'>📊 Mel Spectrogram Visualization</h2>", unsafe_allow_html=True)
    with st.container():
        st.markdown("""
            <div class='about-section fadeIn'>
                <p>Below is an example of a Mel Spectrogram, which is a visual representation of the audio signal's frequency content over time. This is the type of data our model uses to classify music genres:</p>
            </div>
            """, unsafe_allow_html=True)
        # Display Mel Spectrogram Image
        mel_image = Image.open("./mel-img.png")
        st.image(mel_image, caption="Example of a Mel Spectrogram", use_column_width=True)

    # About Dataset Section
    st.markdown("<h1 class='fadeIn'>📊 About the Dataset</h1>", unsafe_allow_html=True)
    with st.container():
        st.markdown("""
            <div class='about-section fadeIn'>
                <h3>🔍 Dataset Overview</h3>
                <ul>
                    <li><b>Genres:</b> 10 music genres (Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock).</li>
                    <li><b>Audio Files:</b> Each genre contains **100 tracks**, each **30 seconds long**.</li>
                    <li><b>Feature Extraction:</b> Audio files converted to **Mel Spectrograms** for neural network processing.</li>
                    <li><b>Enhanced Dataset:</b> 3-second audio segments are also available, increasing dataset size **10x**.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

 
# ... (your other functions: preprocess_file, model_pred, etc.) .

elif app_mode == "Prediction":
    st.markdown("## 🎶 **Music Genre Prediction**")
    st.write("Upload an MP3 file, and let the model predict its genre.")

    # Upload Section
    st.markdown("### 📂 Upload an Audio File")
    test_mp3 = st.file_uploader("", type=["mp3"])

    if test_mp3 is not None:
        # Save File
        filepath = os.path.join("Test_Music", test_mp3.name)
        with open(filepath, "wb") as f:
            f.write(test_mp3.getvalue())

        st.success("✅ **File uploaded successfully!**")

        # Create columns for buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("▶️ **Play Audio**"):
                st.audio(test_mp3, format="audio/mp3")

        with col2:
            if st.button("🔍 **Predict Genre**"):
                try:
                    with st.spinner("🎧 Analyzing audio... Please wait..."):
                        progress_bar = st.progress(0)
                        for percent in range(100):
                            progress_bar.progress(percent + 1)

                        X_test = preprocess_file(filepath)
                        music_class = model_pred(X_test)
                        progress_bar.empty()  # Remove progress bar

                        # Show result with balloons
                        st.balloons()
                        st.markdown(
                            f"## 🎼 **Prediction:** *It's a **:red[{music_class}]** music!*"
                        )

                except Exception as e:
                    st.error(f"⚠️ An error occurred: {e}")

    else:
        st.warning("⚠️ Please upload an MP3 file to make a prediction.")
