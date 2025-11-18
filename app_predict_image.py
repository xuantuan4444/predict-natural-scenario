import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

st.set_page_config(
    page_title="Predict Natural scenery",
    layout="wide"
)

st.title("Predict Natural scenery with Deep Learning")
st.markdown("---")

img_height, img_width = 224, 224
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

@st.cache_resource
def load_models():
    models = {}
    model_paths = {
        'VGG16': 'models/vgg16_best.h5',
        'DenseNet121': 'models/densenet121_best.h5',
        'MobileNetV2': 'models/mobilenet_best.h5'
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[name] = load_model(path)
                st.sidebar.success(f"{name} loaded")
            except Exception as e:
                st.sidebar.error(f"{name}: {str(e)}")
        else:
            st.sidebar.warning(f"{name}: File does not exist")
    
    return models


def preprocess_image(image):
    image = image.resize((img_height, img_width))
    
    img_array = np.array(image)
    if len(img_array.shape) == 2: 
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    # Normalize
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_with_model(model, image_array):
    start_time = time.time()
    predictions = model.predict(image_array, verbose=0)
    inference_time = time.time() - start_time  # Convert to seconds
    
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    return predicted_class, confidence, inference_time, predictions[0]

def plot_confidence_histogram(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    confidences = [results[m]['confidence'] for m in models]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax.bar(models, confidences, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    return fig

def plot_inference_time_histogram(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    times = [results[m]['inference_time'] for m in models]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax.bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Inference Time (s)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    return fig

def plot_class_probabilities(predictions, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    probabilities = predictions * 100
    colors = ['#FF6B6B' if i == np.argmax(predictions) else '#95a5a6' for i in range(len(class_names))]
    
    bars = ax.barh(class_names, probabilities, color=colors, alpha=0.8, edgecolor='black')
    
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{prob:.2f}%',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Class Probabilities - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    return fig


def main():
    st.sidebar.header("Setting")
    st.sidebar.header("Load Models")
    models = load_models()
    
    if not models:
        st.error("Don't have model to be loaded!")
        st.info("Please run file ipynb first")
        return
    
    st.sidebar.success(f"loaded {len(models)} models")
    
    st.header("Upload")
    uploaded_file = st.file_uploader(
        "Select a picture",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Drag and drop or click to choose file"
    )
    
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption='Picture uploaded', use_container_width=True)
        
        st.markdown("---")
        
        with st.spinner("Preprocessing..."):
            image_array = preprocess_image(image)
        
        # predict with all models
        results = {}
        with st.spinner("Predicting with models..."):
            for model_name, model in models.items():
                predicted_class, confidence, inference_time, all_predictions = predict_with_model(model, image_array)
                results[model_name] = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'inference_time': inference_time,
                    'all_predictions': all_predictions
                }
        
        st.header("Result")
        
        table_data = []
        for model_name in results.keys():
            table_data.append({
                'Model': model_name,
                'Predicted Class': results[model_name]['predicted_class'],
                'Confidence (%)': f"{results[model_name]['confidence']:.2f}",
                'Inference Time (s)': f"{results[model_name]['inference_time']:.4f}"
            })
        
        df_results = pd.DataFrame(table_data)
        

        
        st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True
        )
        
        
        st.markdown("---")
        
        # 2. Histogram Confidence
        st.header("Confidence")
        fig_confidence = plot_confidence_histogram(results)
        st.pyplot(fig_confidence)
        
        st.markdown("---")
        
        # 3. Histogram Inference Time
        st.header("Inference Time")
        fig_time = plot_inference_time_histogram(results)
        st.pyplot(fig_time)
        
        st.markdown("---")
        
        # 4. Probability Distribution
        st.header("Probability Distribution Details")
        
        tabs = st.tabs(list(models.keys()))
        
        for idx, (model_name, tab) in enumerate(zip(models.keys(), tabs)):
            with tab:
                fig_prob = plot_class_probabilities(
                    results[model_name]['all_predictions'],
                    model_name
                )
                st.pyplot(fig_prob)
                
                #top 3 predictions
                top3_indices = np.argsort(results[model_name]['all_predictions'])[-3:][::-1]
                st.subheader("Top 3 Predictions:")
                for i, idx in enumerate(top3_indices, 1):
                    prob = results[model_name]['all_predictions'][idx] * 100
                    st.write(f"{i}. **{class_names[idx]}**: {prob:.2f}%")
        
        st.markdown("---")
        
        # Summary
        st.header("Summary")
        
        best_confidence_model = max(results.keys(), key=lambda k: results[k]['confidence'])
        st.success(f"**Best confidence:** {best_confidence_model} ({results[best_confidence_model]['confidence']:.2f}%)")
    
        fastest_model = min(results.keys(), key=lambda k: results[k]['inference_time'])
        st.info(f"**fastest:** {fastest_model} ({results[fastest_model]['inference_time']:.4f}s)")
        
    else:

        st.info("Please upload a picture")
        
        st.header("Class can be predicted")
        cols = st.columns(6)
        for idx, class_name in enumerate(class_names):
            with cols[idx]:
                st.markdown(f"**{idx + 1}. {class_name.capitalize()}**")

if __name__ == "__main__":
    main()
