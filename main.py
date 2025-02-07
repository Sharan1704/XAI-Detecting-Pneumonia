import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from data_preprocessing import create_data_generators
from train import train_model

# Function to generate Grad-CAM heatmap
def grad_cam(model, img_array, layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
        
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(output, weights), axis=-1)
    cam = tf.maximum(cam, 0)  # ReLU to discard negative values
    cam = cam / tf.reduce_max(cam)  # Normalize heatmap
    return cam.numpy()

# Interactive visualization using Plotly
def interactive_visualization(img_path, cam, prediction, confidence):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Original Image", "Grad-CAM Heatmap")
    )
    
    # Original image
    img = image.load_img(img_path, target_size=(224, 224))
    fig.add_trace(
        go.Image(z=img),
        row=1, col=1
    )
    
    # Heatmap
    fig.add_trace(
        go.Heatmap(
            z=cam, colorscale='Jet', opacity=0.6,
            hoverinfo='z', name="Heatmap"
        ),
        row=1, col=2
    )
    
    # Title and layout
    fig.update_layout(
        title_text=f"Prediction: {prediction} (Confidence: {confidence:.2%})",
        title_x=0.5, showlegend=False,
        annotations=[
            dict(
                text="This area indicates lung consolidation.", x=0.75, y=0.5,
                xref="paper", yref="paper", showarrow=True, arrowhead=2
            )
        ]
    )
    fig.show()

# Comparison tool
def compare_normal_abnormal(normal_img_path, abnormal_img_path):
    normal_img = image.load_img(normal_img_path, target_size=(224, 224))
    abnormal_img = image.load_img(abnormal_img_path, target_size=(224, 224))
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Normal X-ray", "Abnormal X-ray")
    )
    
    fig.add_trace(go.Image(z=normal_img), row=1, col=1)
    fig.add_trace(go.Image(z=abnormal_img), row=1, col=2)
    
    fig.update_layout(title_text="Normal vs Abnormal X-ray Comparison", title_x=0.5)
    fig.show()

if __name__ == "__main__":
    # Paths
    base_dir = r'E:\Medical Image Classifier\Data'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    
    # Load or train model
    model_path = 'pneumonia_classification_model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model, history = train_model(train_dir, val_dir, test_dir)
    
    # Test on sample image
    test_image_path = r'E:\Medical Image Classifier\Data\chest_xray\test\NORMAL\NORMAL2-IM-0098-0001.jpeg'
    img = image.load_img(test_image_path, target_size=(224, 224))
    x = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    
    # Prediction and Grad-CAM
    cam = grad_cam(model, x)
    prediction = "PNEUMONIA" if model.predict(x)[0][0] > 0.5 else "NORMAL"
    confidence = model.predict(x)[0][0] if model.predict(x)[0][0] > 0.5 else 1 - model.predict(x)[0][0]
    
    # Interactive visualization
    interactive_visualization(test_image_path, cam, prediction, confidence)
    
    # Comparison tool
    compare_normal_abnormal(
        normal_img_path=r'E:\Medical Image Classifier\Data\chest_xray\test\NORMAL\IM-0027-0001.jpeg',
        abnormal_img_path=r'E:\Medical Image Classifier\Data\chest_xray\test\PNEUMONIA\person1_virus_12.jpeg'
    )
