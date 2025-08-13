import json, numpy as np, cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="X-ray Pneumonia Classifier", page_icon="🩺")
st.title("🩺 Chest X-ray Pneumonia Classifier")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("models/mobilenetv2_finetuned.keras")
    with open("models/class_names.json") as f:
        class_names = json.load(f)
    tau_default = 0.5
    try:
        with open("models/threshold.json") as f:
            tau_default = float(json.load(f)["threshold"])
    except Exception:
        pass
    # find last conv for Grad-CAM
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    # fallback if conv not found in top graph
    if last_conv is None:
        for l in model.layers:
            if isinstance(l, tf.keras.Model):
                for sub in reversed(l.layers):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        last_conv = sub.name
                        break
                if last_conv: break
    return model, class_names, last_conv, tau_default

model, class_names, last_conv, tau_default = load_assets()
IMG_SIZE = (224, 224)
tau = st.slider("Decision threshold (τ)", 0.10, 0.95, value=float(tau_default), step=0.01)

def to_tensor_rgb(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)[None, ...]
    return preprocess_input(x.copy())

def grad_cam_array(img_array, model, last_conv_layer_name):
    conv_layer = None
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
    except Exception:
        for l in model.layers:
            if isinstance(l, tf.keras.Model):
                try:
                    conv_layer = l.get_layer(last_conv_layer_name)
                    break
                except Exception:
                    pass
    grad_model = tf.keras.models.Model([model.inputs],
                                       [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
    conv_out = conv_out[0].numpy()
    heatmap = np.maximum(np.tensordot(conv_out, pooled, axes=([2],[0])), 0)
    heatmap /= (heatmap.max() + 1e-10)
    return heatmap

uploaded = st.file_uploader("Upload a chest X-ray (JPG/PNG)", type=["jpg","jpeg","png"])
if uploaded:
    pil = Image.open(uploaded)
    st.image(pil, caption="Input image", use_container_width=True)

    x_pp = to_tensor_rgb(pil)
    prob = float(model.predict(x_pp, verbose=0)[0][0])
    pred = int(prob >= tau)
    st.subheader(f"Prediction: **{class_names[pred]}**  (p={prob:.3f}, τ={tau:.2f})")

    heatmap = grad_cam_array(x_pp, model, last_conv)
    orig = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, 0.35, orig, 0.65, 0)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Grad-CAM", use_container_width=True)

st.caption("⚠️ Educational demo only — not for clinical use.")
