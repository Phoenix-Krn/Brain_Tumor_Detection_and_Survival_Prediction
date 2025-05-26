import streamlit as st
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import os
import torch.nn.functional as F
from monai.networks.nets import UNet
import joblib
from sklearn.preprocessing import LabelEncoder
import shap
from fpdf import FPDF
import base64
import time
import imageio
import smtplib
from email.message import EmailMessage
from skimage import measure, morphology

# ========== Configuration ========== 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "3d_unet_brain_tumor_segmenter.pth" #Change path to your actual path
SURVIVAL_MODEL_PATH = "rf_survival_model.pkl" #Change path to your actual path

# ========== Load Models ========== 
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

survival_model = joblib.load(SURVIVAL_MODEL_PATH)
le = LabelEncoder()
le.fit(["GTR", "STR", "NA"])

# ========== Helper Functions ========== 
def load_nifti_image(uploaded_file):
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    img = nib.load(tmp_path)
    return img.get_fdata(), img.header.get_zooms()

def preprocess_volume(volume):
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    volume = F.interpolate(volume, size=(128, 128, 128), mode='trilinear', align_corners=False)
    return volume.to(DEVICE)

def clean_mask(mask, min_size=1000):
    labeled = measure.label(mask > 0.5)
    cleaned = morphology.remove_small_objects(labeled, min_size=min_size)
    return (cleaned > 0).astype(np.uint8)

def predict(image_tensor, voxel_threshold=500):
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        tumor_mask = (output > 0.6).float()
        mask_np = tumor_mask.cpu().numpy()[0, 0]
        cleaned_mask = clean_mask(mask_np, min_size=1000)
        tumor_voxel_count = np.sum(cleaned_mask > 0.5)
        tumor_present = tumor_voxel_count > voxel_threshold
    return tumor_present, cleaned_mask

def calculate_tumor_volume(mask, voxel_spacing):
    voxel_volume_cm3 = np.prod(voxel_spacing) / 1000.0  # Convert mm^3 to cm^3
    return np.sum(mask > 0.5) * voxel_volume_cm3

def explain_prediction(model, features, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    st.subheader("\U0001F50D SHAP Explanation for Survival Prediction")
    fig = plt.figure()
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=features[0],
        feature_names=feature_names
    ), show=False)
    st.pyplot(fig)

def generate_multi_views(volume, mask=None):
    views = {}
    cols = st.columns(3)
    for i, (axis, name) in enumerate(zip([0, 1, 2], ["axial", "sagittal", "coronal"])):
        slice_ = volume.take(indices=volume.shape[axis] // 2, axis=axis)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(slice_, cmap='gray')
        ax.axis('off')
        path = os.path.join(tempfile.gettempdir(), f"{name}_view.png")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        views[name] = path
        cols[i].image(path, caption=f"{name.capitalize()} View", use_column_width=True)
    return views

def generate_pdf(age, resection, volume, survival_days, views):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Brain Tumor Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Age: {age}\nResection Type: {resection}\nTumor Volume: {volume:.2f} cm³\nEstimated Survival Days: {int(survival_days)}")

    for name, path in views.items():
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"{name.capitalize()} View", ln=True)
        pdf.image(path, x=10, w=180)

    tmp_path = os.path.join(tempfile.gettempdir(), "report.pdf")
    pdf.output(tmp_path)
    with open(tmp_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="report.pdf">\U0001F4C4 Download PDF Report</a>'
    st.markdown(href, unsafe_allow_html=True)
    return tmp_path

def send_email_with_report(receiver_email, pdf_path):
    sender_email = "your_email@example.com"
    sender_password = "your_app_password"

    msg = EmailMessage()
    msg['Subject'] = 'Brain Tumor Detection Report'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content('Please find the attached brain tumor detection report.')

    with open(pdf_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype='application', subtype='pdf', filename='report.pdf')

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

# ========== Streamlit App ========== 
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
st.title("\U0001F9E0 Brain Tumor Detection App")
st.write("Upload a brain MRI scan in `.nii` or `.nii.gz` format.")

uploaded_file = st.file_uploader("Upload MRI Scan", type=["nii", "nii.gz"])

if uploaded_file is not None:
    st.info("⏳ Processing image...")
    try:
        if "volume_np" not in st.session_state:
            volume_np, voxel_spacing = load_nifti_image(uploaded_file)
            st.session_state.voxel_spacing = voxel_spacing  # Store voxel spacing in session state
            preprocessed_volume = preprocess_volume(volume_np)
            tumor_present, prediction_mask = predict(preprocessed_volume)
            st.session_state.volume_np = volume_np
            st.session_state.prediction_mask = prediction_mask
            st.session_state.tumor_present = tumor_present
        else:
            volume_np = st.session_state.volume_np
            prediction_mask = st.session_state.prediction_mask
            tumor_present = st.session_state.tumor_present
            voxel_spacing = st.session_state.voxel_spacing  # Retrieve voxel spacing from session state

        st.subheader("Prediction Results")
        if tumor_present:
            st.error("⚠️ Tumor Detected")
            tumor_volume = calculate_tumor_volume(prediction_mask, voxel_spacing)

            age = st.number_input("Enter Patient Age", min_value=0, max_value=120, value=60)
            resection = st.selectbox("Select Extent of Resection", ["Complete", "Incomplete", "None"])
            resection_map = {"Complete": "GTR", "Incomplete": "STR", "None": "NA"}
            resection_label = resection_map[resection]
            resection_encoded = le.transform([resection_label])[0]

            if st.button("Predict Survival"):
                features = np.array([[age, resection_encoded, tumor_volume]])
                survival_days = survival_model.predict(features)[0]
                st.success(f"🗓️ Estimated Survival: **{int(survival_days)} days**")
                explain_prediction(survival_model, features, ["Age", "Extent of Resection", "Tumor Volume (cm³)"])
                multi_view_images = generate_multi_views(volume_np, prediction_mask)
                pdf_path = generate_pdf(age, resection, tumor_volume, survival_days, multi_view_images)
                
                if st.checkbox("📧 Email me the report"):
                    user_email = st.text_input("Enter your email address")
                    if st.button("Send Email"):
                        if user_email:
                            send_email_with_report(user_email, pdf_path)
                            st.success("✅ Report sent successfully!")
                        else:
                            st.warning("⚠️ Please enter a valid email address.")

                st.subheader("🎞️ Animated Slice Viewer")
                scroll_speed = st.slider("Scroll Speed (seconds)", 0.05, 1.0, 0.2)
                save_gif = st.checkbox("💾 Save animation as GIF")
                gif_frames = []

                max_slices = min(volume_np.shape[0], prediction_mask.shape[0])
                for idx in range(max_slices):
                    mri_slice = (volume_np[idx] - np.min(volume_np[idx])) / (np.max(volume_np[idx]) - np.min(volume_np[idx]) + 1e-8)
                    tumor_slice = prediction_mask[idx]
                    fig, ax = plt.subplots()
                    ax.imshow(mri_slice, cmap='gray')
                    ax.imshow(tumor_slice, cmap='Reds', alpha=0.4)
                    ax.set_title(f"Slice {idx}")
                    ax.axis('off')
                    st.pyplot(fig)

                    if save_gif:
                        buf = BytesIO()
                        fig.savefig(buf, format='png')
                        buf.seek(0)
                        gif_frames.append(imageio.v2.imread(buf))
                        plt.close(fig)

                    time.sleep(scroll_speed)

                if save_gif and gif_frames:
                    gif_path = os.path.join(tempfile.gettempdir(), "tumor_animation.gif")
                    imageio.mimsave(gif_path, gif_frames, duration=scroll_speed)
                    with open(gif_path, "rb") as f:
                        gif_data = f.read()
                        b64 = base64.b64encode(gif_data).decode()
                        st.markdown(f'<a href="data:image/gif;base64,{b64}" download="tumor_animation.gif">💾 Download GIF</a>', unsafe_allow_html=True)

        else:
            st.success("✅ No Tumor Detected")
            st.write("👍 You are safe!")
            multi_view_images = generate_multi_views(volume_np)
            pdf_path = generate_pdf("NA", "NA", 0.0, 0, multi_view_images)
            if st.checkbox("📧 Email me the report"):
                user_email = st.text_input("Enter your email address")
                if st.button("Send Email"):
                    if user_email:
                        send_email_with_report(user_email, pdf_path)
                        st.success("✅ Report sent successfully!")
                    else:
                        st.warning("⚠️ Please enter a valid email address.")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
