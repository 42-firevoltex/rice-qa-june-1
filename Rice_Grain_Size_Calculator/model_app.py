import os
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import base64
from streamlit_image_comparison import image_comparison
from torchvision.ops import nms
import torch

st.set_page_config(
    page_title="Rice Grain Dimensioner", page_icon=":seedling:"
)

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model_path = os.path.join("Rice_Grain_Size_Calculator", "mediumv2.pt")
model = load_model(model_path)

class_names = model.names


@st.cache_data
def process_image(image, _model):
    image = cv2.resize(image, (2048, 2048))

    # Perform segmentation
    results = model(image)[0]

    # Extract masks, class IDs, and bounding boxes from results
    masks = results.masks
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()

    if masks is not None:
        masks_data = masks.data.cpu().numpy()

        output_image = image.copy() 

        total_grain_count = 0
        rice_grain_dimensions = []
        whole_grain_count = 0
        broken_grain_count = 0
        normal_grain_count = 0

        # Conversion factor (mm per pixel)
        conversion_factor = 310 / 2048  

        grain_images = []
        contours = []

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        confidences_tensor = torch.tensor(confidences, dtype=torch.float32)
        indices = nms(boxes_tensor, confidences_tensor, iou_threshold=0.3).cpu().numpy()

        for i in indices:
            mask = masks_data[i]
            class_id = class_ids[i]
            box = boxes[i]
            confidence = confidences[i]

            x_min, y_min, x_max, y_max = box

            contour = np.where(mask)
            if len(contour[0]) >= 5:
                total_grain_count += 1
                contour = np.stack((contour[1], contour[0]), axis=1)
                contours.append(contour)
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                major_axis_length = max(axes)  # major axis
                minor_axis_length = min(axes)  # minor axis

                major_axis_mm = major_axis_length * conversion_factor
                minor_axis_mm = minor_axis_length * conversion_factor

                # Classify the rice grain
                if major_axis_mm >= 7:
                    whole_grain_count += 1
                    color_name = "Whole Rice"
                elif major_axis_mm <= 4.5:
                    broken_grain_count += 1
                    color_name = "Broken Rice"
                else:
                    normal_grain_count += 1
                    color_name = "Normal Rice"

                class_name = class_names[class_id]

                # Store the dimensions
                rice_grain_dimensions.append(
                    (total_grain_count, major_axis_mm, minor_axis_mm, color_name, class_name)
                )

                grain_image = image[y_min:y_max, x_min:x_max]
                if grain_image is None or not isinstance(grain_image, np.ndarray):
                    st.warning(f"Invalid grain image at index {total_grain_count}")
                else:
                    grain_images.append(grain_image)

                # Add a label next to each ellipse
                label_position = (int(center[0]), int(center[1]))
                cv2.putText(
                    output_image,
                    str(total_grain_count),
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )  

        return (
            output_image,
            rice_grain_dimensions,
            total_grain_count,
            whole_grain_count,
            broken_grain_count,
            normal_grain_count,
            grain_images,
            contours,
        )
    else:
        return image, [], 0, 0, 0, 0, [], []

@st.cache_data
def draw_selected_grain(image,rice_grain_dimensions, contours, selected_grain_id=None):
    output_image = image.copy()

    if selected_grain_id is None:  
        for grain_id, (_, major_axis_mm, minor_axis_mm, _, _) in enumerate(rice_grain_dimensions, start=1):
            contour = contours[grain_id - 1]
            if major_axis_mm >= 7:
                contour_color = (0, 255, 0)  # Green for whole kernels
            elif major_axis_mm <= 4.5:
                contour_color = (0, 0, 255)  # Red for broken kernels
            else:
                contour_color = (0, 165, 255)  # Orange for normal rice

            # Draw the contour and label
            cv2.drawContours(output_image, [contour], -1, contour_color, 1)
            center = tuple(map(int, contour.mean(axis=0)))
            cv2.putText(
                output_image,
                str(grain_id),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
    else:
        for grain_id, (_, major_axis_mm, minor_axis_mm, _, _) in enumerate(rice_grain_dimensions, start=1):
            if grain_id == selected_grain_id:
                contour = contours[grain_id - 1]
                if major_axis_mm >= 7:
                    contour_color = (0, 255, 0)  
                elif major_axis_mm <= 4.5:
                    contour_color = (0, 0, 255)  
                else:
                    contour_color = (0, 165, 255)  

                # Draw the contour and label for the selected grain
                cv2.drawContours(output_image, [contour], -1, contour_color, 1)
                center = tuple(map(int, contour.mean(axis=0)))
                cv2.putText(
                    output_image,
                    str(selected_grain_id),
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                break

    return output_image

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

st.title("Rice Grain Dimensioner")
st.markdown(
    """
    <style>
    body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    }
    .sidebar .sidebar-content {
        background: rgba(0, 0, 0, 0.5); /* Optional: Adds a semi-transparent background color to the sidebar */
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color);
    }
    .css-1d391kg, .css-1offfwp, .css-1ux4gn, .css-1l1m2wd {
        color: var(--text-color);
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png", "tiff"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    if image is None:
        st.error("Error loading the image. Please try again with a different file.")
    else:
        # Check the resolution of the image
        if image.shape[0] < 3000 or image.shape[1] < 4000:
            st.error("Image resolution is too low. Please upload an image with at least 4000x3000 resolution.")
        else:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            (
                output_image,
                rice_grain_dimensions,
                total_grain_count,
                whole_grain_count,
                broken_grain_count,
                normal_grain_count,
                grain_images,
                contours,
            ) = process_image(image, model)

            # Selection box for grain ID
            selected_grain_id = st.selectbox("Select Grain ID to display (or 'All' to display all):", ["All"] + list(range(1, total_grain_count + 1)))

            if selected_grain_id == "All":
                selected_grain_id = None
            else:
                selected_grain_id = int(selected_grain_id)

            output_image = draw_selected_grain(output_image, rice_grain_dimensions, contours, selected_grain_id)

            if output_image is not None:
                output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                output_image_rgb = cv2.resize(output_image_rgb, (original_image_rgb.shape[1], original_image_rgb.shape[0]))

                # Image slider
                image_comparison(
                    img1=original_image_rgb,
                    img2=output_image_rgb,
                    label1="Original Image",
                    label2="Processed Image",
                )

                st.write(f"Total number of rice grains: {total_grain_count}")
                st.write(f"Total number of whole grains (Green): {whole_grain_count}")
                st.write(f"Total number of broken grains (Red): {broken_grain_count}")
                st.write(f"Total number of normal grains (Orange): {normal_grain_count}")

                st.write("Dimensions of the rice grains in millimeters:")
                df = pd.DataFrame(
                    rice_grain_dimensions,
                    columns=["Grain ID", "Height (mm)", "Width (mm)", "Classification", "Detected Class"],
                )

                # Display each detected rice grain image in the DataFrame
                df["Grain Image"] = [f'<img src="data:image/png;base64,{image_to_base64(img)}" width="150" height="150">' for img in grain_images]
                df.set_index("Grain ID", inplace=True)

                # Convert the DataFrame to HTML with escaping disabled for displaying images
                df_html = df.to_html(escape=False)

                # Display the DataFrame as a scrollable table
                st.markdown(
                    f"""
                    <div style="height:500px;overflow:auto;">
                        {df_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Provide a download button for the DataFrame
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='rice_grain_dimensions.csv',
                    mime='text/csv',
                )