import streamlit as st
import cv2
import numpy as np

# Function to process the image and find ellipses
def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    _, thresh_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # Apply morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    morph_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel, iterations=3)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours in the binary image
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define minimum and maximum area
    min_area = 1000
    max_area = 15000

    # Filter contours based on area range
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    # Convert the binary image to BGR (for displaying colored contours)
    output_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Draw all contours on the image to visualize them
    # cv2.drawContours(output_image, filtered_contours, -1, (255, 0, 0), 2)  # blue color contours

    total_grain_count = 0

    # Conversion factor (mm per pixel)
    conversion_factor = 310 / 7015  # mm per pixel

    # List to store the dimensions of the rice grains
    rice_grain_dimensions = []

    # Counters for whole, broken, and normal grains
    whole_grain_count = 0
    broken_grain_count = 0
    normal_grain_count = 0

    # Loop over the contours
    for i, contour in enumerate(filtered_contours):
        # Fit an ellipse to the contour if it has at least 5 points
        if len(contour) >= 5:
            total_grain_count += 1
            ellipse = cv2.fitEllipse(contour)
            # Get the dimensions of the ellipse
            (center, axes, orientation) = ellipse
            major_axis_length = max(axes)  # major axis
            minor_axis_length = min(axes)  # minor axis

            # Convert dimensions to millimeters
            major_axis_mm = major_axis_length * conversion_factor
            minor_axis_mm = minor_axis_length * conversion_factor

            # Classify the rice grain and draw the ellipse
            if major_axis_mm >= 7:
                whole_grain_count += 1
                ellipse_color = (0, 255, 0)  # Green for whole kernels
            elif major_axis_mm <= 4.5:
                broken_grain_count += 1
                ellipse_color = (0, 0, 255)  # Orange for broken kernels
            else:
                normal_grain_count += 1
                ellipse_color = (0, 165, 255)  # Blue for normal rice

            # Draw the ellipse on the image
            cv2.drawContours(output_image, [contour], 0, ellipse_color, 10)

            # Store the dimensions
            rice_grain_dimensions.append((major_axis_mm, minor_axis_mm))

            # Add a label next to each ellipse
            label_position = (int(center[0]), int(center[1]))
            cv2.putText(output_image, str(total_grain_count), label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 7, cv2.LINE_AA)  # Red color labels

    return output_image, rice_grain_dimensions, total_grain_count, whole_grain_count, broken_grain_count, normal_grain_count

# Streamlit UI
st.title('Rice Grain Measurement')


# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tiff"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    # Check if the image is loaded correctly
    if image is None:
        st.error("Error loading the image. Please try again with a different file.")
    else:
        # Convert grayscale images to BGR format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Process the image
        output_image, rice_grain_dimensions, total_grain_count, whole_grain_count, broken_grain_count, normal_grain_count = process_image(image)

        # Convert the output image to RGB format for display
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        # Display the original and processed images
        st.image([image, output_image_rgb], caption=['Original Image', 'Processed Image with Ellipses'], use_column_width=True)

        # Display the dimensions of the rice grains
        st.write("Dimensions of the rice grains in millimeters:")
        for i, (major_mm, minor_mm) in enumerate(rice_grain_dimensions):
            st.write(f"Rice grain {i+1}: Height = {major_mm:.2f} mm, Width = {minor_mm:.2f} mm")

        # Display the counts of whole, broken, and normal grains
        st.write(f"Total number of rice grains: {total_grain_count}")
        st.write(f"Total number of whole grains(Green): {whole_grain_count}")
        st.write(f"Total number of broken grains(red): {broken_grain_count}")
        st.write(f"Total number of normal grains(orange): {normal_grain_count}")