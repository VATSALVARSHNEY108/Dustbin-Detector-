import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import urllib.request


st.set_page_config(
    page_title="Dustbin Detector",
    page_icon="🗑️",
    layout="wide"
)

@st.cache_resource
def load_detection_model():
    try:
        # Use YOLOv3-tiny via OpenCV DNN - smaller and more reliable than full YOLO
        weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
        config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
        names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

        weights_path = "yolov3-tiny.weights"
        config_path = "yolov3-tiny.cfg"
        names_path = "coco.names"

        try:
            if not os.path.exists(weights_path):
                urllib.request.urlretrieve(weights_url, weights_path)

            if not os.path.exists(config_path):
                urllib.request.urlretrieve(config_url, config_path)

            if not os.path.exists(names_path):
                urllib.request.urlretrieve(names_url, names_path)

        except Exception as download_error:
            return ""

        try:
            net = cv2.dnn.readNet(weights_path, config_path)


            with open(names_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]

            return {"net": net, "classes": classes, "type": "yolo"}
        except Exception as load_error:
            st.warning(f"Failed to load YOLO model: {load_error}")
            st.info("Falling back to color-based detection...")
            return "color_detector"

    except Exception as e:
        st.error(f"Failed to initialize detection system: {str(e)}")
        return None


def detect_dustbins_yolo(image, model_data):
    net = model_data["net"]
    classes = model_data["classes"]

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    detections = []
    annotated_image = image.copy()

    if len(indices) > 0:
        indices = np.array(indices).flatten()
        for i in indices:
            x, y, w_box, h_box = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]

            if class_id < len(classes):
                class_name = classes[class_id]

                container_classes = ['bottle', 'cup', 'bowl', 'vase', 'potted plant', 'chair', 'toilet', 'sink']
                is_dustbin_like = class_name in container_classes

                color = (0, 255, 0) if is_dustbin_like else (255, 0, 0)
                thickness = 3 if is_dustbin_like else 2

                cv2.rectangle(annotated_image, (x, y), (x + w_box, y + h_box), color, thickness)

                label = f"{class_name}: {confidence:.2f}"
                if is_dustbin_like:
                    label = f"Possible Dustbin ({class_name}): {confidence:.2f}"

                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                cv2.rectangle(
                    annotated_image,
                    (x, y - text_height - baseline - 10),
                    (x + text_width, y),
                    color,
                    -1
                )

                cv2.putText(
                    annotated_image,
                    label,
                    (x, y - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                detections.append({
                    'class': class_name,
                    'confidence': float(confidence),
                    'bbox': [int(x), int(y), int(x + w_box), int(y + h_box)],
                    'is_dustbin_like': is_dustbin_like
                })

    return annotated_image, detections


def detect_dustbins_color(image):
    (h, w) = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    detections = []
    annotated_image = image.copy()

    colors_info = [
        ("Blue Dustbin", np.array([100, 50, 50]), np.array([130, 255, 255]), (255, 0, 0)),
        ("Green Dustbin", np.array([40, 50, 50]), np.array([80, 255, 255]), (0, 255, 0)),
        ("Red/Orange Dustbin", np.array([0, 50, 50]), np.array([10, 255, 255]), (0, 0, 255)),
    ]

    for color_name, lower, upper, box_color in colors_info:
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)

                # Calculate aspect ratio to filter for dustbin-like shapes
                aspect_ratio = w_rect / h_rect
                if 0.3 < aspect_ratio < 3.0:

                    confidence = min(0.95, area / (h * w * 0.1))

                    cv2.rectangle(annotated_image, (x, y), (x + w_rect, y + h_rect), box_color, 3)

                    label = f"{color_name}: {confidence:.2f}"

                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )

                    cv2.rectangle(
                        annotated_image,
                        (x, y - text_height - baseline - 10),
                        (x + text_width, y),
                        box_color,
                        -1
                    )

                    cv2.putText(
                        annotated_image,
                        label,
                        (x, y - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    detections.append({
                        'class': color_name,
                        'confidence': float(confidence),
                        'bbox': [int(x), int(y), int(x + w_rect), int(y + h_rect)],
                        'is_dustbin_like': True
                    })

    return annotated_image, detections


def detect_dustbins(image, model):
    if model is None:
        return None, []

    try:
        if isinstance(model, dict) and model.get("type") == "yolo":
            return detect_dustbins_yolo(image, model)
        else:
            return detect_dustbins_color(image)

    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        return None, []


def display_detection_summary(detections):
    st.subheader("Detection Summary")

    if detections:
        dustbin_detections = [d for d in detections if d['is_dustbin_like']]
        other_detections = [d for d in detections if not d['is_dustbin_like']]

        if dustbin_detections:
            st.success(f"Found {len(dustbin_detections)} potential dustbin(s)!")

            st.write("**Potential Dustbins/Containers:**")
            for i, detection in enumerate(dustbin_detections, 1):
                st.write(f"{i}. **{detection['class']}** - Confidence: {detection['confidence']:.1%}")
                bbox = detection['bbox']
                st.write(f"   Location: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
        else:
            st.warning(
                "No clear dustbins detected. The image may contain other objects that could potentially be containers.")

        if other_detections:
            with st.expander(f"Other Detected Objects ({len(other_detections)})"):
                for i, detection in enumerate(other_detections, 1):
                    st.write(f"{i}. **{detection['class']}** - Confidence: {detection['confidence']:.1%}")

        st.subheader("Detection Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Objects", len(detections))
        with col2:
            st.metric("Potential Dustbins", len(dustbin_detections))
        with col3:
            avg_confidence = np.mean([d['confidence'] for d in dustbin_detections]) if dustbin_detections else 0
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")

    else:
        st.info("No objects detected in the image. Try uploading an image with clearer objects or containers.")


def display_batch_results(batch_results, all_detections):
    st.subheader("Batch Processing Results")

    total_dustbins = sum(len([d for d in result['detections'] if d['is_dustbin_like']]) for result in batch_results)
    total_objects = len(all_detections)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Images Processed", len(batch_results))
    with col2:
        st.metric("Total Objects", total_objects)
    with col3:
        st.metric("Total Dustbins", total_dustbins)
    with col4:
        avg_conf = np.mean([d['confidence'] for d in all_detections if d['is_dustbin_like']]) if [d for d in
                                                                                                  all_detections if d[
                                                                                                      'is_dustbin_like']] else 0
        st.metric("Avg Confidence", f"{avg_conf:.1%}")

    st.subheader("Individual Results")

    tab1, tab2, tab3 = st.tabs(["Gallery View", "Detailed View", "Summary Table"])

    with tab1:
        cols = st.columns(3)
        for idx, result in enumerate(batch_results):
            with cols[idx % 3]:
                st.write(f"**{result['filename']}**")
                st.image(result['annotated_image'], use_column_width=True)
                dustbin_count = len([d for d in result['detections'] if d['is_dustbin_like']])
                st.write(f"Dustbins found: {dustbin_count}")

    with tab2:
        for result in batch_results:
            dustbin_count = len([d for d in result['detections'] if d['is_dustbin_like']])
            total_count = len(result['detections'])

            with st.expander(f"{result['filename']} - {dustbin_count} dustbins, {total_count} total objects"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original**")
                    st.image(result['original_image'], use_column_width=True)
                with col2:
                    st.write("**Detected**")
                    st.image(result['annotated_image'], use_column_width=True)
                if result['detections']:
                    st.write("**Detections:**")
                    for i, detection in enumerate(result['detections'], 1):
                        icon = "🗑️" if detection['is_dustbin_like'] else "📦"
                        st.write(f"{icon} {detection['class']} - {detection['confidence']:.1%}")

    with tab3:
        import pandas as pd

        summary_data = []
        for result in batch_results:
            dustbin_count = len([d for d in result['detections'] if d['is_dustbin_like']])
            total_count = len(result['detections'])
            max_conf = max([d['confidence'] for d in result['detections']], default=0)

            summary_data.append({
                'Filename': result['filename'],
                'Total Objects': total_count,
                'Dustbins Found': dustbin_count,
                'Max Confidence': f"{max_conf:.1%}",
                'Status': '✅ Processed'
            })

        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)


def main():
    st.title("🗑️ Dustbin Detector")
    st.write("Upload an image to detect dustbins and waste containers using computer vision object detection.")

    # Load detection model
    with st.spinner("Loading detection model..."):
        model = load_detection_model()

    if model is None:
        st.error("Failed to load the detection model. Please check your internet connection and try again.")
        return

    st.success("Detection model loaded successfully!")

    # File upload - support both single and batch processing
    upload_mode = st.radio(
        "Upload Mode",
        ["Single Image", "Batch Processing"],
        help="Choose single image for quick analysis or batch processing for multiple images"
    )

    if upload_mode == "Single Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image containing dustbins or waste containers"
        )
        uploaded_files = [uploaded_file] if uploaded_file is not None else []
    else:
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )

    if uploaded_files:
        if upload_mode == "Single Image":
            # Single image processing
            image = Image.open(uploaded_files[0])

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)

            # Convert PIL image to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Perform detection
            with st.spinner("Detecting objects..."):
                annotated_image, detections = detect_dustbins(image_cv, model)

            if annotated_image is not None:
                # Convert back to RGB for display
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                with col2:
                    st.subheader("Detection Results")
                    st.image(annotated_image_rgb, caption="Detected Objects", use_column_width=True)

                # Display detection summary
                display_detection_summary(detections)

            else:
                st.error("Failed to process the image. Please try again with a different image.")

        else:
            # Batch processing
            st.subheader(f"Batch Processing - {len(uploaded_files)} Images")

            # Initialize batch results
            batch_results = []
            all_detections = []

            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process each image
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing image {idx + 1} of {len(uploaded_files)}: {uploaded_file.name}")

                image = Image.open(uploaded_file)
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Perform detection
                annotated_image, detections = detect_dustbins(image_cv, model)

                if annotated_image is not None:
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                    # Store results
                    batch_results.append({
                        'filename': uploaded_file.name,
                        'original_image': image,
                        'annotated_image': annotated_image_rgb,
                        'detections': detections
                    })
                    all_detections.extend(detections)

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("Batch processing complete!")

            display_batch_results(batch_results, all_detections)


if __name__ == "__main__":
    main()
