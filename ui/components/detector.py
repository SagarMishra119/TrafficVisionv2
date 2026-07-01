import numpy as np


def draw_boxes(image: np.ndarray, detections: list) -> np.ndarray:
    """Kept for compatibility; prefers detector.annotate_image if available."""
    import cv2
    img = image.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (6, 182, 212), 2)
        label = f'{d["vehicle_type"]} {d["confidence"]:.2f}'
        cv2.putText(img, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (6, 182, 212), 1, cv2.LINE_AA)
    return img


def run_pipeline_on_image(image_bgr: np.ndarray):
    """Runs detector -> counter -> classifier -> congestion using session backend objects.
    Returns (detections, annotated, counts, density, congestion)."""
    import streamlit as st
    from ui.helpers import get_counter, get_classifier, get_analytics

    detector = st.session_state.detector
    counter = get_counter()
    classifier = get_classifier()
    analytics = get_analytics()

    detections = detector.detect_image(image_bgr)
    try:
        annotated = detector.annotate_image(image_bgr, detections)
    except AttributeError:
        annotated = draw_boxes(image_bgr, detections)

    counts = counter.count(detections)
    density = (
        classifier.classify_with_weights(counts)
        if st.session_state.get("use_weighted")
        else classifier.classify(counts["total_vehicles"])
    )
    congestion = analytics.compute_congestion_index(counts)
    return detections, annotated, counts, density, congestion
