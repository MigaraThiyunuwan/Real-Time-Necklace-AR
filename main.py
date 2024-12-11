import cv2
import numpy as np
import mediapipe as mp
import os

# Define the paths for the Haar cascade files and necklace images
cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
necklace5 = os.path.join(os.path.dirname(__file__), 'necklace5.png')
necklace3 = os.path.join(os.path.dirname(__file__), 'necklace3.png')
necklace7 = os.path.join(os.path.dirname(__file__), 'necklace7.png')

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load and resize the necklace images
def load_and_resize_necklace(image_path, size=(100, 100)):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load necklace image: {image_path}")
    return cv2.resize(img, size)

necklace_img1 = load_and_resize_necklace(necklace5)
necklace_img2 = load_and_resize_necklace(necklace3)
necklace_img3 = load_and_resize_necklace(necklace7)

# Combine the necklace images horizontally with spacing
def combine_necklaces(necklace_imgs, spacing=20):
    total_width = sum(img.shape[1] for img in necklace_imgs) + spacing * (len(necklace_imgs) - 1)
    height = max(img.shape[0] for img in necklace_imgs)
    
    combined_img = np.zeros((height, total_width, 4), dtype=np.uint8)
    
    x_offset = 0
    for img in necklace_imgs:
        combined_img[:, x_offset:x_offset + img.shape[1]] = img
        x_offset += img.shape[1] + spacing
    
    return combined_img

combined_necklaces = combine_necklaces([necklace_img1, necklace_img2, necklace_img3])

# Convert combined necklace image to BGR format
def convert_to_bgr(image):
    if image.shape[2] == 4:
        bgr_img = image[:, :, :3]
        alpha = image[:, :, 3] / 255.0
        return bgr_img, alpha
    return image, None

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def detect_face_and_neck(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        neck_x = x + int(w * 0.09) 
        neck_y = y + int(h * 1)
        #neck_w = w
        neck_w = int(w * 0.9)
        neck_h = int(h * 0.8)
        return (neck_x, neck_y, neck_w, neck_h)
    
    return None

def detect_hand_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    
    index_finger_tip = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    return image, index_finger_tip

def get_finger_position(index_finger_tip, image_width, image_height):
    if index_finger_tip:
        x = int(index_finger_tip.x * image_width)
        y = int(index_finger_tip.y * image_height)
        return (x, y)
    return None

def is_finger_on_necklace(finger_position, target_zones):
    for zone in target_zones:
        if zone[0] <= finger_position[0] <= zone[2] and zone[1] <= finger_position[1] <= zone[3]:
            return True
    return False

def overlay_necklace(frame, necklace_img, neck_region):
    (x, y, w, h) = neck_region
    
    # Resize the necklace image to fit the neck region
    resized_necklace = cv2.resize(necklace_img, (w, h))
    
    necklace_rgb = resized_necklace[:, :, :3]
    alpha_mask = resized_necklace[:, :, 3] / 255.0
    
    # Extract the region of interest (ROI) from the frame
    roi = frame[y:y+h, x:x+w]
    
    # Check if the size of the roi matches the size of the resized necklace
    if roi.shape[0] < necklace_rgb.shape[0] or roi.shape[1] < necklace_rgb.shape[1]:
        # Resize the necklace image to fit the roi dimensions
        resized_necklace = cv2.resize(necklace_img, (roi.shape[1], roi.shape[0]))
        necklace_rgb = resized_necklace[:, :, :3]
        alpha_mask = resized_necklace[:, :, 3] / 255.0
    
    # Resize the roi to match the resized necklace dimensions
    roi_resized = cv2.resize(roi, (necklace_rgb.shape[1], necklace_rgb.shape[0]))

    # Perform the overlay
    for c in range(3):
        roi_resized[:, :, c] = roi_resized[:, :, c] * (1 - alpha_mask) + necklace_rgb[:, :, c] * alpha_mask

    # Place the overlay on the frame
    frame[y:y+h, x:x+w] = roi_resized



cap = cv2.VideoCapture(0)

# Create a named window
cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)

# Set the window size
window_width = 1400
window_height = 1000
cv2.resizeWindow('Webcam', window_width, window_height)

# Initialize the default selected index to 0 (necklace5)
selected_index = 0

# Dictionary to map selected index to necklace name
necklace_names = {
    0: "Necklace 1",
    1: "Necklace 2",
    2: "Necklace 3"
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    combined_necklaces_bgr, alpha_mask = convert_to_bgr(combined_necklaces)
    
    top_area = combined_necklaces_bgr.shape[0]
    frame_with_necklaces = np.copy(frame)
    
    if top_area <= frame_with_necklaces.shape[0]:
        frame_with_necklaces[:top_area, :combined_necklaces_bgr.shape[1]] = combined_necklaces_bgr
    
    frame_with_necklaces, index_finger_tip = detect_hand_landmarks(frame_with_necklaces)
    finger_position = get_finger_position(index_finger_tip, frame_with_necklaces.shape[1], frame_with_necklaces.shape[0])
    
    if finger_position:
        necklace_zones = [
            (0, 0, combined_necklaces_bgr.shape[1] // 3, top_area),
            (combined_necklaces_bgr.shape[1] // 3, 0, 2 * (combined_necklaces_bgr.shape[1] // 3), top_area),
            (2 * (combined_necklaces_bgr.shape[1] // 3), 0, combined_necklaces_bgr.shape[1], top_area)
        ]
        
        if is_finger_on_necklace(finger_position, necklace_zones):
            selected_index = necklace_zones.index(next(zone for zone in necklace_zones if is_finger_on_necklace(finger_position, [zone])))
    
    neck_region = detect_face_and_neck(frame)
    if neck_region:
        if selected_index == 0:
            overlay_necklace(frame_with_necklaces, necklace_img1, neck_region)
        elif selected_index == 1:
            overlay_necklace(frame_with_necklaces, necklace_img2, neck_region)
        elif selected_index == 2:
            overlay_necklace(frame_with_necklaces, necklace_img3, neck_region)
    
    # Add message indicating the currently overlaying necklace
    message = f"Wearing {necklace_names[selected_index]}"
    cv2.putText(frame_with_necklaces, message, (frame_with_necklaces.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1, cv2.LINE_AA)
    
    if isinstance(frame_with_necklaces, np.ndarray) and len(frame_with_necklaces.shape) == 3 and frame_with_necklaces.shape[2] in [3, 4]:
        cv2.imshow('Webcam', frame_with_necklaces)
    else:
        print("Error: Invalid frame format for display.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
