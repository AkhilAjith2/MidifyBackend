import os
import time
from collections import Counter

import cv2
import tensorflow as tf
import numpy as np

def normalize(image):
    return (255. - image) / 255.

def resize(image, height):
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img

def sparse_tensor_to_strs(sparse_tensor):
    indices = sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]

    strs = [[] for i in range(dense_shape[0])]

    string = []
    ptr = 0
    b = 0

    for idx in range(len(indices)):
        if indices[idx][0] != b:
            strs[b] = string
            string = []
            b = indices[idx][0]

        string.append(values[ptr])

        ptr = ptr + 1

    strs[b] = string

    return strs

def extract_staves(sheet_music_path, output_dir):
    # Step 1: Read the input image
    image = cv2.imread(sheet_music_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read image from {sheet_music_path}.")
        return []

    # Step 2: Threshold and preprocess
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    connected_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
    contours, _ = cv2.findContours(connected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # Step 3: Determine valid contours
    widths = [cv2.boundingRect(c)[2] for c in sorted_contours]  # Extract widths of contours
    print("Widths", widths)
    dominant_width = max(widths)
    print(f"Dominant width (highest mode): {dominant_width}")
    valid_width_min = dominant_width - 100
    valid_width_max = dominant_width + 100
    valid_contours = [
        c for c in sorted_contours if valid_width_min <= cv2.boundingRect(c)[2] <= valid_width_max
    ]
    print(f"Number of valid staves: {len(valid_contours)}")

    staves = []

    # Step 4: Process each valid contour
    for stave_index, contour in enumerate(valid_contours):
        try:
            x, y, w, h = cv2.boundingRect(contour)

            # Validate dimensions
            if w <= 0 or h <= 0:
                print(f"Skipping invalid contour with width={w}, height={h}.")
                continue

            # Clamp cropping indices
            y_start = y - 2
            y_end = y + h + 2
            x_start = max(0, x)
            x_end = min(image.shape[1], x + w)
            stave = image[y_start:y_end, x_start:x_end]

            if stave is None or stave.size == 0:
                print(f"Invalid stave dimensions for contour {stave_index}. Skipping.")
                continue

            # Process stave
            stave = crop_to_first_and_last_vertical_line(stave)
            split_point = find_row_with_max_white_pixels(stave, int(stave.shape[0] // 2), 15)
            left_hand = stave[split_point:, :]
            right_hand = stave[:split_point, :]

            # Trim the lower half
            trim_row = find_trim_row_dynamic(left_hand)
            left_hand_trimmed = left_hand[trim_row:, :]

            # Save the trimmed images
            left_output_path = os.path.join(output_dir, f"stave_{stave_index}_left.png")
            right_output_path = os.path.join(output_dir, f"stave_{stave_index}_right.png")

            # Avoid overwriting existing files
            counter = 0
            while os.path.exists(left_output_path):
                left_output_path = os.path.join(output_dir, f"stave_{stave_index}_left_{counter}.png")
                counter += 1
            counter = 0
            while os.path.exists(right_output_path):
                right_output_path = os.path.join(output_dir, f"stave_{stave_index}_right_{counter}.png")
                counter += 1

            cv2.imwrite(left_output_path, left_hand_trimmed)
            cv2.imwrite(right_output_path, right_hand)
            staves.append((left_output_path, right_output_path))

        except Exception as e:
            print(f"Error processing contour {stave_index}: {e}")
            continue

    return staves





def preprocess_image(image):
    """
    Preprocesses an image to enhance black vertical barlines for reliable detection.
    """
    # Step 1: Adaptive Thresholding
    binary_image = cv2.adaptiveThreshold(
        image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV, blockSize=11, C=2
    )

    # Step 2: Enhance Vertical Lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))  # Tall and narrow kernel
    vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel)

    # Step 3: Noise Removal (with sigmaX)
    denoised_image = cv2.GaussianBlur(vertical_lines, (5, 5), 0)

    # Step 4: Edge Detection
    edges = cv2.Canny(denoised_image, 50, 150)

    return edges


def crop_to_first_and_last_vertical_line(image):
    """
    Crops the original image to include only the region between the first and last vertical barlines.
    Uses a preprocessed version of the image to find the barlines.

    Parameters:
    - image: Grayscale image of the stave.

    Returns:
    - cropped_image: Image cropped between the first and last vertical barlines, or the original image.
    """
    # Step 1: Preprocess the image to enhance vertical barlines
    preprocessed_image = preprocess_image(image)

    # Step 2: Detect vertical lines and find contours
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize min_x and max_x
    min_x = image.shape[1]  # Initialize to the far right (maximum width)
    max_x = 0  # Initialize to the far left (minimum width)

    # Identify the leftmost and rightmost vertical lines
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out small contours that might be noise
        if h > 0.5 * image.shape[0]:
            if x < min_x:
                min_x = x
            if x + w > max_x:
                max_x = x + w

    # Step 3: Validate and crop the original image
    cropped_image = image[:, min_x + 5:max_x - 5]
    if image.shape[0] >= 50 and image.shape[1] >= 400:
        return cropped_image
    else:
        print("Invalid size of image.")


def quantize_image(image, num_colors):
    """
    Quantizes a grayscale or color image to a specified number of colors.
    Clamps the first range to 0 and the last range to 255.

    Parameters:
    - image: Input image (color or grayscale).
    - num_colors: Number of colors to reduce the image to.
    - save_path: Path to save the processed image (optional).

    Returns:
    - processed_image: Grayscale image with quantized values.
    """
    # Step 1: Convert the image to grayscale if it is not already
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Step 2: Define the intensity ranges
    intensity_range = 256 // num_colors  # Calculate the range for each quantized level

    # Step 3: Initialize the processed image
    processed_image = np.zeros_like(grayscale_image)

    # Step 4: Apply quantization
    for i in range(num_colors):
        lower_bound = i * intensity_range
        upper_bound = (i + 1) * intensity_range

        if i == 0:  # First range: Clamp to 0
            processed_image[(grayscale_image >= lower_bound) & (grayscale_image < upper_bound)] = 0
        elif i == num_colors - 1:  # Last range: Clamp to 255
            processed_image[(grayscale_image >= lower_bound) & (grayscale_image <= upper_bound)] = 255
        else:  # Middle ranges: Use midpoints
            midpoint = lower_bound + intensity_range // 2
            processed_image[(grayscale_image >= lower_bound) & (grayscale_image < upper_bound)] = midpoint
    return processed_image


def find_white_pixel_rows(image):
    """
    Finds rows in a binary image that contain only white pixels.

    Parameters:
    - image: Grayscale binary image (white=255, black=0).

    Returns:
    - List of row indices containing only white pixels.
    """
    # Ensure the image is binary
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Calculate the row sums
    row_sums = np.sum(binary == 255, axis=1)  # Count white pixels per row

    # A row with all white pixels will have a sum equal to the image width
    all_white_value = binary.shape[1]
    white_pixel_rows = np.where(row_sums == all_white_value)[0]

    return white_pixel_rows

def find_row_with_max_white_pixels(image, middle_row, search_range):
    """
    Finds the row with the maximum number of white pixels near the middle of the image.

    Parameters:
    - image: Grayscale binary image (white=255, black=0).
    - middle_row: The approximate middle row of the image.
    - search_range: The range above and below the middle row to search.

    Returns:
    - best_row: The row index with the maximum number of white pixels near the middle.
    """
    # Ensure the image is binary
    quantized_image = quantize_image(image,num_colors=4)

    # Define the search region
    start_row = max(0, middle_row - search_range)
    end_row = min(image.shape[0], middle_row + search_range)

    # Analyze pixel density in the search region
    best_row = middle_row
    max_white_pixels = 0

    for row in range(start_row, end_row):
        white_pixel_count = np.sum(quantized_image[row, :] == 255)  # Count white pixels in the row
        if white_pixel_count > max_white_pixels:
            max_white_pixels = white_pixel_count
            best_row = row

    # Step 4: Fallback in case no valid row is found
    if max_white_pixels == 0:
        print("No significant white pixel row found. Returning middle_row.")
        return middle_row

    print(f"Best Row: {best_row} Middle Row: {middle_row} Max White Pixels: {max_white_pixels}")
    return best_row

def find_stave_top(image, max_rows=50, min_black_pixels=10):
    """
    Finds the first row with significant black pixel density to determine the top of the stave area.

    Parameters:
    - image: Grayscale binary image (white=255, black=0).
    - max_rows: Maximum number of rows to scan from the top.
    - min_black_pixels: Minimum number of black pixels in a row to consider it part of the stave.

    Returns:
    - top_row: The row index of the first significant stave line or note.
    """
    # Ensure the image is binary
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Scan rows from top to bottom within the max_rows limit
    for row in range(min(max_rows, image.shape[0])):
        black_pixel_count = np.sum(binary[row, :] == 0)  # Count black pixels in the row
        if black_pixel_count >= min_black_pixels:
            return row

    # Fallback to the first row if no valid row is found
    return 0

def find_trim_row_dynamic(image):
    """
    Dynamically finds the ideal row to trim unnecessary white space from the top,
    based on the largest change in black pixel count.

    Parameters:
    - image: Grayscale binary image (white=255, black=0).

    Returns:
    - trim_row: The row index where trimming should start.
    """
    # Ensure the image is binary
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Step 2: Analyze rows from the top to the first stave line
    trim_row = 0
    previous_black_pixel_count = np.sum(binary[0, :] == 0)  # Black pixels in the first row

    # Iterate through rows to find the first significant change
    for row in range(binary.shape[0]):
        current_black_pixel_count = np.sum(binary[row, :] == 0)
        print("Current black pixel count",current_black_pixel_count)
        if abs(current_black_pixel_count - previous_black_pixel_count) > 1:
            trim_row = row
            break
        previous_black_pixel_count = current_black_pixel_count

    return trim_row - 1

def predict(image_path, model_path, voc_file):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.InteractiveSession()
    dict_list = open(voc_file).read().splitlines()
    int2word = {i: word for i, word in enumerate(dict_list)}

    saver = tf.compat.v1.train.import_meta_graph(model_path)
    saver.restore(sess, model_path[:-5])
    graph = tf.compat.v1.get_default_graph()
    input = graph.get_tensor_by_name("model_input:0")
    seq_len = graph.get_tensor_by_name("seq_lengths:0")
    rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
    height_tensor = graph.get_tensor_by_name("input_height:0")
    width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
    logits = tf.compat.v1.get_collection("logits")[0]
    WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])
    decoded, _ = tf.compat.v1.nn.ctc_greedy_decoder(logits, seq_len)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = resize(image, HEIGHT)
    image = normalize(image)
    image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)
    seq_lengths = [image.shape[2] / WIDTH_REDUCTION]
    prediction = sess.run(decoded, feed_dict={input: image, seq_len: seq_lengths, rnn_keep_prob: 1.0})
    semantic_list = [int2word[w] for w in sparse_tensor_to_strs(prediction)[0]]
    return semantic_list
