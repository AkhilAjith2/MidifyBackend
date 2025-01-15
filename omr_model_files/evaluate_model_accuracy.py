import os
from collections import Counter

import cv2
import tensorflow as tf
import numpy as np
from ctc_utils import resize,normalize,sparse_tensor_to_strs,levenshtein
import matplotlib.pyplot as plt



def predict(image_path, model_path, voc_file):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    dict_list = open(voc_file).read().splitlines()
    int2word = {i: word for i, word in enumerate(dict_list)}

    with tf.compat.v1.Session() as sess:
        # Load the model
        saver = tf.compat.v1.train.import_meta_graph(model_path)
        saver.restore(sess, model_path[:-5])
        graph = tf.compat.v1.get_default_graph()

        # Extract tensors
        input = graph.get_tensor_by_name("model_input:0")
        seq_len = graph.get_tensor_by_name("seq_lengths:0")
        rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
        height_tensor = graph.get_tensor_by_name("input_height:0")
        width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
        logits = tf.compat.v1.get_collection("logits")[0]

        # Get static values
        WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

        # Preprocess the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = resize(image, HEIGHT)
        image = normalize(image)
        image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)
        seq_lengths = [image.shape[2] / WIDTH_REDUCTION]

        # Run predictions
        decoded, _ = tf.compat.v1.nn.ctc_greedy_decoder(logits, seq_len)
        prediction = sess.run(decoded, feed_dict={input: image, seq_len: seq_lengths, rnn_keep_prob: 1.0})

        # Decode the predictions
        semantic_list = [int2word[w] for w in sparse_tensor_to_strs(prediction)[0]]

    return semantic_list

def evaluate_model(test_file, corpus_dir, model_path, voc_file):
    """
        Evaluate the model's predictions with a tolerance for mismatches.
        Args:
            test_file (str): Path to the test.txt file.
            corpus_dir (str): Path to the corpus directory.
            model_path (str): Path to the model checkpoint file.
            voc_file (str): Path to the vocabulary file.
        """
    with open(test_file, 'r') as file:
        test_subdirectories = file.read().splitlines()

    mismatch_threshold = 1

    total = 0
    errors = 0
    mismatch_counts_above_threshold = []

    for subdir in test_subdirectories:
        subdir_path = os.path.join(corpus_dir, subdir)
        image_path = os.path.join(subdir_path, f"{subdir}.png")
        semantic_path = os.path.join(subdir_path, f"{subdir}.semantic")

        if not os.path.exists(image_path) or not os.path.exists(semantic_path):
            print(f"Missing files for {subdir}, skipping.")
            continue

        # Get model predictions
        predicted_semantics = predict(image_path, model_path, voc_file)

        # Read the ground truth from the .semantic file
        with open(semantic_path, 'r') as semantic_file:
            ground_truth = semantic_file.read().strip().split("\t")

        # Calculate mismatches using Levenshtein distance
        mismatches = levenshtein(predicted_semantics, ground_truth)

        if mismatches > mismatch_threshold:
            errors += 1
            mismatch_counts_above_threshold.append(mismatches)
            print(f"Mismatch for {subdir}:")
            print(f"Predicted: {predicted_semantics}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Mismatches: {mismatches}")

        total += 1

    # Calculate and print accuracy
    accuracy = (total - errors) / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%} ({total - errors}/{total})")
    print(f"Total Errors: {errors}/{total}")

    # Plot mismatch distribution for mismatches above the threshold
    if mismatch_counts_above_threshold:
        mismatch_distribution = Counter(mismatch_counts_above_threshold)
        x = list(mismatch_distribution.keys())
        y = list(mismatch_distribution.values())

        plt.figure(figsize=(10, 6))
        plt.bar(x, y, tick_label=x)
        plt.title(f"Distribution of Mismatches Above Threshold\nAccuracy: {accuracy:.2%}")
        plt.xlabel("Number of Mismatches")
        plt.ylabel("Frequency")
        plt.text(
            max(x) * 0.7, max(y) * 0.9,  # Position the accuracy text
            f"Accuracy: {accuracy:.2%}",
            fontsize=12,
            color="blue",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="blue")
        )
        plt.show()
    else:
        print("No mismatches exceeded the threshold.")



# Example usage
test_file = "C:/Users/David Abraham/PycharmProjects/Dill_OMR/tf-end-to-end/Data/test.txt"
corpus_dir = "C:/Users/David Abraham/Desktop/Semester 7/Thesis/Corpus"
model_path = r"C:\Users\David Abraham\PycharmProjects\Dill_OMR\tf-end-to-end\model-checkpoints\models-10-10.meta"
voc_file = "C:/Users/David Abraham/PycharmProjects/Dill_OMR/tf-end-to-end/Data/vocabulary_semantic.txt"

evaluate_model(test_file, corpus_dir, model_path, voc_file)