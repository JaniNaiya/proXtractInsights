from PIL import Image
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


"""pdf 2 image

"""

"""convert the pdf to image

"""

from pdf2image import convert_from_path

images = convert_from_path('3.pdf')

for i in range(len(images)):
    images[i].save('pdf3new/page' + str(i) + '.jpg', 'JPEG')

from paddleocr import PaddleOCR, draw_ocr
import paddle

from paddleocr import PaddleOCR
import cv2
import matplotlib.pyplot as plt
import os

# Initialize PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en', gpu_mem=500, det_limit_side_len=640)

# Path to the folder containing images
folder_path = 'pdf3new'

# Define keywords for identifying the header of the desired table
header_keyword = "Adjusted Protein Efficiency Ratio"
table_keywords = ["in Vitro Protein Digestibility-Corrected", "%TPD", "adj PER"]

# Iterate through all files in the folder
table_found = False
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process image files only
        image_path = os.path.join(folder_path, filename)
        print(f"Processing file: {image_path}")

        # Preprocess the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        preprocessed_image_path = 'preprocessed_image.jpg'
        cv2.imwrite(preprocessed_image_path, binary_image)

        # Perform OCR on the preprocessed image
        ocr_results = ocr.ocr(preprocessed_image_path)

        # Step 1: Locate the table header
        header_region = None
        for result in ocr_results[0]:
            text = result[1][0]
            if header_keyword.lower() in text.lower():
                header_region = result[0]  # Save the bounding box of the header
                print(f"Found header in file {filename}: {text}")
                table_found = True
                break

        # Step 2: If header is found, expand the bounding box to include all rows
        if header_region:
            x_min = int(header_region[0][0])
            y_min = int(header_region[0][1])
            x_max = int(header_region[2][0])
            y_max = int(header_region[2][1])

            # Include rows below the header
            for result in ocr_results[0]:
                box = result[0]
                text = result[1][0]
                # Check if the row lies below the header and matches a table keyword
                if box[0][1] > y_max:  # Below the header
                    for keyword in table_keywords:
                        if keyword.lower() in text.lower():
                            # Expand bounding box to include the row
                            x_min = min(x_min, int(box[0][0]))
                            y_max = max(y_max, int(box[2][1]))
                            x_max = max(x_max, int(box[2][0]))

            # Step 3: Crop and save the desired table
            table_crop = binary_image[y_min:y_max, x_min:x_max]
            output_path = f'desired_table_{filename}'
            cv2.imwrite(output_path, table_crop)
            print(f"Cropped desired table saved as '{output_path}'")

            # Visualize the cropped table
            plt.imshow(table_crop, cmap='gray')
            plt.axis('off')
            plt.show()

            # Stop the loop as the table is found
            break

if not table_found:
    print("Table header not found in any file!")

image_path = output_path
print(output_path)
image_cv = cv2.imread(image_path)
image_height = image_cv.shape[0]
image_width = image_cv.shape[1]
output = ocr.ocr(image_path)[0]

boxes = [line[0] for line in output]
texts = [line[1][0] for line in output]
probabilities = [line[1][1] for line in output]

image_boxes = image_cv.copy()

for box, text in zip(boxes, texts):
    cv2.rectangle(image_boxes, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])), (0, 0, 255), 1)
    cv2.putText(image_boxes, text, (int(box[0][0]), int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 0, 0), 1)

cv2.imwrite('detections.jpg', image_boxes)

im = image_cv.copy()

horiz_boxes = []
vert_boxes = []

for box in boxes:
    x_h, x_v = 0, int(box[0][0])
    y_h, y_v = int(box[0][1]), 0
    width_h, width_v = image_width, int(box[2][0] - box[0][0])
    height_h, height_v = int(box[2][1] - box[0][1]), image_height

    horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
    vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])

    cv2.rectangle(im, (x_h, y_h), (x_h + width_h, y_h + height_h), (0, 0, 255), 1)
    cv2.rectangle(im, (x_v, y_v), (x_v + width_v, y_v + height_v), (0, 255, 0), 1)

cv2.imwrite('horiz_vert.jpg', im)

import tensorflow as tf

horiz_out = tf.image.non_max_suppression(
    horiz_boxes,
    probabilities,
    max_output_size=1000,
    iou_threshold=0.1,
    score_threshold=float('-inf'),
    name=None
)

import numpy as np

horiz_lines = np.sort(np.array(horiz_out))
print(horiz_lines)

im_nms = image_cv.copy()

for val in horiz_lines:
    cv2.rectangle(im_nms, (int(horiz_boxes[val][0]), int(horiz_boxes[val][1])),
                  (int(horiz_boxes[val][2]), int(horiz_boxes[val][3])), (0, 0, 255), 1)

cv2.imwrite('im_nms.jpg', im_nms)

vert_out = tf.image.non_max_suppression(
    vert_boxes,
    probabilities,
    max_output_size=1000,
    iou_threshold=0.1,
    score_threshold=float('-inf'),
    name=None
)

vert_lines = np.sort(np.array(vert_out))
print(vert_lines)

cv2.imwrite('im_nms.jpg', im_nms)

out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]
print(np.array(out_array).shape)
print(out_array)

unordered_boxes = []

for i in vert_lines:
    print(vert_boxes[i])
    unordered_boxes.append(vert_boxes[i][0])

ordered_boxes = np.argsort(unordered_boxes)
print(ordered_boxes)


def intersection(box_1, box_2):
    return [box_2[0], box_1[1], box_2[2], box_1[3]]


def iou(box_1, box_2):
    x_1 = max(box_1[0], box_2[0])
    y_1 = max(box_1[1], box_2[1])
    x_2 = min(box_1[2], box_2[2])
    y_2 = min(box_1[3], box_2[3])

    inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
    if inter == 0:
        return 0

    box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
    box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

    return inter / float(box_1_area + box_2_area - inter)


for i in range(len(horiz_lines)):
    for j in range(len(vert_lines)):
        resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])

        for b in range(len(boxes)):
            the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
            if (iou(resultant, the_box) > 0.1):
                out_array[i][j] = texts[b]

out_array = np.array(out_array)

import pandas as pd

pd.DataFrame(out_array).to_csv('sample.csv', index=False)

import re

# Load the CSV with possible multi-row headers
file_path = 'sample.csv'

# List of possible header rows
header_rows = [0, 1, 2, 3, 4]
filtered_df = None

# Try to find the correct header row dynamically
for header_row in header_rows:
    try:
        # Attempt to load the CSV with the current header row
        df = pd.read_csv(file_path, header=header_row)

        # Ensure column names are strings
        df.columns = df.columns.astype(str)

        # Check for expected columns like 'TPD' or 'IVPD' in any variation
        if any("TPD" in col or "IVPD" in col for col in df.columns):
            print(f"Valid header found at row {header_row}")

            # Clean 'TPDb' column if present
            if any("TPD" in col for col in df.columns):
                tpd_col = [col for col in df.columns if "TPD" in col][0]
                df[tpd_col] = df[tpd_col].astype(str).str.extract(r'(\d+\.\d+)')[0]

            # Clean 'IVPDc' column if present
            if any("IVPD" in col for col in df.columns):
                ivpd_col = [col for col in df.columns if "IVPD" in col][0]
                df[ivpd_col] = df[ivpd_col].astype(str).str.extract(r'(\d+\.\d+)')[0]
            # Clean 'IVPD' column if present
            if any("IVPD" in col for col in df.columns):
                # Identify the column with "IVPD" in its name
                ivpd_col = [col for col in df.columns if "IVPD" in col][0]

                # Extract the first numeric value from each cell in the column
                df[ivpd_col] = df[ivpd_col].astype(str).str.extract(r'(\d+\.\d+)')[0]

                # Find the column that contains 'casein' in any row
            for col in df.columns:
                if "casein" in df[col].astype(str).values:
                    casein_col = col
                    break

            # Extract the first two columns and the cleaned 'TPD' and 'IVPD' columns
            filtered_columns = [casein_col, tpd_col, ivpd_col]
            filtered_df = df[filtered_columns]

            # Stop searching after finding a valid header
            break
    except Exception as e:
        print(f"Failed to process with header row {header_row}: {e}")

# Save the filtered DataFrame if a valid header was found
if filtered_df is not None:
    filtered_df.to_csv('filtered_sample.csv', index=False)
    print("Filtered data saved to 'filtered_sample.csv'")
else:
    print("No valid header found or no valid data to process.")