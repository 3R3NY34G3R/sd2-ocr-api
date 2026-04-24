import cv2
import pytesseract
from pytesseract import Output
import easyocr
import os
import csv

# Load EasyOCR ONCE (important)
reader = easyocr.Reader(['en'], gpu=False)

def run_easyocr(img_path):
    return reader.readtext(img_path)


def run_pytesseract(img_path, conf_threshold=30):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=Output.DICT)

    results = []
    for i, text in enumerate(data['text']):
        if int(data['conf'][i]) > conf_threshold and text.strip():
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            bbox = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
            results.append((bbox, text, float(data['conf'][i])))
    return results


def merge_results(*results):
    merged = []
    for r in results:
        merged.extend(r)
    return merged


def group_text_by_rows(results, y_threshold=20):
    rows = []
    for bbox, text, conf in results:
        y_center = sum(pt[1] for pt in bbox) / 4

        placed = False
        for row in rows:
            if abs(row['y'] - y_center) < y_threshold:
                row['items'].append((bbox, text, conf))
                placed = True
                break

        if not placed:
            rows.append({'y': y_center, 'items': [(bbox, text, conf)]})

    return rows


def sort_row_items(row):
    return sorted(row['items'], key=lambda item: sum(pt[0] for pt in item[0]) / 4)


def extract_table(img_path, csv_name="output.csv"):
    easy = run_easyocr(img_path)
    tess = run_pytesseract(img_path)

    combined = merge_results(easy, tess)

    rows = sorted(group_text_by_rows(combined), key=lambda r: r['y'])

    table = []
    for row in rows:
        sorted_items = sort_row_items(row)
        cells = [text.strip() for _, text, _ in sorted_items if text.strip()]
        if cells:
            table.append(cells)

    os.makedirs("Output", exist_ok=True)
    csv_path = f"Output/{csv_name}"

    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        csv.writer(f).writerows(table)

    return table, csv_path