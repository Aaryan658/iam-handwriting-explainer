import json

notebook_path = "d:/iam-handwriting-explainer/validate_samples.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "Running Florence-2..." in src and "app.correct_spaces" in src:
            new_source = []
            for line in cell["source"]:
                if "Running Florence-2..." in line:
                    new_source.append("    print(\"Running TrOCR...\")\n")
                elif "corrected_ocr = app.correct_spaces(raw_ocr)" in line:
                    new_source.append("    corrected_ocr = raw_ocr\n")
                elif "print(f\"[SPACE CORRECTED]: {corrected_ocr}\")" in line:
                    continue # Drop this line
                elif "# Validate Florence-2 + Groq Pipeline" in line:
                    new_source.append(line.replace("Florence-2", "TrOCR"))
                else:
                    new_source.append(line)
            cell["source"] = new_source

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        if "Validate Florence-2" in src:
            new_source = []
            for line in cell["source"]:
                if "Validate Florence-2" in line:
                    new_source.append(line.replace("Florence-2", "TrOCR"))
                elif "spacing corrections" in line:
                    new_source.append(line.replace("spacing corrections and token-overlap validations.", "token-overlap validations."))
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated validation notebook to fix TrOCR references.")
