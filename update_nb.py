import json
import os

notebook_path = "d:/iam-handwriting-explainer/validate_samples.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update pip install cell
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "!pip install" in src:
            cell["source"] = ["!pip install -q transformers torch pillow groq wordninja jiwer datasets matplotlib"]
            break

# Add new cells at the end
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Performance Analysis\n",
            "This section runs the TrOCR pipeline against a larger batch of 30 samples from the `Teklia/IAM-line` dataset to compute Word Error Rate (WER) and Character Error Rate (CER), and analyzes the distribution of Groq confidence scores."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from datasets import load_dataset\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "\n",
            "print(\"Loading 30 samples from Teklia/IAM-line dataset...\")\n",
            "# Use streaming to avoid downloading the entire large dataset\n",
            "dataset = load_dataset(\"Teklia/IAM-line\", split=\"train\", streaming=True)\n",
            "\n",
            "batch_samples = []\n",
            "for i, item in enumerate(dataset):\n",
            "    batch_samples.append({\n",
            "        'image': item['image'].convert('RGB'),\n",
            "        'text': item['text'].strip()\n",
            "    })\n",
            "    if len(batch_samples) >= 30:\n",
            "        break\n",
            "\n",
            "print(f\"Successfully loaded {len(batch_samples)} samples.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "wers = []\n",
            "cers = []\n",
            "\n",
            "confidence_counts = {\"HIGH\": 0, \"MEDIUM\": 0, \"LOW\": 0}\n",
            "hallucination_overrides = 0\n",
            "total_high_claims = 0\n",
            "\n",
            "print(\"Running batch evaluation...\")\n",
            "for idx, sample in enumerate(batch_samples):\n",
            "    ground_truth = sample['text']\n",
            "    img = sample['image']\n",
            "    \n",
            "    # 1. Run TrOCR\n",
            "    raw_ocr = app.transcribe(img)\n",
            "    \n",
            "    # Calculate Error Rates\n",
            "    try:\n",
            "        wer = jiwer.wer(ground_truth.lower(), raw_ocr.lower())\n",
            "        cer = jiwer.cer(ground_truth.lower(), raw_ocr.lower())\n",
            "        wers.append(wer)\n",
            "        cers.append(cer)\n",
            "    except:\n",
            "        pass # Skip empty strings or jiwer failures\n",
            "        \n",
            "    # 2. Run Groq Explanation\n",
            "    explanation = app.explain(raw_ocr)\n",
            "    \n",
            "    # 3. Parse Confidence and Overrides\n",
            "    # We can detect an override if the output contains the specific override warning string\n",
            "    is_overridden = \"OVERRIDDEN from HIGH\" in explanation\n",
            "    \n",
            "    if is_overridden:\n",
            "        hallucination_overrides += 1\n",
            "        total_high_claims += 1\n",
            "        confidence_counts[\"MEDIUM\"] += 1 # It gets downgraded to MEDIUM\n",
            "    else:\n",
            "        if \"**Confidence:** HIGH\" in explanation:\n",
            "            confidence_counts[\"HIGH\"] += 1\n",
            "            total_high_claims += 1\n",
            "        elif \"**Confidence:** MEDIUM\" in explanation:\n",
            "            confidence_counts[\"MEDIUM\"] += 1\n",
            "        elif \"**Confidence:** LOW\" in explanation:\n",
            "            confidence_counts[\"LOW\"] += 1\n",
            "            \n",
            "print(\"Batch evaluation complete!\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "avg_wer = np.mean(wers) if wers else 0\n",
            "avg_cer = np.mean(cers) if cers else 0\n",
            "word_accuracy = max(0.0, 1.0 - avg_wer)\n",
            "char_accuracy = max(0.0, 1.0 - avg_cer)\n",
            "\n",
            "catch_rate = (hallucination_overrides / total_high_claims) if total_high_claims > 0 else 0\n",
            "\n",
            "print(\"=\"*40)\n",
            "print(\"SUMMARY STATISTICS (30 Samples)\")\n",
            "print(\"=\"*40)\n",
            "print(f\"Average Word Error Rate (WER): {avg_wer:.2%}\")\n",
            "print(f\"Average Char Error Rate (CER): {avg_cer:.2%}\")\n",
            "print(f\"Overall Word Accuracy:         {word_accuracy:.2%}\")\n",
            "print(f\"Overall Char Accuracy:         {char_accuracy:.2%}\")\n",
            "print(\"-\"*40)\n",
            "print(f\"Groq 'HIGH' Claims:            {total_high_claims}\")\n",
            "print(f\"Hallucination Overrides:       {hallucination_overrides}\")\n",
            "print(f\"Hallucination Catch Rate:      {catch_rate:.2%}\")\n",
            "print(\"=\"*40)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create Visualizations\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Plot 1: Error Rates\n",
            "metrics = ['WER', 'CER']\n",
            "values = [avg_wer, avg_cer]\n",
            "bars1 = ax1.bar(metrics, values, color=['#ef4444', '#f97316'])\n",
            "ax1.set_title('Average Error Rates', fontsize=14, pad=15)\n",
            "ax1.set_ylabel('Error Rate', fontsize=12)\n",
            "ax1.set_ylim(0, max(max(values) * 1.2, 0.1) if values else 1)\n",
            "\n",
            "# Add percentage labels on bars\n",
            "for bar in bars1:\n",
            "    height = bar.get_height()\n",
            "    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,\n",
            "            f'{height:.1%}', ha='center', va='bottom', fontsize=11)\n",
            "\n",
            "# Plot 2: Confidence Distribution\n",
            "labels = ['HIGH', 'MEDIUM', 'LOW']\n",
            "counts = [confidence_counts['HIGH'], confidence_counts['MEDIUM'], confidence_counts['LOW']]\n",
            "bars2 = ax2.bar(labels, counts, color=['#22c55e', '#eab308', '#ef4444'])\n",
            "ax2.set_title('Groq Final Confidence Distribution', fontsize=14, pad=15)\n",
            "ax2.set_ylabel('Number of Samples', fontsize=12)\n",
            "ax2.set_ylim(0, max(max(counts) * 1.2, 5) if counts else 10)\n",
            "\n",
            "# Add count labels on bars\n",
            "for bar in bars2:\n",
            "    height = bar.get_height()\n",
            "    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,\n",
            "            f'{int(height)}', ha='center', va='bottom', fontsize=11)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

nb["cells"].extend(new_cells)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated notebook.")
