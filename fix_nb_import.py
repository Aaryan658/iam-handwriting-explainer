import json

notebook_path = "d:/iam-handwriting-explainer/validate_samples.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell that imports app
import_cell_index = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "import app" in "".join(cell["source"]):
        import_cell_index = i
        break

# Find the cell that sets the env var
env_cell_index = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "userdata.get('GROQ_API_KEY')" in "".join(cell["source"]):
        env_cell_index = i
        break

if import_cell_index != -1 and env_cell_index != -1:
    # We want to move the userdata injection block BEFORE import app
    # Actually, the simplest fix is to just move the entire env_cell BEFORE import_cell
    
    # Or, even simpler: just modify app.py dynamically or set os.environ before import app in the notebook.
    
    # Let's extract the env var injection logic from env_cell
    env_cell = nb["cells"][env_cell_index]
    
    # Let's create a new cell specifically for setting up environment variables
    # and put it BEFORE the import app cell.
    
    # But wait, env_cell has the SAMPLES list and the evaluation loop too!
    # Let's separate the env var injection logic out.
    
    source = "".join(env_cell["source"])
    lines = source.split("\n")
    
    env_lines = []
    loop_lines = []
    
    in_env = False
    for line in lines:
        if "# Automatically inject Colab secrets" in line:
            in_env = True
        
        if in_env:
            env_lines.append(line)
            if line.strip() == "os.environ['GROQ_API_KEY'] = 'YOUR_GROQ_API_KEY_HERE'":
                in_env = False
        else:
            loop_lines.append(line)
            
    # Now put env_lines into a new cell and insert it before import_cell
    env_cell_new = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in env_lines]
    }
    
    # Remove the env lines from the loop cell
    # Note: we need to handle the trailing newlines properly, but simple string joining is fine.
    nb["cells"][env_cell_index]["source"] = [line + "\n" for line in loop_lines if line.strip() or line == ""]
    
    # Insert env_cell_new before import_cell_index
    nb["cells"].insert(import_cell_index, env_cell_new)

    # Let's also do the same for the Performance analysis cell, wait, performance analysis cell doesn't import app, it uses it.
    
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated validation notebook to set env vars before importing app.")
