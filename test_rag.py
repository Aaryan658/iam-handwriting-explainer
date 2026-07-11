import sys
sys.path.append(r"D:\Projects\Sethu-RAG")
from generate import answer

q = "financial assistance for women entrepreneurs"
a, c = answer(q)

with open("test_rag_output.txt", "w", encoding="utf-8") as f:
    f.write("=== ANSWER ===\n")
    f.write(a)
    f.write("\n=== CITATIONS ===\n")
    f.write(str(c))
    f.write("\n")
