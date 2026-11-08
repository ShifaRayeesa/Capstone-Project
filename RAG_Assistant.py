import os
from pathlib import Path

# 1. SET YOUR TARGET DIRECTORY HERE
TARGET_FOLDER = Path("/data/policies")


def chunk_text(text, chunk_size=40, overlap=10):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break
        start += step
    return chunks


def process_documents():
    # 2. Get all .txt files from the targeted location
    txt_files = list(TARGET_FOLDER.glob("*.txt"))
    print(txt_files)
    for file in TARGET_FOLDER.iterdir():
        print(file)
    if not txt_files:
        print(f"No .txt files found in: {TARGET_FOLDER.resolve()}")
        return

    all_chunks = []

    for file_path in txt_files:
        try:
            # pathlib handles opening files cleanly using open()
            with file_path.open('r', encoding='utf-8') as f:
                content = f.read()

            file_chunks = chunk_text(content)

            for idx, chunk in enumerate(file_chunks):
                all_chunks.append({
                    "source": file_path.name,  # Saves just the filename (e.g., 'doc1.txt')
                    "chunk_id": idx + 1,
                    "text": chunk
                })

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    print(f"Processed {len(txt_files)} file(s) into {len(all_chunks)} chunks total.\n")
    for item in all_chunks[:3]:
        print(f"Source: {item['source']} (Chunk {item['chunk_id']})")
        print(f"Content: {item['text']}\n" + "-" * 40)


if __name__ == "__main__":
    process_documents()
