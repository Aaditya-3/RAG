# Standard library module to interact with the operating system (used here to parse file paths and extensions)
import os

# Standard library module for Regular Expressions (used here for pattern-based text cleaning)
import re

# Imports datetime (to grab the current timestamp) and timezone (to ensure the timestamp is in UTC, not local machine time)
from datetime import datetime, timezone

# Imports LangChain's loader classes:
# - DirectoryLoader: scans a folder and runs a chosen loader on every matching file
# - TextLoader: reads a plain .txt file into a single LangChain Document object
# - PyPDFLoader: uses the pypdf library to extract text from a PDF, creating one Document object per page
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)


# Defines a helper function that takes a raw text string and returns a cleaned text string
def clean_text(text: str) -> str:
    """Fix common PDF/text extraction noise without losing paragraph breaks."""

    # re.sub(pattern, replacement, string) searches for regex matches and replaces them.
    # r"[ \t]+" means: match 1 or more (+) spaces or tab characters (\t) in a row.
    # We do NOT use \s+ here because \s+ would also erase newlines (\n) and destroy paragraph structure.
    text = re.sub(r"[ \t]+", " ", text)

    # r"\n{3,}" means: match 3 or more consecutive newline characters.
    # Replaces them with "\n\n" so empty vertical gaps shrink into a single clean paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # .strip() removes any leftover leading/trailing spaces or newlines from the start and end of the string
    return text.strip()


# Defines the main ingestion function that accepts a directory path (e.g., "./docs")
def load_documents(folder_path: str):
    # Creates an empty Python list to collect all loaded Document objects from both .txt and .pdf files
    all_docs = []

    # Initializes a DirectoryLoader specifically for .txt files.
    # By default, DirectoryLoader uses UnstructuredFileLoader, so we explicitly override it with loader_cls.
    txt_loader = DirectoryLoader(
        folder_path,                          # Root directory to scan
        glob="**/*.txt",                      # Glob pattern: "**/" searches current folder + all nested subfolders; "*.txt" matches .txt files
        loader_cls=TextLoader,                # Tells DirectoryLoader to instantiate TextLoader(file_path, **loader_kwargs) for each matched file
        loader_kwargs={"encoding": "utf-8"},  # Extra keyword args forwarded directly to TextLoader so UTF-8 characters (like ₹, emojis, smart quotes) don't crash on Windows
        silent_errors=True,                   # If one file is corrupted or unreadable, skip/log it instead of crashing the whole script
    )

    # Calls .load() to actually read every matched .txt file into a list of Document objects,
    # then .extend() unpacks that list and appends each Document individually into all_docs
    all_docs.extend(txt_loader.load())

    # Initializes a second DirectoryLoader specifically for .pdf files
    pdf_loader = DirectoryLoader(
        folder_path,            # Same root directory to scan
        glob="**/*.pdf",        # Matches all .pdf files in the folder and any subfolders
        loader_cls=PyPDFLoader, # Instantiates PyPDFLoader(file_path) for each PDF, producing 1 Document per page
        silent_errors=True,     # Skips broken/encrypted PDFs rather than halting the pipeline
    )

    # Loads all PDF pages into a list of Document objects and appends them into all_docs
    all_docs.extend(pdf_loader.load())

    # Creates a new list that will hold only valid, non-empty, cleaned Documents
    cleaned_docs = []

    # Gets the current UTC date/time and formats it as a string like "2026-11-22" (stored once so every doc in this run shares the same ingestion date)
    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Loops through every loaded Document object one by one
    for doc in all_docs:
        # Overwrites the Document's raw text (page_content) with the cleaned version
        doc.page_content = clean_text(doc.page_content)

        # In Python, an empty string "" is falsy.
        # If cleaning left no text (e.g., blank page or scanned image-only PDF page), skip this Document completely
        if not doc.page_content:
            continue

        # Every loader stores the file path under doc.metadata["source"].
        # .get("source", "unknown") safely fetches it, falling back to "unknown" if the key is somehow missing.
        source_path = doc.metadata.get("source", "unknown")

        # os.path.basename("./docs/sub/notes.pdf") extracts just "notes.pdf" — cleaner for citations than a full OS path
        doc.metadata["filename"] = os.path.basename(source_path)

        # os.path.splitext("notes.pdf") splits the path into a 2-item tuple: ("notes", ".pdf").
        # [1] grabs the extension (".pdf") and .lower() normalizes ".PDF" -> ".pdf" for consistent filtering.
        doc.metadata["file_type"] = os.path.splitext(source_path)[1].lower()

        # Attaches today's UTC date string to metadata so you can filter or audit when a doc was indexed
        doc.metadata["ingested_at"] = ingested_at

        # Checks if "page" exists in metadata (PyPDFLoader adds it; TextLoader does not)
        if "page" in doc.metadata:
            # PyPDFLoader counts pages starting from 0 (0, 1, 2...).
            # Adding 1 stores a human-friendly page number (1, 2, 3...) for user-facing citations.
            doc.metadata["page_number"] = doc.metadata["page"] + 1

        # Appends the cleaned, metadata-enriched Document into our final list
        cleaned_docs.append(doc)

    # Returns the final list of clean Document objects to the caller
    return cleaned_docs


# Standard Python entry-point guard:
# Code inside this block runs ONLY when you execute `python module1_load.py` directly.
# If you import `load_documents` into Module 2's script later, this block will NOT auto-run.
if __name__ == "__main__":
    # Runs the loader on the "./docs" folder and stores the returned list of Documents
    docs = load_documents("./docs")

    # Prints how many total valid Documents/pages survived cleaning
    print(f"Total loaded documents/pages: {len(docs)}\n")

    # docs[:3] slices the first 3 Documents so the terminal isn't flooded.
    # enumerate(..., start=1) gives a 1-based counter (i = 1, 2, 3) alongside each Document object.
    for i, doc in enumerate(docs[:], start=1):
        # Prints a visual header for the current Document
        print(f"--- Document {i} ---")

        # Prints the full metadata dictionary (source, page, filename, file_type, ingested_at, page_number)
        print("Metadata:", doc.metadata)

        # doc.page_content[:200] grabs the first 200 characters.
        # .replace("\n", " ") flattens newlines into spaces just for a neat single-line terminal preview.
        print("Preview :", doc.page_content[:200].replace("\n", " "), "...\n")