import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from a .env file
load_dotenv()

def load_and_parse_pdfs(folder="data/reports"):
    """
    Load all PDF files from `folder`, parse them with unstructured.partition.pdf,
    convert parsed elements with text into langchain_core Document objects,
    and return the list of Documents.
    """
    all_docs = []  # list to accumulate Document objects
    
    # Iterate over files in the folder
    for filename in os.listdir(folder):
        # Only process files with .pdf extension
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder, filename)
            print(f"Parsing: {filename}")  # log of which file is being parsed at the moment
            
            # Use unstructured's partition_pdf to split the PDF into elements.
            # - strategy="hi_res" uses a higher-resolution layout strategy.
            # - infer_table_structure attempts to reconstruct table structure.
            # - extract_image_block_types includes images and tables as blocks.
            elements = partition_pdf(
                filename=filepath,
                strategy="hi_res",
                infer_table_structure=True,
                extract_image_block_types=["Image", "Table"]
            )
            
            # Convert each parsed element that contains non-empty text into a Document
            for element in elements:
                # If empty text, skip
                if element.text and element.text.strip():
                    metadata = {
                        "source": filename,  # original file name
                        # Try to get page number from element.metadata, defaulting to None if not found
                        "page": getattr(element.metadata, "page_number", None),
                        "element_type": element.category,  # type/category of element
                        "industry": "ecommerce"
                    }
                    # Create a langchain_core Document with the element text and metadata
                    all_docs.append(Document(
                        page_content=element.text.strip(),
                        metadata=metadata
                    ))
        
        break # only for testing: stops after the first file (remove in production)
    
    # Summary log and return collected documents
    print(f"Total elements extracted: {len(all_docs)}")
    return all_docs

def create_vectorstore(documents):
    """
    Two-level (parent -> child) chunking pipeline:
    - Parent splitter creates larger chunks that preserve broad context (e.g., full sections).
    - Child splitter breaks each parent chunk into smaller, high-recall subchunks that are
      better suited for retrieval and fine-grained embedding.
    Reasoning:
    - Embedding and retrieval work best when chunks are neither too large (diluted signal)
      nor too small (loss of context). Two-level chunking gives both: parent chunks keep
      context and allow retrieving semantically-relevant regions, while child chunks provide
      precise, high-quality passages to embed or use in downstream LLM prompts.
    - Strategy: split documents into parent chunks, then split each parent chunk into child
      chunks and keep child chunks as the final units stored/embedded.
    """
    # Parent splitter: larger chunks to preserve section-level context.
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    # Child splitter: smaller chunks focused on high recall and better fit for LLM prompts.
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    # Step 1: split original documents into parent chunks
    parent_docs = parent_splitter.split_documents(documents)
    print(f"Created {len(parent_docs)} parent chunks")

    # Step 2: for each parent chunk, create child chunks.
    # Preserve metadata from the parent and add an explicit reference to the parent's id/index
    # So, it's possible to map child chunks back to their parent (such as if we want for hierarchical retrieval).
    child_docs = []
    for idx, parent in enumerate(parent_docs):
        # child_splitter expects a list-like Document; pass single-item list and get list back
        children = child_splitter.split_documents([parent])
        for child in children:
            # Copy parent's metadata and add parent_index for traceability
            if parent.metadata:
                meta = dict(parent.metadata)
            else:
                meta = {}
            meta.update({
                "parent_index": idx,
                # keep original source if present
                "source": parent.metadata.get("source") if parent.metadata else None,
                # keep parent element_type if present
                "parent_element_type": parent.metadata.get("element_type") if parent.metadata else None,
                # keep parent industry if present (default to "ecommerce")
                "industry": parent.metadata.get("industry") if parent.metadata else "ecommerce"
            })
            child_docs.append(child.copy_with_metadata(meta))

    print(f"Created {len(child_docs)} child chunks (final units to embed)")

    # Initialize embeddings 
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Create Chroma vector store from child chunks.
    # Chroma stores dense vectors and associated metadata on disk,
    # providing fast nearest-neighbor search over embeddings for retrieval tasks.
    # collection_name groups related documents; allowing loading and querying this collection later.
    vectorstore = Chroma.from_documents(
        documents=child_docs,
        embedding=embeddings,
        collection_name="ecommerce_reports",
        persist_directory="./chroma_db"
    )

    print("Two-level vector store created successfully!")
    return vectorstore

# Run it if executed as a script
if __name__ == "__main__":
    docs = load_and_parse_pdfs()
    vectorstore = create_vectorstore(docs)