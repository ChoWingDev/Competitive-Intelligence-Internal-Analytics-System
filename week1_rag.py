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

# Run it if executed as a script
if __name__ == "__main__":
    docs = load_and_parse_pdfs()
    vectorstore = create_vectorstore(docs)