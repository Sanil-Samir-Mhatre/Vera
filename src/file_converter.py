import os
import io
import pandas as pd
import fitz # PyMuPDF
from docx import Document

def normalize_to_excel(uploaded_file, output_path: str = None) -> tuple[pd.DataFrame, str]:
    """
    ANY uploaded format (PDF/DOCX/XLSX/CSV/TXT) -> normalized .xlsx DataFrame and file path.
    Always writes the normalized result to a real .xlsx file AND returns it for on-screen st.dataframe display.
    """
    filename = getattr(uploaded_file, 'name', 'uploaded_document')
    ext = os.path.splitext(filename)[1].lower()
    
    # Read bytes if buffer, or read from file path if string
    if isinstance(uploaded_file, str):
        file_bytes = open(uploaded_file, 'rb').read()
    else:
        file_bytes = uploaded_file.read()
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
            
    df = None
    
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(io.BytesIO(file_bytes))
    elif ext == '.csv':
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif ext == '.pdf':
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        paragraphs = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            # Split by blocks or paragraphs
            blocks = text.split('\n\n')
            for b in blocks:
                cleaned = b.strip().replace('\n', ' ')
                if len(cleaned) > 10:
                    paragraphs.append({
                        'Page': page_num + 1,
                        'Paragraph': cleaned
                    })
        if not paragraphs:
            paragraphs = [{'Page': 1, 'Paragraph': 'Empty document'}]
        df = pd.DataFrame(paragraphs)
    elif ext == '.docx':
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = []
        for i, p in enumerate(doc.paragraphs):
            cleaned = p.text.strip()
            if len(cleaned) > 5:
                paragraphs.append({
                    'Paragraph_No': i + 1,
                    'Paragraph': cleaned
                })
        if not paragraphs:
            paragraphs = [{'Paragraph_No': 1, 'Paragraph': 'Empty document'}]
        df = pd.DataFrame(paragraphs)
    else: # Plain text or fallback
        text_content = file_bytes.decode('utf-8', errors='ignore')
        lines = [l.strip() for l in text_content.split('\n') if len(l.strip()) > 5]
        if not lines:
            lines = ['Empty document']
        df = pd.DataFrame({
            'Line_No': range(1, len(lines) + 1),
            'Text': lines
        })

    # Ensure output path
    if output_path is None:
        base_name = os.path.splitext(filename)[0]
        os.makedirs('temp_excel', exist_ok=True)
        output_path = os.path.join('temp_excel', f"{base_name}_normalized.xlsx")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    df.to_excel(output_path, index=False)
    return df, output_path
