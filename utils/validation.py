import sys
import os
from typing import Tuple, Optional
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# File size limits (in bytes)
MAX_PDF_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CSV_SIZE = 5 * 1024 * 1024   # 5MB
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

ALLOWED_PDF_EXTENSIONS = ['.pdf']
ALLOWED_CSV_EXTENSIONS = ['.csv']
ALLOWED_DOC_EXTENSIONS = ['.docx', '.doc']
ALLOWED_EXCEL_EXTENSIONS = ['.xlsx', '.xls']


def validate_file(file, file_type: str = "auto") -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file.
    Returns (is_valid, error_message)
    """
    try:
        if file is None:
            return False, "No file provided"
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            return False, f"File size ({file_size / 1024 / 1024:.2f}MB) exceeds maximum allowed size (10MB)"
        
        if file_size == 0:
            return False, "File is empty"
        
        # Check file extension
        filename = getattr(file, 'name', '')
        if not filename:
            return False, "Filename not available"
        
        ext = os.path.splitext(filename)[1].lower()
        
        if file_type == "auto":
            # Auto-detect based on extension
            if ext in ALLOWED_PDF_EXTENSIONS:
                if file_size > MAX_PDF_SIZE:
                    return False, f"PDF size exceeds maximum ({MAX_PDF_SIZE / 1024 / 1024}MB)"
                return True, None
            elif ext in ALLOWED_CSV_EXTENSIONS:
                if file_size > MAX_CSV_SIZE:
                    return False, f"CSV size exceeds maximum ({MAX_CSV_SIZE / 1024 / 1024}MB)"
                return True, None
            elif ext in ALLOWED_DOC_EXTENSIONS:
                return True, None
            elif ext in ALLOWED_EXCEL_EXTENSIONS:
                return True, None
            else:
                return False, f"Unsupported file type: {ext}. Supported: PDF, CSV, DOCX, XLSX"
        
        # Type-specific validation
        if file_type == "pdf" and ext not in ALLOWED_PDF_EXTENSIONS:
            return False, f"Expected PDF file, got {ext}"
        if file_type == "csv" and ext not in ALLOWED_CSV_EXTENSIONS:
            return False, f"Expected CSV file, got {ext}"
        
        return True, None
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_csv_structure(df) -> Tuple[bool, Optional[str]]:
    """Validate CSV structure."""
    try:
        if df is None or df.empty:
            return False, "CSV file is empty"
        
        if len(df.columns) == 0:
            return False, "CSV has no columns"
        
        if len(df) == 0:
            return False, "CSV has no rows"
        
        # Check for reasonable column count
        if len(df.columns) > 100:
            return False, "CSV has too many columns (>100)"
        
        return True, None
    except Exception as e:
        return False, f"CSV validation error: {str(e)}"

