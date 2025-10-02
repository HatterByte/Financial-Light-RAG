#!/usr/bin/env python3
"""
PDF Processor for Financial Documents
Handles PDF extraction and processing for the Financial RAG system
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import re
import PyPDF2
import pdfplumber

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinancialDocument:
    """Financial document with extracted content"""
    file_name: str
    file_path: str
    text_content: str
    total_pages: int
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class FinancialPDFProcessor:
    """Advanced PDF processor for financial documents"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
        
        # Financial document patterns for better extraction
        self.financial_patterns = {
            'currency': r'\$[\d,]+(?:\.\d{2})?',
            'percentage': r'\d+\.?\d*%',
            'dates': r'\b(?:Q[1-4]|quarter|fiscal year|FY)\s+\d{4}\b',
            'financial_terms': r'\b(?:revenue|profit|loss|earnings|ebitda|cash flow|margin)\b'
        }
    
    def process_pdf_file(self, file_path: str) -> Optional[FinancialDocument]:
        """Process a single PDF file"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            if file_path.suffix.lower() not in self.supported_extensions:
                logger.error(f"Unsupported file type: {file_path.suffix}")
                return None
            
            logger.info(f"Processing PDF: {file_path.name}")
            
            # Try pdfplumber first (better for financial documents)
            text_content = self._extract_with_pdfplumber(file_path)
            
            # Fallback to PyPDF2 if pdfplumber fails
            if not text_content or len(text_content.strip()) < 100:
                logger.warning(f"pdfplumber extraction poor, trying PyPDF2 for {file_path.name}")
                text_content = self._extract_with_pypdf2(file_path)
            
            if not text_content or len(text_content.strip()) < 50:
                logger.error(f"Failed to extract meaningful content from {file_path.name}")
                return None
            
            # Get page count
            total_pages = self._get_page_count(file_path)
            
            # Clean and enhance text
            cleaned_text = self._clean_text(text_content)
            
            # Create document object
            doc = FinancialDocument(
                file_name=file_path.name,
                file_path=str(file_path),
                text_content=cleaned_text,
                total_pages=total_pages,
                metadata={
                    'extraction_method': 'pdfplumber',
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                    'char_count': len(cleaned_text),
                    'word_count': len(cleaned_text.split()),
                    'financial_indicators': self._analyze_financial_content(cleaned_text)
                }
            )
            
            logger.info(f"Successfully processed {file_path.name}: {total_pages} pages, {len(cleaned_text):,} chars")
            return doc
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def process_pdf_directory(self, directory_path: str) -> List[FinancialDocument]:
        """Process all PDF files in a directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        documents = []
        for pdf_file in pdf_files:
            doc = self.process_pdf_file(pdf_file)
            if doc:
                documents.append(doc)
        
        logger.info(f"Successfully processed {len(documents)} out of {len(pdf_files)} PDF files")
        return documents
    
    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """Extract text using pdfplumber (better for tables and financial data)"""
        try:
            text_parts = []
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                        
                        # Extract tables if present
                        tables = page.extract_tables()
                        for table in tables:
                            if table:
                                # Convert table to text
                                table_text = self._table_to_text(table)
                                if table_text:
                                    text_parts.append(f"\n[TABLE]\n{table_text}\n[/TABLE]\n")
                    
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1} from {file_path.name}: {e}")
                        continue
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {file_path.name}: {e}")
            return ""
    
    def _extract_with_pypdf2(self, file_path: Path) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        try:
            text_parts = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1} from {file_path.name}: {e}")
                        continue
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {file_path.name}: {e}")
            return ""
    
    