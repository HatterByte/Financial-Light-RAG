import re
from typing import Dict, List, Any
from datetime import datetime
from tqdm import tqdm
from logger import logger
from data_models import TextChunk

class AdvancedFinancialChunker:
    """Advanced chunker with financial document understanding"""
    
    def __init__(self, 
                 base_chunk_size: int = 1200,
                 overlap_ratio: float = 0.15,
                 max_chunks_per_doc: int = 1000,
                 min_chunk_size: int = 100):
        
        self.base_chunk_size = base_chunk_size
        self.overlap_ratio = overlap_ratio
        self.max_chunks_per_doc = max_chunks_per_doc
        self.min_chunk_size = min_chunk_size
        
        # Financial document structure patterns
        self.section_patterns = {
            'executive_summary': r'(?i)\b(executive\s+summary|management\s+summary|key\s+highlights)\b',
            'business_overview': r'(?i)\b(business\s+overview|company\s+overview|business\s+description)\b',
            'financial_performance': r'(?i)\b(financial\s+performance|financial\s+results|financial\s+highlights)\b',
            'revenue_analysis': r'(?i)\b(revenue|net\s+sales|total\s+revenue|sales\s+performance)\b',
            'profitability': r'(?i)\b(profit|earnings|income|ebitda|operating\s+income)\b',
            'balance_sheet': r'(?i)\b(balance\s+sheet|statement\s+of\s+financial\s+position|assets\s+and\s+liabilities)\b',
            'cash_flow': r'(?i)\b(cash\s+flow|statement\s+of\s+cash\s+flows|cash\s+position)\b',
            'risk_factors': r'(?i)\b(risk\s+factors|principal\s+risks|risk\s+management|risk\s+assessment)\b',
            'market_analysis': r'(?i)\b(market\s+analysis|industry\s+analysis|competitive\s+landscape)\b',
            'outlook': r'(?i)\b(outlook|guidance|forecast|future\s+prospects|forward\s+looking)\b',
            'segment_analysis': r'(?i)\b(segment|division|business\s+unit|product\s+line)\b'
        }
        
        # Financial importance weights for different sections
        self.section_importance = {
            'financial_performance': 1.0,
            'revenue_analysis': 0.9,
            'profitability': 0.9,
            'balance_sheet': 0.8,
            'cash_flow': 0.8,
            'risk_factors': 0.7,
            'executive_summary': 0.8,
            'business_overview': 0.6,
            'market_analysis': 0.6,
            'outlook': 0.7,
            'segment_analysis': 0.6
        }
    
    def create_chunks(self, text: str, doc_id: str) -> List[TextChunk]:
        """Create sophisticated financial document chunks"""
        
        if not text or len(text.strip()) < self.min_chunk_size:
            logger.warning(f"Text too short for chunking: {len(text)} chars")
            return []
        
        logger.info(f"Advanced chunking for {doc_id}: {len(text):,} characters")
        
        # Step 1: Detect document structure
        structure = self._analyze_document_structure(text)
        logger.info(f"Detected structure: {len(structure['sections'])} sections")
        
        # Step 2: Create adaptive chunks based on content type
        chunks = self._create_adaptive_chunks(text, doc_id, structure)
        
        # Step 3: Post-process and validate chunks
        validated_chunks = self._validate_and_enhance_chunks(chunks, doc_id)
        
        logger.info(f"Created {len(validated_chunks)} chunks for {doc_id}")
        return validated_chunks
    
    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure and identify financial sections"""
        
        sections = []
        section_scores = {}
        
        # Find all section markers
        all_matches = []
        for section_type, pattern in self.section_patterns.items():
            for match in re.finditer(pattern, text):
                all_matches.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group().strip(),
                    'type': section_type,
                    'importance': self.section_importance.get(section_type, 0.5)
                })
        
        # Sort by position
        all_matches.sort(key=lambda x: x['start'])
        
        # Create sections with boundaries
        if all_matches:
            for i, match in enumerate(all_matches):
                section_start = match['start']
                
                # Find section end (next section or document end)
                if i + 1 < len(all_matches):
                    section_end = all_matches[i + 1]['start']
                else:
                    section_end = len(text)
                
                # Only include substantial sections
                section_length = section_end - section_start
                if section_length >= self.min_chunk_size:
                    sections.append({
                        'start': section_start,
                        'end': section_end,
                        'title': match['text'],
                        'type': match['type'],
                        'importance': match['importance'],
                        'length': section_length
                    })
                    section_scores[match['type']] = match['importance']
        
        # If no clear sections, create artificial ones based on content density
        if not sections:
            sections = self._create_artificial_sections(text)
        
        return {
            'sections': sections,
            'section_scores': section_scores,
            'total_length': len(text),
            'avg_section_length': sum(s['length'] for s in sections) / len(sections) if sections else 0
        }
    
    def _create_artificial_sections(self, text: str) -> List[Dict[str, Any]]:
        """Create artificial sections when no clear structure is found"""
        
        # Use paragraph breaks as section boundaries
        paragraphs = re.split(r'\n\s*\n', text)
        sections = []
        current_pos = 0
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) >= self.min_chunk_size:
                sections.append({
                    'start': current_pos,
                    'end': current_pos + len(paragraph),
                    'title': f"Section {i+1}",
                    'type': 'general',
                    'importance': 0.5,
                    'length': len(paragraph)
                })
            current_pos += len(paragraph) + 2  # Account for paragraph breaks
        
        return sections
    
    