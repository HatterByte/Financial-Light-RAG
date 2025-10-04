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
    
    def _create_adaptive_chunks(self, text: str, doc_id: str, structure: Dict[str, Any]) -> List[TextChunk]:
        """Create chunks with adaptive sizing based on content importance"""
        
        chunks = []
        chunk_index = 0
        
        with tqdm(structure['sections'], desc="Creating adaptive chunks", leave=False) as section_pbar:
            
            for section in structure['sections']:
                section_text = text[section['start']:section['end']].strip()
                
                if len(section_text) < self.min_chunk_size:
                    section_pbar.update(1)
                    continue
                
                # Adaptive chunk size based on importance
                importance = section['importance']
                adaptive_chunk_size = int(self.base_chunk_size * (0.7 + 0.6 * importance))
                adaptive_overlap = int(adaptive_chunk_size * self.overlap_ratio)
                
                # Create chunks for this section
                section_chunks = self._split_section_smartly(
                    section_text, 
                    adaptive_chunk_size, 
                    adaptive_overlap,
                    section['type']
                )
                
                # Convert to TextChunk objects
                for chunk_text in section_chunks:
                    chunk = TextChunk(
                        id=f"{doc_id}_chunk_{chunk_index}",
                        content=chunk_text,
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        tokens=len(chunk_text.split()),
                        section_title=section['title'],
                        section_index=len(chunks),
                        metadata={
                            'section_type': section['type'],
                            'section_importance': section['importance'],
                            'adaptive_chunk_size': adaptive_chunk_size,
                            'section_start': section['start'],
                            'section_end': section['end'],
                            'created_at': datetime.now().isoformat(),
                            'chunk_chars': len(chunk_text)
                        }
                    )
                    
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Safety check
                    if len(chunks) >= self.max_chunks_per_doc:
                        logger.warning(f"Reached max chunks limit for {doc_id}")
                        section_pbar.close()
                        return chunks
                
                section_pbar.update(1)
                section_pbar.set_postfix({
                    "Chunks": len(chunks),
                    "Type": section['type'][:8],
                    "Importance": f"{importance:.2f}"
                })
        
        return chunks
    
    def _split_section_smartly(self, text: str, chunk_size: int, overlap: int, section_type: str) -> List[str]:
        """Smart section splitting with financial content awareness"""
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        iteration_count = 0
        max_iterations = len(text) // 50 + 100
        
        # Financial content markers for better splitting
        financial_markers = [
            r'\$[\d,.]+(?: million| billion| thousand)?',  # Dollar amounts
            r'\d+\.?\d*%',  # Percentages
            r'(?:Q[1-4]|quarter|fiscal year|FY)\s+\d{4}',  # Time periods
            r'(?:revenue|profit|loss|earnings|sales|income|ebitda)',  # Financial terms
        ]
        
        while start < len(text) and iteration_count < max_iterations:
            iteration_count += 1
            
            end = min(start + chunk_size, len(text))
            
            # Smart boundary detection based on content type
            if end < len(text):
                end = self._find_smart_boundary(text, start, end, section_type, financial_markers)
            
            # Extract and validate chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text and len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            elif chunk_text and len(chunks) == 0:
                # Keep short chunks if they're the only content
                chunks.append(chunk_text)
            
            # Calculate next start position
            new_start = end - overlap
            
            # Ensure progress to prevent infinite loops
            if new_start <= start:
                new_start = start + max(self.min_chunk_size, chunk_size // 4)
            
            start = new_start
            
            # Safety break
            if iteration_count >= max_iterations:
                logger.warning(f"Iteration limit reached for section type: {section_type}")
                break
        
        return chunks
    
    def _find_smart_boundary(self, text: str, start: int, end: int, section_type: str, financial_markers: List[str]) -> int:
        """Find intelligent chunk boundaries based on content type"""
        
        # Look for natural boundaries in a window around the target end
        search_window = min(200, (end - start) // 3)
        search_start = max(start, end - search_window)
        search_end = min(len(text), end + search_window // 2)
        search_text = text[search_start:search_end]
        
        # Priority 1: Financial statement boundaries (for financial sections)
        if section_type in ['financial_performance', 'revenue_analysis', 'profitability']:
            financial_boundaries = []
            for pattern in financial_markers:
                for match in re.finditer(pattern, search_text):
                    boundary_pos = search_start + match.end()
                    if boundary_pos > start + self.min_chunk_size:
                        financial_boundaries.append(boundary_pos)
            
            if financial_boundaries:
                # Use the boundary closest to target end
                closest_boundary = min(financial_boundaries, key=lambda x: abs(x - end))
                if abs(closest_boundary - end) < search_window:
                    return closest_boundary
        
        # Priority 2: Sentence boundaries
        sentence_pattern = r'[.!?]\s+'
        sentence_matches = []
        for match in re.finditer(sentence_pattern, search_text):
            boundary_pos = search_start + match.end()
            if boundary_pos > start + self.min_chunk_size:
                sentence_matches.append(boundary_pos)
        
        if sentence_matches:
            # Prefer boundaries closer to target end
            best_boundary = min(sentence_matches, key=lambda x: abs(x - end))
            return best_boundary
        
        # Priority 3: Paragraph boundaries
        paragraph_pattern = r'\n\s*\n'
        for match in re.finditer(paragraph_pattern, search_text):
            boundary_pos = search_start + match.start()
            if boundary_pos > start + self.min_chunk_size:
                return boundary_pos
        
        # Fallback: Use original end
        return end
    
    def _validate_and_enhance_chunks(self, chunks: List[TextChunk], doc_id: str) -> List[TextChunk]:
        """Validate and enhance chunks with additional metadata"""
        
        validated_chunks = []
        
        for chunk in chunks:
            # Skip empty or too-short chunks
            if not chunk.content or len(chunk.content.strip()) < self.min_chunk_size:
                continue
            
            # Enhance metadata with content analysis
            content_analysis = self._analyze_chunk_content(chunk.content)
            chunk.metadata.update(content_analysis)
            
            # Add financial relevance score
            chunk.metadata['financial_relevance'] = self._calculate_financial_relevance(chunk.content)
            
            validated_chunks.append(chunk)
        
        # Re-index chunks
        for i, chunk in enumerate(validated_chunks):
            chunk.chunk_index = i
            chunk.id = f"{doc_id}_chunk_{i}"
        
        return validated_chunks
    
    def _analyze_chunk_content(self, content: str) -> Dict[str, Any]:
        """Analyze chunk content for financial indicators"""
        
        analysis = {
            'has_numbers': bool(re.search(r'\d+', content)),
            'has_percentages': bool(re.search(r'\d+\.?\d*%', content)),
            'has_currency': bool(re.search(r'\$[\d,]+', content)),
            'has_dates': bool(re.search(r'\b\d{4}\b|\b(?:Q[1-4]|quarter)\b', content)),
            'sentence_count': len(re.findall(r'[.!?]+', content)),
            'word_count': len(content.split()),
            'avg_sentence_length': 0
        }
        
        if analysis['sentence_count'] > 0:
            analysis['avg_sentence_length'] = analysis['word_count'] / analysis['sentence_count']
        
        return analysis
    
    def _calculate_financial_relevance(self, content: str) -> float:
        """Calculate financial relevance score for content"""
        
        financial_terms = [
            'revenue', 'profit', 'loss', 'earnings', 'sales', 'income', 'ebitda',
            'margin', 'cash', 'debt', 'equity', 'assets', 'liabilities',
            'growth', 'decline', 'increase', 'decrease', 'performance',
            'quarter', 'annual', 'fiscal', 'financial', 'business'
        ]
        
        content_lower = content.lower()
        score = 0.0
        
        # Term frequency scoring
        for term in financial_terms:
            count = content_lower.count(term)
            score += count * 0.1
        
        # Financial indicators
        if re.search(r'\$[\d,]+', content):
            score += 0.3
        if re.search(r'\d+\.?\d*%', content):
            score += 0.2
        if re.search(r'\b(?:Q[1-4]|quarter|fiscal)\s+\d{4}', content):
            score += 0.2
        
        # Normalize to 0-1 range
        return min(1.0, score)