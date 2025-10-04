import hashlib
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, field

@dataclass
class FinancialEntity:
    """Financial entity with domain-specific attributes"""
    id: str
    name: str
    type: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source_chunks: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        content = f"{self.name}_{self.type}_{self.description[:30]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass
class FinancialRelationship:
    """Financial relationship between entities"""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    description: str
    keywords: str = ""
    weight: float = 1.0
    context: str = ""
    confidence: float = 1.0
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        content = f"{self.source_id}_{self.target_id}_{self.relation_type}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
@dataclass
class TextChunk:
    """Text chunk with comprehensive metadata"""
    id: str
    content: str
    doc_id: str
    chunk_index: int
    tokens: int
    entities: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    section_title: Optional[str] = None
    section_index: Optional[int] = None

@dataclass
class QueryParam:
    """Query parameters with advanced options"""
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "hybrid"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    top_k: int = 20
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000
