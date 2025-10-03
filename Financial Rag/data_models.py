import hashlib
from typing import Dict, List, Any
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