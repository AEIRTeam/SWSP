# extractors.py
import re
import spacy
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class RobustPatternExtractor:
    """
    Robust pattern-based extractor that works well on new data
    without requiring training on specific datasets.
    """
    
    def __init__(self):
        self.confidence_threshold = 0.3
        self.use_fuzzy_matching = False
        self.strict_date_format = False
        self.require_injury_keywords = True
        
        # Enhanced patterns for better extraction
        self.date_patterns = [
            # Standard formats
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
            # Month name formats
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b',
            # Alternative formats
            r'\b(\d{1,2}(st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2})\b'  # YY format
        ]
        
        self.time_patterns = [
            r'\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm))\b',
            r'\b(\d{1,2}:\d{2}(?::\d{2})?)\b',
            r'\b(\d{1,2}\s*(?:AM|PM|am|pm))\b',
            r'\b((?:morning|afternoon|evening|night))\b'
        ]
        
        self.name_patterns = [
            # Full names
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            # Names with titles
            r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|Miss)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            # Names in context
            r'(?:employee|worker|person|individual|staff|member)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'(?:reported by|report from|submitted by)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            # Names with employee context
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:was|is|reported|stated|mentioned)'
        ]
        
        self.department_patterns = [
            r'\b(warehouse|office|facility|department|production|manufacturing|maintenance|administration|security|shipping|receiving|quality|safety)\b',
            r'\b(floor\s+\d+|level\s+\d+|room\s+\d+|building\s+[A-Z]|section\s+[A-Z0-9]+)\b',
            r'\b([A-Z][a-z]+\s+(?:department|division|section|area|zone))\b'
        ]
        
        self.location_patterns = [
            r'\b(warehouse|factory|office|plant|facility|building|parking lot|cafeteria|break room|storage area)\b',
            r'\b(loading dock|assembly line|conveyor|machine shop|laboratory|reception|entrance|exit)\b',
            r'\b(?:at|in|near|by)\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+){0,2})\b'
        ]
        
        self.injury_keywords = [
            'cut', 'bruise', 'burn', 'sprain', 'fracture', 'injury', 'hurt', 'pain', 
            'ache', 'wound', 'injured', 'wounded', 'burned', 'bruised', 'bleeding',
            'broken', 'twisted', 'strained', 'puncture', 'laceration', 'abrasion',
            'contusion', 'trauma', 'damage', 'harm'
        ]
        
        self.injury_indicators = [
            'slipped', 'fell', 'tripped', 'caught', 'struck', 'hit', 'bumped',
            'pinched', 'crushed', 'dropped', 'lifted', 'pulled', 'twisted'
        ]
    
    def extract_dates(self, text: str) -> Optional[str]:
        """Extract dates with multiple pattern matching"""
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Return the first match, handling tuple results
                match = matches[0]
                if isinstance(match, tuple):
                    return match[0]
                return match
        return None
    
    def extract_times(self, text: str) -> Optional[str]:
        """Extract times with multiple pattern matching"""
        for pattern in self.time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        return None
    
    def extract_names(self, text: str) -> Dict[str, str]:
        """Extract names with context awareness"""
        names = {}
        found_names = []
        
        for pattern in self.name_patterns:
            matches = re.findall(pattern, text)
            found_names.extend(matches)
        
        # Remove duplicates while preserving order
        unique_names = []
        seen = set()
        for name in found_names:
            if name not in seen:
                unique_names.append(name)
                seen.add(name)
        
        # Assign names based on context
        if len(unique_names) >= 1:
            names['reporter_name'] = unique_names[0]
        if len(unique_names) >= 2:
            names['person_involved'] = unique_names[1]
        elif len(unique_names) == 1:
            # If only one name, try to determine if it's reporter or person involved
            text_lower = text.lower()
            if any(word in text_lower for word in ['reported', 'report', 'submitted']):
                names['reporter_name'] = unique_names[0]
            else:
                names['person_involved'] = unique_names[0]
        
        return names
    
    def extract_departments_locations(self, text: str) -> Dict[str, str]:
        """Extract department and location information"""
        result = {}
        
        # Extract departments
        for pattern in self.department_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                result['department'] = matches[0]
                break
        
        # Extract locations
        for pattern in self.location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                result['location'] = matches[0]
                break
        
        return result
    
    def analyze_injury(self, text: str) -> Dict[str, str]:
        """Analyze injury-related information"""
        text_lower = text.lower()
        
        # Check for injury keywords
        injury_found = any(keyword in text_lower for keyword in self.injury_keywords)
        indicator_found = any(indicator in text_lower for indicator in self.injury_indicators)
        
        result = {
            'was_injured': 'Yes' if (injury_found or indicator_found) else 'No'
        }
        
        # Extract injury description if injury detected
        if injury_found or indicator_found:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                if any(keyword in sentence_lower for keyword in self.injury_keywords + self.injury_indicators):
                    result['injury_description'] = sentence.strip()
                    break
        
        return result
    
    def categorize_incident(self, text: str) -> str:
        """Categorize the incident based on content"""
        text_lower = text.lower()
        
        # Define category keywords
        categories = {
            'Fall/Slip': ['slip', 'fall', 'trip', 'stumble', 'wet floor', 'slippery'],
            'Cut/Laceration': ['cut', 'sharp', 'blade', 'knife', 'glass', 'laceration'],
            'Burn': ['burn', 'hot', 'fire', 'steam', 'chemical', 'acid'],
            'Equipment': ['equipment', 'machine', 'malfunction', 'mechanical', 'tool'],
            'Lifting/Strain': ['lift', 'heavy', 'strain', 'back', 'muscle', 'pull'],
            'Chemical': ['chemical', 'spill', 'exposure', 'toxic', 'fumes'],
            'Transportation': ['vehicle', 'forklift', 'truck', 'driving', 'collision'],
            'Electrical': ['electrical', 'shock', 'wire', 'power', 'electric']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'General'
    
    def extract_comprehensive(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive extraction using all pattern methods
        """
        if not text or not text.strip():
            return {field: "" for field in ['reporter_name', 'person_involved', 'incident_date', 
                                          'incident_time', 'department', 'incident_description', 
                                          'location', 'label', 'was_injured', 'injury_description']}
        
        result = {}
        
        # Extract dates and times
        result['incident_date'] = self.extract_dates(text) or ""
        result['incident_time'] = self.extract_times(text) or ""
        
        # Extract names
        names = self.extract_names(text)
        result['reporter_name'] = names.get('reporter_name', "")
        result['person_involved'] = names.get('person_involved', "")
        
        # Extract departments and locations
        dept_loc = self.extract_departments_locations(text)
        result['department'] = dept_loc.get('department', "")
        result['location'] = dept_loc.get('location', "")
        
        # Analyze injuries
        injury_info = self.analyze_injury(text)
        result['was_injured'] = injury_info.get('was_injured', "No")
        result['injury_description'] = injury_info.get('injury_description', "")
        
        # Set incident description
        result['incident_description'] = text[:500] + "..." if len(text) > 500 else text
        
        # Categorize incident
        result['label'] = self.categorize_incident(text)
        
        return result

class EnhancedSpaCyExtractor:
    """
    Enhanced spaCy-based extractor with fallback capabilities
    """
    
    def __init__(self, nlp_model=None):
        self.nlp = nlp_model
        self.confidence_scores = {}
    
    def extract(self, text: str) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Extract entities using spaCy with confidence scores"""
        if not self.nlp or not text:
            return {}, {}
        
        result = {}
        confidence = {}
        
        try:
            doc = self.nlp(text)
            
            # Extract named entities
            persons = [ent.text for ent in doc.ents if ent.label_ in ["PERSON"]]
            dates = [ent.text for ent in doc.ents if ent.label_ in ["DATE"]]
            times = [ent.text for ent in doc.ents if ent.label_ in ["TIME"]]
            orgs = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "FAC"]]
            
            # Assign extracted entities
            if persons:
                result['reporter_name'] = persons[0] if len(persons) >= 1 else ""
                result['person_involved'] = persons[1] if len(persons) >= 2 else ""
                confidence['spacy_reporter_name'] = 0.8 if len(persons) >= 1 else 0
                confidence['spacy_person_involved'] = 0.8 if len(persons) >= 2 else 0
            
            if dates:
                result['incident_date'] = dates[0]
                confidence['spacy_incident_date'] = 0.9
            
            if times:
                result['incident_time'] = times[0]
                confidence['spacy_incident_time'] = 0.8
            
            if orgs:
                result['department'] = orgs[0]
                result['location'] = orgs[0]
                confidence['spacy_department'] = 0.7
                confidence['spacy_location'] = 0.7
            
            # Basic injury detection
            injury_words = ['injured', 'hurt', 'wound', 'pain', 'cut', 'burn']
            text_lower = text.lower()
            if any(word in text_lower for word in injury_words):
                result['was_injured'] = 'Yes'
                confidence['spacy_was_injured'] = 0.6
            else:
                result['was_injured'] = 'No'
                confidence['spacy_was_injured'] = 0.4
            
            result['incident_description'] = text[:500] + "..." if len(text) > 500 else text
            result['label'] = 'General'
            
        except Exception as e:
            logger.error(f"SpaCy extraction error: {str(e)}")
            result = {}
            confidence = {}
        
        return result, confidence
    
    def train(self, texts: List[str], labels: List[Dict]):
        """Placeholder training method"""
        # In a real implementation, this would train the spaCy model
        pass

class EnsembleVotingExtractor:
    """
    Enhanced ensemble extractor that combines multiple approaches
    """
    
    def __init__(self, nlp_model=None):
        self.pattern_extractor = RobustPatternExtractor()
        self.spacy_extractor = EnhancedSpaCyExtractor(nlp_model)
        
        # Placeholder for other extractors (for compatibility)
        self.hybrid_extractor = self._create_placeholder_extractor()
        self.template_extractor = self._create_placeholder_extractor()
        self.advanced_extractor = self._create_placeholder_extractor()
    
    def _create_placeholder_extractor(self):
        """Create placeholder extractor for compatibility"""
        class PlaceholderExtractor:
            def train(self, *args, **kwargs):
                pass
            def train_ml_components(self, *args, **kwargs):
                pass
            def train_classifiers(self, *args, **kwargs):
                pass
            def extract(self, text):
                return {}
        
        return PlaceholderExtractor()
    
    def extract_with_voting(self, text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract using ensemble voting with improved logic
        """
        if not text or not text.strip():
            empty_result = {field: "" for field in ['reporter_name', 'person_involved', 'incident_date', 
                                                  'incident_time', 'department', 'incident_description', 
                                                  'location', 'label', 'was_injured', 'injury_description']}
            return empty_result, {}
        
        # Get extractions from different methods
        pattern_result = self.pattern_extractor.extract_comprehensive(text)
        spacy_result, spacy_confidence = self.spacy_extractor.extract(text)
        
        # Combine results with intelligent voting
        final_result = {}
        model_breakdown = {
            'pattern_extraction': pattern_result,
            'spacy_extraction': spacy_result,
            'spacy_confidence': spacy_confidence
        }
        
        fields = ['reporter_name', 'person_involved', 'incident_date', 'incident_time',
                 'department', 'incident_description', 'location', 'label',
                 'was_injured', 'injury_description']
        
        for field in fields:
            candidates = []
            
            # Add pattern result (high reliability for new data)
            if field in pattern_result and pattern_result[field]:
                candidates.append((pattern_result[field], 0.8, 'pattern'))
            
            # Add spaCy result if confident
            if field in spacy_result and spacy_result[field]:
                spacy_conf = spacy_confidence.get(f'spacy_{field}', 0.5)
                if spacy_conf > 0.6:  # Only use high-confidence spaCy results
                    candidates.append((spacy_result[field], spacy_conf, 'spacy'))
            
            # Choose best candidate
            if candidates:
                # Sort by confidence and choose the best
                best_candidate = max(candidates, key=lambda x: x[1])
                final_result[field] = best_candidate[0]
            else:
                final_result[field] = ""
        
        # Ensure all required fields are present
        for field in fields:
            if field not in final_result:
                final_result[field] = ""
        
        return final_result, model_breakdown

# Backward compatibility classes (simplified versions)
class SpaCyExtractor:
    def __init__(self, nlp_model=None):
        self.enhanced = EnhancedSpaCyExtractor(nlp_model)
    
    def train(self, texts, labels):
        return self.enhanced.train(texts, labels)
    
    def extract(self, text):
        result, _ = self.enhanced.extract(text)
        return result

class HybridExtractor:
    def __init__(self):
        self.pattern_extractor = RobustPatternExtractor()
    
    def train_ml_components(self, texts, labels):
        pass
    
    def extract(self, text):
        return self.pattern_extractor.extract_comprehensive(text)

class TemplateExtractor:
    def train_classifiers(self, texts, labels):
        pass
    
    def extract(self, text):
        # Placeholder implementation
        return {}
