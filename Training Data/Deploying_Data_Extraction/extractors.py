# extractors.py
import spacy
import pandas as pd
import random
import re
import numpy as np
from spacy.training.example import Example
from dateutil import parser as date_parser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

class SpacyNERExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.ner = self.nlp.get_pipe("ner")
    
    def filter_overlapping_entities(self, entities):
        entities = sorted(entities, key=lambda x: x[0])
        non_overlapping = []
        last_end = -1
        for start, end, label in entities:
            if start >= last_end:
                non_overlapping.append((start, end, label))
                last_end = end
        return non_overlapping
    
    def prepare_training_data(self, texts, labels):
        data = []
        for i, (text, label_dict) in enumerate(zip(texts, labels)):
            entities = []

            for col in ['reporter_name', 'person_involved', 'incident_date', 'incident_time',
                       'department', 'incident_description', 'location', 'injury_description']:

                if col in label_dict:
                    value = str(label_dict[col]).strip()
                    if value != 'N/A' and value != 'nan' and pd.notna(value) and value in text:
                        start = text.find(value)
                        if start != -1:
                            end = start + len(value)
                            entities.append((start, end, col.upper()))

            entities = self.filter_overlapping_entities(entities)
            if entities:
                data.append((text, {"entities": entities}))

        return data
    
    def train(self, train_texts, train_labels):
        print("Preparing spaCy training data...")
        training_data = self.prepare_training_data(train_texts, train_labels)
        print(f"Prepared {len(training_data)} training samples for spaCy")

        # Add custom labels
        for _, annotations in training_data:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

        # Train model
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.resume_training()
            for itn in range(30):
                random.shuffle(training_data)
                losses = {}
                for text, annotations in training_data:
                    try:
                        example = Example.from_dict(self.nlp.make_doc(text), annotations)
                        self.nlp.update([example], drop=0.5, losses=losses)
                    except:
                        continue
                if itn % 10 == 0:
                    print(f"Iteration {itn+1}, Losses: {losses}")
    
    def extract(self, text):
        doc = self.nlp(text)
        extracted = {}

        for ent in doc.ents:
            field_name = ent.label_.lower()
            if field_name not in extracted:  # Take first occurrence
                extracted[field_name] = ent.text

        return extracted

class HybridExtractor:
    def __init__(self):
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        self.time_patterns = [
            r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            r'\bat\s+\d{1,2}:\d{2}\b'
        ]
        self.name_patterns = [
            r'\breported by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:from|reported|involved)'
        ]

        # ML components for contextual fields
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.department_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500)),
            ('classifier', MultinomialNB())
        ])
        self.injury_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500)),
            ('classifier', MultinomialNB())
        ])
    
    def extract_with_regex(self, text):
        extracted = {}

        # Extract dates
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    parsed_date = date_parser.parse(matches[0])
                    extracted['incident_date'] = parsed_date.strftime('%d/%m/%Y')
                    break
                except:
                    continue

        # Extract times
        for pattern in self.time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted['incident_time'] = matches[0].replace('at ', '')
                break

        # Extract reporter name
        for pattern in self.name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['reporter_name'] = match.group(1)
                break

        # Extract location patterns
        location_patterns = [
            r'(?:at|in|near)\s+([A-Z][a-z]*(?:\s+[A-Z]*[a-z]*)*\s*\d*)',
            r'(Warehouse\s+[A-Z])',
            r'(Dry Dock\s+\d+)',
            r'(Building\s+\d+)'
        ]

        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted['location'] = match.group(1)
                break

        return extracted
    
    def train_ml_components(self, train_texts, train_labels):
        print("Training ML components for contextual extraction...")

        # Prepare department labels
        dept_labels = []
        injury_labels = []

        for label in train_labels:
            dept = label.get('department', 'Unknown')
            dept_labels.append(dept if pd.notna(dept) and dept != 'N/A' else 'Unknown')

            was_injured = label.get('was_injured', 'No')
            injury_labels.append(was_injured if pd.notna(was_injured) else 'No')

        # Train classifiers
        self.department_classifier.fit(train_texts, dept_labels)
        self.injury_classifier.fit(train_texts, injury_labels)

        print("ML components trained successfully")
    
    def extract(self, text):
        # Start with regex extraction
        extracted = self.extract_with_regex(text)

        # Add ML predictions
        try:
            dept_pred = self.department_classifier.predict([text])[0]
            if dept_pred != 'Unknown':
                extracted['department'] = dept_pred

            injury_pred = self.injury_classifier.predict([text])[0]
            extracted['was_injured'] = injury_pred
        except:
            pass

        return extracted

class TemplateMLExtractor:
    def __init__(self):
        self.templates = {
            'incident_description': r'(?:incident|accident|event).*?(?:caused|resulted|leading|involving)\s+(.+?)(?:\.|The|,\s*[A-Z])',
            'injury_description': r'(?:suffered|sustained|injury|injured|hurt|damage)\s+(.+?)(?:\.|from|due to|$)',
            'person_involved': r'(?:involving|victim|worker|employee|person)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            'department_mention': r'(?:from the|department of|in the)\s+([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)*)\s+department'
        }

        self.classifiers = {}

    def extract_with_templates(self, text):
        extracted = {}

        for field, pattern in self.templates.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Clean up extracted text
                value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
                extracted[field] = value[:200]  # Limit length

        return extracted

    def train_classifiers(self, train_texts, train_labels):
        print("Training template-based classifiers...")

        # Build classifiers for different field types
        field_mappings = {
            'location': 'location',
            'label': 'label',
            'department': 'department'
        }

        for field_name, label_key in field_mappings.items():
            try:
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=300, ngram_range=(1, 2))),
                    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
                ])

                y = []
                for label in train_labels:
                    value = label.get(label_key, 'Unknown')
                    if pd.isna(value) or value == 'N/A':
                        value = 'Unknown'
                    y.append(str(value))

                pipeline.fit(train_texts, y)
                self.classifiers[field_name] = pipeline
                print(f"Trained classifier for {field_name}")
            except Exception as e:
                print(f"Error training {field_name} classifier: {e}")

    def extract(self, text):
        # Get template-based extractions
        extracted = self.extract_with_templates(text)

        # Add ML predictions
        for field_name, classifier in self.classifiers.items():
            try:
                prediction = classifier.predict([text])[0]
                if prediction != 'Unknown':
                    extracted[field_name] = prediction
            except:
                continue

        return extracted

class AdvancedEnsembleExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        self.field_classifiers = {}

    def extract_features(self, text):
        features = {}

        # Text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))

        # Pattern features
        features['has_date'] = int(bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', text)))
        features['has_time'] = int(bool(re.search(r'\d{1,2}:\d{2}', text)))
        features['has_names'] = int(bool(re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text)))
        features['has_injury_words'] = int(bool(re.search(r'\b(?:injury|injured|hurt|damage|burn|cut|fall|fell)\b', text, re.IGNORECASE)))

        # Department indicators
        dept_words = ['facilities', 'health', 'safety', 'operations', 'maintenance', 'security']
        features['dept_mentions'] = sum(1 for word in dept_words if word in text.lower())

        return features

    def train(self, train_texts, train_labels):
        print("Training advanced ensemble extractor...")

        # First, fit the TF-IDF vectorizer on all training texts
        tfidf_features = self.vectorizer.fit_transform(train_texts).toarray()

        # Extract statistical features for all texts
        stat_features = []
        for text in train_texts:
            text_features = self.extract_features(text)
            stat_features.append(list(text_features.values()))

        stat_features = np.array(stat_features)

        # Combine statistical and TF-IDF features
        X = np.hstack([stat_features, tfidf_features])

        # Train classifiers for each field
        target_fields = ['department', 'location', 'was_injured', 'label']

        for field in target_fields:
            try:
                y = []
                for label in train_labels:
                    value = label.get(field, 'Unknown')
                    if pd.isna(value) or value == 'N/A':
                        value = 'Unknown'
                    y.append(str(value))

                classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                classifier.fit(X, y)
                self.field_classifiers[field] = classifier

                print(f"Trained ensemble classifier for {field}")
            except Exception as e:
                print(f"Error training ensemble classifier for {field}: {e}")

    def extract(self, text):
        extracted = {}

        # Extract statistical features
        text_features = self.extract_features(text)
        stat_features = np.array(list(text_features.values())).reshape(1, -1)

        # Extract TF-IDF features
        tfidf_features = self.vectorizer.transform([text]).toarray()

        # Combine features
        combined_features = np.hstack([stat_features, tfidf_features])

        # Make predictions for each field
        for field, classifier in self.field_classifiers.items():
            try:
                prediction = classifier.predict(combined_features)[0]
                if prediction != 'Unknown':
                    extracted[field] = prediction
            except Exception as e:
                continue

        return extracted

class EnsembleVotingExtractor:
    def __init__(self):
        self.spacy_extractor = SpacyNERExtractor()
        self.hybrid_extractor = HybridExtractor()
        self.template_extractor = TemplateMLExtractor()
        self.advanced_extractor = AdvancedEnsembleExtractor()

    def train_all_models(self, train_texts, train_labels):
        print("Training all ensemble models...")
        print("1. Training spaCy NER...")
        self.spacy_extractor.train(train_texts, train_labels)
        print("2. Training Hybrid extractor...")
        self.hybrid_extractor.train_ml_components(train_texts, train_labels)
        print("3. Training Template extractor...")
        self.template_extractor.train_classifiers(train_texts, train_labels)
        print("4. Training Advanced extractor...")
        self.advanced_extractor.train(train_texts, train_labels)
        print("All models trained successfully!")

    def extract_with_voting(self, text):
        # Your existing voting logic
        predictions = {}
        
        try:
            predictions['spacy'] = self.spacy_extractor.extract(text)
        except:
            predictions['spacy'] = {}

        try:
            predictions['hybrid'] = self.hybrid_extractor.extract(text)
        except:
            predictions['hybrid'] = {}

        try:
            predictions['template'] = self.template_extractor.extract(text)
        except:
            predictions['template'] = {}

        try:
            predictions['advanced'] = self.advanced_extractor.extract(text)
        except:
            predictions['advanced'] = {}

        # Combine predictions with voting
        final_result = {}
        all_fields = set()
        for model_preds in predictions.values():
            all_fields.update(model_preds.keys())

        for field in all_fields:
            votes = {}
            for model_name, model_preds in predictions.items():
                if field in model_preds and model_preds[field]:
                    value = model_preds[field]
                    votes[value] = votes.get(value, 0) + 1

            if votes:
                final_result[field] = max(votes.keys(), key=votes.get)

        return final_result, predictions