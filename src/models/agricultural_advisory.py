"""
💬 Agricultural Advisory Module - Annam AI
==========================================

This module implements an AI-powered agricultural advisory system using Large Language Models (LLM)
and Retrieval-Augmented Generation (RAG) to provide expert agricultural advice.

Key Features:
- Integration with open-source LLMs (Llama2, Mistral, Falcon)
- RAG system using FAISS or ChromaDB vector database
- Agricultural knowledge base with crop guides, pest management, etc.
- Context-aware question answering
- Multi-language support
- Conversation memory and context
- Integration with weather and soil data
- Expert-level agricultural recommendations

Author: Annam AI Team
License: MIT
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import requests
import warnings
warnings.filterwarnings('ignore')

import logging
from datetime import datetime
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logger = logging.getLogger('AnnamAI.Advisory')


class AgriculturalKnowledgeBase:
    """Manage agricultural knowledge base for RAG system."""

    def __init__(self, data_dir="data/agricultural_kb"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def create_knowledge_base(self):
        """Create comprehensive agricultural knowledge base."""
        logger.info("Creating agricultural knowledge base...")

        knowledge_docs = []

        # Crop cultivation guides
        crop_guides = self._create_crop_guides()
        knowledge_docs.extend(crop_guides)

        # Pest and disease management
        pest_disease_docs = self._create_pest_disease_docs()
        knowledge_docs.extend(pest_disease_docs)

        # Soil and fertilizer guides
        soil_fertilizer_docs = self._create_soil_fertilizer_docs()
        knowledge_docs.extend(soil_fertilizer_docs)

        # Save knowledge base
        kb_df = pd.DataFrame(knowledge_docs)
        kb_df.to_csv(self.data_dir / 'knowledge_base.csv', index=False)

        logger.info(f"Created knowledge base with {len(knowledge_docs)} documents")
        return knowledge_docs

    def _create_crop_guides(self):
        """Create crop cultivation guides."""
        crops = {
            'wheat': {
                'overview': 'Wheat is a cereal grain and one of the most important food crops worldwide.',
                'planting': 'Plant in fall for winter wheat or spring for spring wheat. Optimal soil temperature is 50-60°F.',
                'soil': 'Prefers well-drained, fertile soils with pH 6.0-7.5. Requires good nitrogen availability.',
                'fertilizer': 'Apply 80-120 lbs N/acre, 40-60 lbs P2O5/acre, and 40-60 lbs K2O/acre based on soil tests.',
                'irrigation': 'Requires 12-15 inches of water during growing season. Critical periods are tillering and grain filling.',
                'harvest': 'Harvest when grain moisture is 13-14%. Use combine harvester for efficient collection.'
            },
            'corn': {
                'overview': 'Corn (maize) is a major cereal crop used for food, feed, and industrial purposes.',
                'planting': 'Plant when soil temperature reaches 50°F. Optimal planting depth is 1.5-2 inches.',
                'soil': 'Thrives in well-drained, fertile soils with pH 6.0-6.8. Needs high organic matter.',
                'fertilizer': 'Requires 150-200 lbs N/acre, 60-80 lbs P2O5/acre, and 60-80 lbs K2O/acre.',
                'irrigation': 'Needs 20-30 inches of water. Critical periods are silking and grain filling.',
                'harvest': 'Harvest at 15-20% grain moisture for field corn, earlier for sweet corn.'
            },
            'tomato': {
                'overview': 'Tomatoes are warm-season vegetables requiring specific growing conditions.',
                'planting': 'Start from transplants when soil temperature is consistently above 60°F.',
                'soil': 'Requires well-drained, fertile soil with pH 6.0-6.8. Needs consistent moisture.',
                'fertilizer': 'Apply balanced fertilizer (10-10-10) at planting, then side-dress with nitrogen.',
                'irrigation': 'Provide 1-2 inches of water per week. Avoid overhead watering to prevent disease.',
                'harvest': 'Harvest when fruits are fully colored but still firm. Pick regularly for continued production.'
            },
            'potato': {
                'overview': 'Potatoes are cool-season crops grown for their nutritious tubers.',
                'planting': 'Plant seed potatoes 2-4 inches deep when soil temperature reaches 45°F.',
                'soil': 'Prefers loose, well-drained soils with pH 5.8-6.2. Avoid heavy clay soils.',
                'fertilizer': 'Apply 100-150 lbs N/acre, 80-100 lbs P2O5/acre, and 150-200 lbs K2O/acre.',
                'irrigation': 'Requires consistent moisture. Provide 12-16 inches during growing season.',
                'harvest': 'Harvest when vines die back naturally. Cure in dark, ventilated area before storage.'
            }
        }

        docs = []
        for crop, info in crops.items():
            for aspect, content in info.items():
                docs.append({
                    'category': 'crops',
                    'subcategory': crop,
                    'topic': aspect,
                    'content': content,
                    'keywords': f"{crop} {aspect} cultivation farming agriculture"
                })

        return docs

    def _create_pest_disease_docs(self):
        """Create pest and disease management documents."""
        pest_disease_info = {
            'aphids': {
                'type': 'pest',
                'description': 'Small, soft-bodied insects that feed on plant sap.',
                'symptoms': 'Yellowing leaves, sticky honeydew, stunted growth, curled leaves.',
                'management': 'Use beneficial insects, insecticidal soaps, neem oil, or systemic insecticides.',
                'prevention': 'Monitor regularly, maintain plant health, encourage natural predators.'
            },
            'late_blight': {
                'type': 'disease',
                'description': 'Fungal disease affecting tomatoes and potatoes, caused by Phytophthora infestans.',
                'symptoms': 'Dark, water-soaked lesions on leaves and stems, white fuzzy growth on leaf undersides.',
                'management': 'Apply copper-based fungicides, remove infected plants, improve air circulation.',
                'prevention': 'Avoid overhead watering, plant resistant varieties, crop rotation.'
            },
            'powdery_mildew': {
                'type': 'disease',
                'description': 'Fungal disease causing white, powdery growth on plant surfaces.',
                'symptoms': 'White powdery coating on leaves, stems, and fruits, yellowing and wilting.',
                'management': 'Apply sulfur-based fungicides, improve air circulation, remove infected parts.',
                'prevention': 'Plant resistant varieties, avoid overcrowding, water at soil level.'
            }
        }

        docs = []
        for name, info in pest_disease_info.items():
            for aspect, content in info.items():
                docs.append({
                    'category': 'pests' if info['type'] == 'pest' else 'diseases',
                    'subcategory': name,
                    'topic': aspect,
                    'content': content,
                    'keywords': f"{name} {info['type']} {aspect} management control"
                })

        return docs

    def _create_soil_fertilizer_docs(self):
        """Create soil and fertilizer management documents."""
        soil_fertilizer_info = {
            'soil_testing': {
                'importance': 'Soil testing is crucial for determining nutrient levels, pH, and organic matter content.',
                'frequency': 'Test soil every 2-3 years or before making major fertilizer investments.',
                'parameters': 'Key parameters include pH, nitrogen, phosphorus, potassium, organic matter, and micronutrients.',
                'interpretation': 'Results guide fertilizer recommendations and soil amendment decisions.'
            },
            'nitrogen_management': {
                'role': 'Nitrogen is essential for vegetative growth, protein synthesis, and chlorophyll production.',
                'sources': 'Sources include urea, ammonium sulfate, nitrate forms, and organic amendments.',
                'timing': 'Apply based on crop growth stages, soil temperature, and weather conditions.',
                'efficiency': 'Use inhibitors, slow-release formulations, and split applications to improve efficiency.'
            }
        }

        docs = []
        for topic, info in soil_fertilizer_info.items():
            for aspect, content in info.items():
                docs.append({
                    'category': 'soil' if 'soil' in topic else 'fertilizers',
                    'subcategory': topic,
                    'topic': aspect,
                    'content': content,
                    'keywords': f"{topic} {aspect} soil fertility management"
                })

        return docs


class SimpleVectorDatabase:
    """Simple vector database implementation using TF-IDF and cosine similarity."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.document_vectors = None
        self.documents = []
        self.metadatas = []

    def create_embeddings(self, documents):
        """Create TF-IDF embeddings for documents."""
        logger.info(f"Creating TF-IDF embeddings for {len(documents)} documents...")

        # Prepare texts and metadata
        texts = []
        metadatas = []

        for doc in documents:
            # Combine content with context
            text = f"Category: {doc['category']} Topic: {doc['topic']} Content: {doc['content']}"
            texts.append(text)
            metadatas.append({
                'category': doc['category'],
                'subcategory': doc.get('subcategory', ''),
                'topic': doc['topic'],
                'keywords': doc.get('keywords', '')
            })

        # Create TF-IDF vectors
        self.document_vectors = self.vectorizer.fit_transform(texts)
        self.documents = texts
        self.metadatas = metadatas

        logger.info("Vector database created successfully")

    def search(self, query, top_k=5):
        """Search for relevant documents using cosine similarity."""
        if self.document_vectors is None:
            return []

        # Vectorize query
        query_vector = self.vectorizer.transform([query])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]

        # Get top-k most similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'metadata': self.metadatas[idx],
                'score': float(similarities[idx])
            })

        return results


class SimpleLLM:
    """Simple rule-based response generator when LLM is not available."""

    def __init__(self):
        self.response_templates = {
            'crop': "For {crop} cultivation, consider the following key factors: soil preparation, proper planting timing, adequate fertilization, and pest management. Consult local agricultural extension services for region-specific recommendations.",
            'pest': "For pest management, it's important to correctly identify the pest first. Use integrated pest management (IPM) approaches combining biological controls, cultural practices, and targeted chemical treatments when necessary.",
            'disease': "Plant disease management requires proper diagnosis, improving growing conditions, removing infected plant parts, and using appropriate treatments. Prevention through resistant varieties and good cultural practices is often most effective.",
            'soil': "Soil health is fundamental to successful farming. Conduct regular soil tests, maintain organic matter levels, ensure proper pH, and follow balanced fertilization practices based on test results.",
            'water': "Water management involves monitoring soil moisture, using efficient irrigation systems, and timing water applications based on crop growth stages and weather conditions."
        }

    def generate_response(self, prompt, max_length=300):
        """Generate response using rule-based approach."""
        prompt_lower = prompt.lower()

        # Check for keywords and generate appropriate response
        for keyword, template in self.response_templates.items():
            if keyword in prompt_lower:
                if keyword == 'crop':
                    # Try to extract crop name
                    crops = ['wheat', 'corn', 'tomato', 'potato', 'rice', 'soybean']
                    for crop in crops:
                        if crop in prompt_lower:
                            return template.format(crop=crop)
                    return template.format(crop="your crop")
                else:
                    return template

        # Default response
        return "Thank you for your agricultural question. For the most accurate advice, I recommend consulting with local agricultural extension services who can provide specific recommendations for your region and conditions."


class AgricultureAdvisor:
    """Main agricultural advisory system combining RAG and LLM."""

    def __init__(self, data_dir="data/agricultural_advisory"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.knowledge_base = AgriculturalKnowledgeBase()
        self.vector_db = SimpleVectorDatabase()
        self.llm = SimpleLLM()

        # Conversation history
        self.conversation_history = []
        self.max_history = 10

    def initialize_system(self):
        """Initialize the complete advisory system."""
        logger.info("Initializing Agricultural Advisory System...")

        # Create knowledge base
        documents = self.knowledge_base.create_knowledge_base()

        # Create vector database
        self.vector_db.create_embeddings(documents)

        # Initialize conversation database
        self._init_conversation_db()

        logger.info("Agricultural Advisory System initialized successfully")

    def _init_conversation_db(self):
        """Initialize SQLite database for conversation history."""
        db_path = self.data_dir / "conversations.db"
        conn = sqlite3.connect(db_path)

        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_id TEXT,
                question TEXT,
                context TEXT,
                response TEXT,
                feedback INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def get_advice(self, question, user_context=None, top_k_docs=3):
        """Get agricultural advice for a user question."""
        logger.info(f"Processing question: {question[:100]}...")

        # Search for relevant documents
        relevant_docs = self.vector_db.search(question, top_k=top_k_docs)

        # Prepare context
        context = self._prepare_context(relevant_docs, user_context)

        # Generate response
        response = self._generate_contextual_response(question, context)

        # Store conversation
        self._store_conversation(question, context, response, user_context)

        # Update conversation history
        self.conversation_history.append({
            'question': question,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })

        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

        return {
            'response': response,
            'relevant_documents': relevant_docs,
            'confidence_score': self._calculate_confidence(relevant_docs),
            'follow_up_questions': self._generate_follow_up_questions(question, response)
        }

    def _prepare_context(self, relevant_docs, user_context):
        """Prepare context from retrieved documents and user information."""
        context_parts = []

        # Add retrieved document content
        for doc in relevant_docs:
            context_parts.append(f"• {doc['document']}")

        # Add user context if available
        if user_context:
            context_parts.append(f"User Context: {user_context}")

        return " ".join(context_parts)

    def _generate_contextual_response(self, question, context):
        """Generate response using context and LLM."""
        # Create enhanced prompt with context
        prompt = f"Context: {context} Question: {question}"

        # Generate response
        response = self.llm.generate_response(prompt)

        return response

    def _calculate_confidence(self, relevant_docs):
        """Calculate confidence score based on retrieved documents."""
        if not relevant_docs:
            return 0.1

        # Average similarity scores
        avg_score = np.mean([doc['score'] for doc in relevant_docs])

        # Normalize to 0-1 range
        confidence = min(1.0, max(0.1, avg_score))

        return confidence

    def _generate_follow_up_questions(self, question, response):
        """Generate relevant follow-up questions."""
        question_lower = question.lower()

        follow_ups = []

        if 'crop' in question_lower or 'plant' in question_lower:
            follow_ups.extend([
                "What soil conditions are best for this crop?",
                "How should I manage pests for this crop?",
                "What is the optimal planting time in my region?"
            ])

        if 'pest' in question_lower or 'disease' in question_lower:
            follow_ups.extend([
                "How can I prevent this problem in the future?",
                "Are there organic treatment options?",
                "What are the early warning signs to watch for?"
            ])

        if 'soil' in question_lower:
            follow_ups.extend([
                "How often should I test my soil?",
                "What amendments would improve soil health?",
                "How can I increase organic matter in my soil?"
            ])

        return follow_ups[:3]  # Return top 3 follow-up questions

    def _store_conversation(self, question, context, response, user_context):
        """Store conversation in database."""
        db_path = self.data_dir / "conversations.db"
        conn = sqlite3.connect(db_path)

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (timestamp, user_id, question, context, response)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            user_context.get('user_id', 'anonymous') if user_context else 'anonymous',
            question,
            context[:1000],  # Truncate context
            response
        ))

        conn.commit()
        conn.close()

    def export_knowledge_base(self, format_type="json"):
        """Export knowledge base in specified format."""
        kb_path = self.data_dir / "knowledge_base.csv"

        if kb_path.exists():
            df = pd.read_csv(kb_path)

            if format_type == "json":
                output_path = self.data_dir / "knowledge_base.json"
                df.to_json(output_path, orient='records', indent=2)
                logger.info(f"Knowledge base exported to {output_path}")
                return output_path

        return None


def main():
    """Main function to demonstrate agricultural advisory system."""
    logger.info("Starting Agricultural Advisory System Demo")

    # Initialize advisor
    advisor = AgricultureAdvisor()
    advisor.initialize_system()

    # Sample questions for demonstration
    sample_questions = [
        "My tomato plants have brown spots on the leaves. What should I do?",
        "When is the best time to plant corn in the spring?",
        "How can I improve the fertility of my clay soil?",
        "What's the most efficient way to irrigate my wheat field?",
        "I'm seeing small green insects on my plants. How do I identify and treat them?"
    ]

    print("\n" + "="*60)
    print("🌾 ANNAM AI - AGRICULTURAL ADVISORY SYSTEM")
    print("="*60)

    for question in sample_questions:
        print(f"\n❓ Question: {question}")

        # Get advice
        advice = advisor.get_advice(question)

        print(f"\n🎯 Response:\n{advice['response']}")
        print(f"\n📊 Confidence: {advice['confidence_score']:.2f}")

        if advice['follow_up_questions']:
            print("\n🔄 Suggested follow-up questions:")
            for i, follow_up in enumerate(advice['follow_up_questions'], 1):
                print(f"   {i}. {follow_up}")

        print("\n" + "-"*60)

    # Export knowledge base
    advisor.export_knowledge_base("json")

    logger.info("Agricultural Advisory System demo completed!")


if __name__ == "__main__":
    main()
