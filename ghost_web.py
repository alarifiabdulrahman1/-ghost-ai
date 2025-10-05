"""
Ghost AI - Self-Improving Web Assistant
"""

import os
import json
import datetime
from typing import List, Dict, Optional
import chromadb
import anthropic
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)


class GhostMemory:
    """Handles long-term memory using vector database"""
    
    def __init__(self, persist_directory="./ghost_memory"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        try:
            self.conversations = self.client.get_collection("conversations")
        except:
            self.conversations = self.client.create_collection(
                name="conversations",
                metadata={"description": "All conversation history"}
            )
        
        try:
            self.knowledge = self.client.get_collection("knowledge")
        except:
            self.knowledge = self.client.create_collection(
                name="knowledge",
                metadata={"description": "Extracted knowledge and learnings"}
            )
    
    def add_conversation(self, user_msg: str, ai_msg: str, rating: Optional[int] = None):
        """Store a conversation exchange"""
        conv_id = f"conv_{datetime.datetime.now().timestamp()}"
        
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "rating": str(rating) if rating else "unrated",
            "user_message": user_msg[:500],
        }
        
        full_text = f"User: {user_msg}\nGhost: {ai_msg}"
        
        self.conversations.add(
            documents=[full_text],
            metadatas=[metadata],
            ids=[conv_id]
        )
        
        return conv_id
    
    def search_memory(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search past conversations for relevant context"""
        if self.conversations.count() == 0:
            return []
        
        results = self.conversations.query(
            query_texts=[query],
            n_results=min(n_results, self.conversations.count())
        )
        
        memories = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                memories.append({
                    'content': doc,
                    'timestamp': metadata.get('timestamp'),
                    'rating': metadata.get('rating')
                })
        
        return memories
    
    def add_knowledge(self, knowledge: str, category: str = "general"):
        """Store extracted knowledge"""
        know_id = f"know_{datetime.datetime.now().timestamp()}"
        
        self.knowledge.add(
            documents=[knowledge],
            metadatas=[{
                "category": category,
                "timestamp": datetime.datetime.now().isoformat()
            }],
            ids=[know_id]
        )


class UserProfile:
    """Tracks user preferences and learned patterns"""
    
    def __init__(self, profile_path="./ghost_profile.json"):
        self.profile_path = profile_path
        self.profile = self.load_profile()
    
    def load_profile(self) -> Dict:
        """Load user profile from disk"""
        if os.path.exists(self.profile_path):
            with open(self.profile_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "name": "User",
                "preferences": {},
                "communication_style": "adaptive",
                "expertise_areas": [],
                "goals": [],
                "total_chats": 0,
                "created_at": datetime.datetime.now().isoformat()
            }
    
    def save_profile(self):
        """Save profile to disk"""
        with open(self.profile_path, 'w') as f:
            json.dump(self.profile, f, indent=2)
    
    def update_interaction_count(self):
        """Increment interaction counter"""
        self.profile["total_chats"] = self.profile.get("total_chats", 0) + 1
        self.save_profile()
    
    def add_preference(self, key: str, value: str):
        """Add or update a user preference"""
        self.profile["preferences"][key] = value
        self.save_profile()
    
    def add_expertise_area(self, area: str):
        """Track areas user is interested in"""
        if area not in self.profile.get("expertise_areas", []):
            if "expertise_areas" not in self.profile:
                self.profile["expertise_areas"] = []
            self.profile["expertise_areas"].append(area)
            self.save_profile()


class Ghost:
    """Main AI brain that orchestrates everything"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Claude API key required! Set ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.memory = GhostMemory()
        self.profile = UserProfile()
        self.sessions = {}
    
    def _build_context(self, user_message: str) -> str:
        """Build enhanced context from memory and profile"""
        context_parts = []
        
        context_parts.append(f"User Profile: {json.dumps(self.profile.profile, indent=2)}")
        
        relevant_memories = self.memory.search_memory(user_message, n_results=3)
        
        if relevant_memories:
            context_parts.append("\nRelevant past interactions:")
            for i, mem in enumerate(relevant_memories, 1):
                context_parts.append(f"\n{i}. [{mem['timestamp']}] {mem['content'][:300]}...")
        
        return "\n".join(context_parts)
    
    def get_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]
    
    def chat(self, user_message: str, session_id: str = "default", use_memory: bool = True) -> str:
        """Main chat interface"""
        self.profile.update_interaction_count()
        
        system_prompt = """You are Ghost, a self-improving AI assistant. You have access to:
1. User profile and preferences
2. Past conversation history
3. Learned knowledge over time

Your goal is to provide increasingly personalized and helpful responses as you learn more about the user.
Be adaptive, remember context, and continuously improve."""
        
        if use_memory:
            context = self._build_context(user_message)
            system_prompt += f"\n\nCONTEXT FROM MEMORY:\n{context}"
        
        session = self.get_session(session_id)
        messages = session.copy()
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=messages
        )
        
        ai_response = response.content[0].text
        
        session.append({"role": "user", "content": user_message})
        session.append({"role": "assistant", "content": ai_response})
        
        if len(session) > 20:
            session = session[-20:]
        
        self.sessions[session_id] = session
        
        return ai_response
    
    def rate_last_response(self, rating: int, session_id: str = "default"):
        """Rate the last AI response (1-5 stars)"""
        session = self.get_session(session_id)
        
        if len(session) >= 2:
            user_msg = session[-2]["content"]
            ai_msg = session[-1]["content"]
            
            conv_id = self.memory.add_conversation(user_msg, ai_msg, rating)
            
            if rating >= 4:
                self.memory.add_knowledge(
                    f"Good response pattern: {ai_msg[:500]}",
                    category="high_rated"
                )
            
            return f"Response rated {rating}/5 and saved to memory!"
        return "No response to rate yet."
    
    def clear_session(self, session_id: str = "default"):
        """Clear current conversation but keep long-term memory"""
        if session_id in self.sessions:
            self.sessions[session_id] = []
        return "Session cleared. Long-term memory preserved."
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "total_conversations": self.profile.profile.get("total_chats", 0),
            "stored_conversations": self.memory.conversations.count(),
            "stored_knowledge": self.memory.knowledge.count(),
            "expertise_areas": self.profile.profile.get("expertise_areas", []),
        }


# Initialize Ghost
ghost = Ghost()


# Web routes
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        response = ghost.chat(user_message, session_id)
        
        return jsonify({
            'response': response,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rate', methods=['POST'])
def rate():
    try:
        data = request.json
        rating = data.get('rating')
        session_id = data.get('session_id', 'default')
        
        if not rating or not (1 <= rating <= 5):
            return jsonify({'error': 'Rating must be 1-5'}), 400
        
        result = ghost.rate_last_response(rating, session_id)
        
        return jsonify({
            'message': result,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        stats = ghost.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        limit = int(request.args.get('limit', 50))
        count = ghost.memory.conversations.count()
        
        if count == 0:
            return jsonify({'conversations': [], 'total': 0, 'success': True})
        
        conversations = ghost.memory.conversations.get(limit=min(limit, count))
        history = []
        
        if conversations and 'documents' in conversations:
            for i, doc in enumerate(conversations['documents']):
                metadata = conversations['metadatas'][i] if i < len(conversations['metadatas']) else {}
                history.append({
                    'content': doc,
                    'timestamp': metadata.get('timestamp', ''),
                    'rating': metadata.get('rating', 'unrated'),
                    'preview': doc[:100]
                })
        
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'conversations': history,
            'total': len(history),
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'alive', 'ghost': ' '})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)