"""
GHOST AI - Your Self-Improving AI Assistant
Built to learn, remember, and get smarter with every interaction
"""

import os
import json
import datetime
from typing import List, Dict, Optional
import chromadb
import anthropic
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()


class GhostMemory:
    """Ghost's long-term memory system"""
    
    def __init__(self, persist_directory="./ghost_memory"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create memory collections
        try:
            self.conversations = self.client.get_collection("conversations")
        except:
            self.conversations = self.client.create_collection(
                name="conversations",
                metadata={"description": "All conversations with user"}
            )
        
        try:
            self.knowledge = self.client.get_collection("knowledge")
        except:
            self.knowledge = self.client.create_collection(
                name="knowledge",
                metadata={"description": "Important learnings and insights"}
            )
    
    def remember_conversation(self, user_msg: str, ghost_msg: str, rating: Optional[int] = None):
        """Remember a conversation exchange"""
        memory_id = f"mem_{datetime.datetime.now().timestamp()}"
        
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "rating": str(rating) if rating else "unrated",
            "preview": user_msg[:100]
        }
        
        full_exchange = f"User: {user_msg}\n\nGhost: {ghost_msg}"
        
        self.conversations.add(
            documents=[full_exchange],
            metadatas=[metadata],
            ids=[memory_id]
        )
        
        return memory_id
    
    def recall(self, query: str, n_results: int = 3) -> List[Dict]:
        """Recall relevant past conversations"""
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
    
    def learn(self, insight: str, category: str = "general"):
        """Store important learnings"""
        learn_id = f"learn_{datetime.datetime.now().timestamp()}"
        
        self.knowledge.add(
            documents=[insight],
            metadatas=[{
                "category": category,
                "timestamp": datetime.datetime.now().isoformat()
            }],
            ids=[learn_id]
        )


class UserProfile:
    """Your personal profile that Ghost learns about you"""
    
    def __init__(self, profile_path="./ghost_profile.json"):
        self.profile_path = profile_path
        self.profile = self.load()
    
    def load(self) -> Dict:
        """Load your profile"""
        if os.path.exists(self.profile_path):
            with open(self.profile_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "name": "User",
                "preferences": {},
                "interests": [],
                "goals": [],
                "communication_style": "adaptive",
                "total_chats": 0,
                "created": datetime.datetime.now().isoformat()
            }
    
    def save(self):
        """Save your profile"""
        with open(self.profile_path, 'w') as f:
            json.dump(self.profile, f, indent=2)
    
    def add_chat(self):
        """Count another conversation"""
        self.profile["total_chats"] += 1
        self.save()
    
    def update(self, key: str, value):
        """Update any profile field"""
        self.profile[key] = value
        self.save()


class Ghost:
    """The Ghost AI - Your self-improving assistant"""
    
    def __init__(self):
        # Get API key
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env file!")
        
        # Initialize components
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.memory = GhostMemory()
        self.profile = UserProfile()
        self.session = []  # Current conversation
    
    def _build_context(self, user_msg: str) -> str:
        """Build enhanced context from memory"""
        context = []
        
        # Add profile
        context.append(f"USER PROFILE:\n{json.dumps(self.profile.profile, indent=2)}")
        
        # Recall relevant memories
        memories = self.memory.recall(user_msg, n_results=3)
        
        if memories:
            context.append("\nRELEVANT MEMORIES:")
            for i, mem in enumerate(memories, 1):
                rating = mem['rating']
                context.append(f"\n[Memory {i}] [Rating: {rating}]\n{mem['content'][:400]}")
        
        return "\n".join(context)
    
    def talk(self, user_message: str) -> str:
        """Chat with Ghost"""
        self.profile.add_chat()
        
        # Build system prompt
        system = """You are Ghost, a self-improving AI assistant. 

Your unique capabilities:
- You remember EVERYTHING from past conversations
- You learn the user's preferences and adapt to them
- You get better with every interaction
- You have access to the user's profile and conversation history

Be helpful, adaptive, and personal. Use the context provided to give increasingly tailored responses."""
        
        # Add memory context
        context = self._build_context(user_message)
        system += f"\n\nCONTEXT:\n{context}"
        
        # Build messages
        messages = self.session.copy()
        messages.append({"role": "user", "content": user_message})
        
        # Call Claude API
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system,
            messages=messages
        )
        
        ghost_response = response.content[0].text
        
        # Update session
        self.session.append({"role": "user", "content": user_message})
        self.session.append({"role": "assistant", "content": ghost_response})
        
        # Keep session manageable (last 10 exchanges)
        if len(self.session) > 20:
            self.session = self.session[-20:]
        
        return ghost_response
    
    def rate(self, rating: int):
        """Rate Ghost's last response (1-5)"""
        if len(self.session) >= 2:
            user_msg = self.session[-2]["content"]
            ghost_msg = self.session[-1]["content"]
            
            self.memory.remember_conversation(user_msg, ghost_msg, rating)
            
            # Learn from highly rated responses
            if rating >= 4:
                self.memory.learn(
                    f"User appreciated: {ghost_msg[:300]}",
                    category="good_response"
                )
                return f"⭐ Rated {rating}/5 - I'll remember what worked!"
            else:
                return f"⭐ Rated {rating}/5 - I'll learn from this."
        
        return "No response to rate yet."
    
    def reset_session(self):
        """Clear current chat (keeps long-term memory)"""
        self.session = []
        return "Session cleared. My long-term memory remains intact."
    
    def stats(self) -> Dict:
        """Get Ghost's statistics"""
        return {
            "Total conversations": self.profile.profile["total_chats"],
            "Stored memories": self.memory.conversations.count(),
            "Learned insights": self.memory.knowledge.count(),
            "Interests tracked": len(self.profile.profile["interests"]),
            "Session length": len(self.session) // 2
        }


def main():
    """Run Ghost AI"""
    print("=" * 60)
    print("👻 GHOST AI - Your Self-Improving Assistant")
    print("=" * 60)
    print("\nCommands:")
    print("  /rate 1-5   - Rate my last response")
    print("  /stats      - Show my statistics")
    print("  /clear      - Clear current session")
    print("  /quit       - Exit (memory saved automatically)")
    print("=" * 60)
    
    try:
        ghost = Ghost()
        total = ghost.profile.profile["total_chats"]
        print(f"\n✓ Ghost initialized!")
        print(f"✓ We've talked {total} times before\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure your .env file has:")
        print("ANTHROPIC_API_KEY=your_key_here")
        return
    
    # Main loop
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                cmd = user_input.lower().split()
                
                if cmd[0] == '/quit':
                    print("\n👻 Ghost: Until next time... *fades away*")
                    break
                
                elif cmd[0] == '/rate':
                    if len(cmd) > 1 and cmd[1].isdigit():
                        rating = int(cmd[1])
                        if 1 <= rating <= 5:
                            result = ghost.rate(rating)
                            print(f"\n{result}")
                        else:
                            print("\n❌ Rating must be 1-5")
                    else:
                        print("\n❌ Usage: /rate 1-5")
                
                elif cmd[0] == '/stats':
                    stats = ghost.stats()
                    print("\n📊 GHOST STATISTICS:")
                    for key, value in stats.items():
                        print(f"  • {key}: {value}")
                
                elif cmd[0] == '/clear':
                    result = ghost.reset_session()
                    print(f"\n{result}")
                
                else:
                    print(f"\n❌ Unknown command: {cmd[0]}")
                
                continue
            
            # Regular chat
            print("\n👻 Ghost: ", end="", flush=True)
            response = ghost.talk(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👻 Ghost: Until next time... *fades away*")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try again or type /quit to exit")


if __name__ == "__main__":
    main()