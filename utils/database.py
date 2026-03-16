import sys
import os
import json
from datetime import datetime
from typing import Optional, Dict, List, Any
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import SUPABASE_URL, SUPABASE_KEY

_client = None


def _get_client():
    """Initialize and return Supabase client."""
    global _client
    if _client is None:
        try:
            from supabase import create_client, Client
            if SUPABASE_URL and SUPABASE_KEY:
                _client = create_client(SUPABASE_URL, SUPABASE_KEY)
            else:
                _client = None  # Fallback to local storage
        except Exception as e:
            print(f"Supabase initialization failed: {e}. Using local storage.")
            _client = None
    return _client


def save_conversation(user_id: str, conversation_data: Dict[str, Any]) -> bool:
    """Save conversation to database."""
    try:
        client = _get_client()
        if not client:
            return _save_local("conversations", user_id, conversation_data)
        
        data = {
            "user_id": user_id,
            "title": conversation_data.get("title", "Untitled Conversation"),
            "messages": json.dumps(conversation_data.get("messages", [])),
            "metadata": json.dumps(conversation_data.get("metadata", {})),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = client.table("conversations").insert(data).execute()
        return len(result.data) > 0
    except Exception as e:
        print(f"Error saving conversation: {e}")
        return _save_local("conversations", user_id, conversation_data)


def load_conversations(user_id: str) -> List[Dict[str, Any]]:
    """Load all conversations for a user."""
    try:
        client = _get_client()
        if not client:
            return _load_local("conversations", user_id)
        
        result = client.table("conversations").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
        
        conversations = []
        for row in result.data:
            conversations.append({
                "id": row.get("id"),
                "title": row.get("title"),
                "messages": json.loads(row.get("messages", "[]")),
                "metadata": json.loads(row.get("metadata", "{}")),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at")
            })
        return conversations
    except Exception as e:
        print(f"Error loading conversations: {e}")
        return _load_local("conversations", user_id)


def save_user_profile(user_id: str, profile_data: Dict[str, Any]) -> bool:
    """Save user profile."""
    try:
        client = _get_client()
        if not client:
            return _save_local("profiles", user_id, profile_data)
        
        data = {
            "user_id": user_id,
            "profile_name": profile_data.get("name", "Default"),
            "career_path": profile_data.get("career_path", ""),
            "data": json.dumps(profile_data),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Upsert based on user_id and profile_name
        result = client.table("user_profiles").upsert(data, on_conflict="user_id,profile_name").execute()
        return len(result.data) > 0
    except Exception as e:
        print(f"Error saving profile: {e}")
        return _save_local("profiles", user_id, profile_data)


def load_user_profiles(user_id: str) -> List[Dict[str, Any]]:
    """Load all profiles for a user."""
    try:
        client = _get_client()
        if not client:
            return _load_local("profiles", user_id)
        
        result = client.table("user_profiles").select("*").eq("user_id", user_id).execute()
        
        profiles = []
        for row in result.data:
            profiles.append({
                "id": row.get("id"),
                "name": row.get("profile_name"),
                "career_path": row.get("career_path"),
                "data": json.loads(row.get("data", "{}")),
                "updated_at": row.get("updated_at")
            })
        return profiles
    except Exception as e:
        print(f"Error loading profiles: {e}")
        return _load_local("profiles", user_id)


def save_bookmark(user_id: str, bookmark_data: Dict[str, Any]) -> bool:
    """Save a bookmark."""
    try:
        client = _get_client()
        if not client:
            return _save_local("bookmarks", user_id, bookmark_data)
        
        data = {
            "user_id": user_id,
            "title": bookmark_data.get("title", ""),
            "content": bookmark_data.get("content", ""),
            "source": bookmark_data.get("source", ""),
            "metadata": json.dumps(bookmark_data.get("metadata", {})),
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = client.table("bookmarks").insert(data).execute()
        return len(result.data) > 0
    except Exception as e:
        print(f"Error saving bookmark: {e}")
        return _save_local("bookmarks", user_id, bookmark_data)


def load_bookmarks(user_id: str) -> List[Dict[str, Any]]:
    """Load all bookmarks for a user."""
    try:
        client = _get_client()
        if not client:
            return _load_local("bookmarks", user_id)
        
        result = client.table("bookmarks").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        
        bookmarks = []
        for row in result.data:
            bookmarks.append({
                "id": row.get("id"),
                "title": row.get("title"),
                "content": row.get("content"),
                "source": row.get("source"),
                "metadata": json.loads(row.get("metadata", "{}")),
                "created_at": row.get("created_at")
            })
        return bookmarks
    except Exception as e:
        print(f"Error loading bookmarks: {e}")
        return _load_local("bookmarks", user_id)


def _save_local(table: str, user_id: str, data: Dict[str, Any]) -> bool:
    """Fallback local storage."""
    try:
        os.makedirs("local_db", exist_ok=True)
        filepath = f"local_db/{table}_{user_id}.json"
        existing = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing = json.load(f)
        existing.append({**data, "id": len(existing), "created_at": datetime.utcnow().isoformat()})
        with open(filepath, 'w') as f:
            json.dump(existing, f, indent=2)
        return True
    except Exception as e:
        print(f"Local save error: {e}")
        return False


def _load_local(table: str, user_id: str) -> List[Dict[str, Any]]:
    """Load from local storage."""
    try:
        filepath = f"local_db/{table}_{user_id}.json"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return []
    except Exception:
        return []

