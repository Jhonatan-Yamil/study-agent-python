# backend/gemini_client.py
import os
from typing import List, Dict
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import google.genai as genai

# Cargar variables de entorno
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Función para búsqueda web
def perform_web_search(query: str, max_results: int = 6) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                if not isinstance(r, dict):
                    continue
                title = r.get("title", "")
                href = r.get("href", "")
                body = r.get("body", "")
                if title and href:
                    results.append({"title": title, "href": href, "body": body})
        return results
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return []

# Cliente Gemini actualizado
class GeminiClient:
    def __init__(self):
        try:
            self.chat = genai.Chat(model="gemini-1.5")  # Nuevo objeto de chat
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
            self.chat = None

    def generate_response(self, user_input: str) -> str:
        if not self.chat:
            return "AI service is not configured correctly."

        try:
            text = user_input.strip()

            # Búsqueda web si el mensaje inicia con "search:"
            search_query = None
            if text.lower().startswith("search:"):
                search_query = text.split(":", 1)[1].strip()

            if search_query:
                web_results = perform_web_search(search_query)
                if not web_results:
                    return "I could not retrieve web results right now. Please try again."

                # Construir contexto de búsqueda
                refs_lines = [
                    f"[{i+1}] {item['title']} — {item['href']}\n{item['body']}"
                    for i, item in enumerate(web_results)
                ]
                refs_block = "\n\n".join(refs_lines)

                prompt = (
                    f"You are an AI research assistant. Use the provided web search results to answer the user query. "
                    f"Synthesize concisely, cite sources inline like [1], [2] where relevant, and include a brief summary.\n\n"
                    f"Web Results:\n{refs_block}\n\nQuery:\n{search_query}"
                )
            else:
                prompt = text  # Mensaje normal

            response = self.chat.send_message(prompt)
            return response.output_text

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error processing your request."