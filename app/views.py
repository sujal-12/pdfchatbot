import os
import json
import re
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from google import genai
from google.genai.errors import APIError
from google.genai.types import Part

# --- CONFIGURATION ---
# IMPORTANT: Replace with your actual Gemini API Key
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBHC9zkhP5gLgEHtqu84Aln8Zv4oKN8s6U")
MODEL_NAME = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = (
    "You are a friendly and concise assistant named 'GemBot'. "
    "Your primary role is to answer questions and generate content based *only* "
    "on the provided PDF document. Keep your answers brief and conversational."
)

# Directory to store uploaded files and JSON data (Ensure this exists and is writable)
MEDIA_ROOT = 'media/'
if not os.path.exists(MEDIA_ROOT):
    os.makedirs(MEDIA_ROOT)

# --- Helper function to clean response ---
def clean_text_response(text: str) -> str:
    """
    Cleans the model response to remove Markdown, code fences, asterisks, and unwanted symbols.
    Returns plain text.
    """
    if not text:
        return ""
    
    # Remove code fences (``` or ```json)
    text = re.sub(r"```(json)?", "", text)
    
    # Remove */ or /* style comments
    text = text.replace("*/", "").replace("/*", "")
    
    # Remove Markdown bold/italic asterisks
    text = re.sub(r'\*+', '', text)
    
    # Remove extra whitespace at start/end and multiple newlines
    text = re.sub(r'\n\s*\n', '\n', text)  # collapse multiple empty lines
    text = text.strip()
    
    return text


class GeminiPDFChatManager:
    """Manages the Gemini Client, file uploads, and chat sessions."""
    
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)
        self.uploaded_file = None
        self.chat_session = None
        
    def _get_pdf_content_json(self, pdf_file_part):
        """Generates a JSON summary of the PDF content."""
        prompt = (
            "Analyze the provided PDF document. Generate a comprehensive JSON object "
            "that summarizes its content, structure, and key topics. "
            "The JSON should include fields like 'title', 'summary', 'key_sections', and 'important_terms'. "
            "Output *only* the valid JSON object, without any surrounding text or explanation."
        )
        try:
            # Use the model to generate the JSON summary from the PDF
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[pdf_file_part, prompt],
            )
            
            # Attempt to clean up and parse the JSON response
            json_text = clean_text_response(response.text)
            
            return json.loads(json_text)
            
        except (APIError, json.JSONDecodeError, Exception) as e:
            print(f"Error generating or parsing PDF content JSON: {e}")
            return {"error": "Failed to generate PDF content summary."}

    def initialize_chat(self, file_path):
        """Uploads the PDF, generates content JSON, and initializes the chat session."""
        try:
            # 1. Upload File
            display_name = os.path.basename(file_path)
            self.uploaded_file = self.client.files.upload(
                file=file_path,
                config={"display_name": display_name, "mime_type": "application/pdf"},
            )

            # 2. Get JSON Content and Save to Disk
            pdf_content_data = self._get_pdf_content_json(self.uploaded_file)
            json_path = os.path.join(MEDIA_ROOT, f"{display_name}_content.json")
            with open(json_path, 'w') as f:
                json.dump(pdf_content_data, f, indent=4)
            print(f"PDF content JSON saved to: {json_path}")
            
            # 3. Initialize Chat Session
            self.chat_session = self.client.chats.create(
                model=self.model_name,
                config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            )
            # Send the initial message with the file to set context
            self.chat_session.send_message([self.uploaded_file, "Hello! I am ready to answer questions about this PDF."])

            return True, "PDF uploaded, content analyzed, and chat initialized."

        except APIError as e:
            self.cleanup()
            return False, f"API Error: {e}"
        except Exception as e:
            self.cleanup()
            return False, f"Unexpected Error: {e}"

    def get_response(self, user_prompt):
        """Sends a message to the chat session and returns a clean plain-text response."""
        if not self.chat_session:
            return "Error: Chat not initialized. Please upload a PDF first."

        try:
            response = self.chat_session.send_message(user_prompt)
            # Clean the response before returning
            clean_response = clean_text_response(response.text)
            return clean_response
        except Exception as e:
            return f"Error communicating with Gemini: {e}"

    def cleanup(self):
        """Deletes the uploaded file from the API."""
        if self.uploaded_file:
            try:
                self.client.files.delete(name=self.uploaded_file.name)
                print(f"Cleaned up uploaded file: {self.uploaded_file.display_name}")
            except Exception as e:
                print(f"Warning: Could not delete uploaded file: {e}")
            self.uploaded_file = None
            self.chat_session = None

# Global manager instance (Simplifies state management for this example)
manager = GeminiPDFChatManager(API_KEY, MODEL_NAME)

# --- DJANGO VIEWS ---

def index(request):
    """Renders the main chat page."""
    return render(request, 'index.html')

@csrf_exempt
def upload_pdf_view(request):
    """Handles PDF file upload and chat initialization."""
    global manager
    if request.method == 'POST' and 'pdf_file' in request.FILES:
        # Cleanup any previous session
        manager.cleanup()
        
        pdf_file = request.FILES['pdf_file']
        file_path = os.path.join(MEDIA_ROOT, pdf_file.name)
        
        # Save the uploaded file to disk temporarily
        with open(file_path, 'wb+') as destination:
            for chunk in pdf_file.chunks():
                destination.write(chunk)
                
        success, message = manager.initialize_chat(file_path)
        
        # Clean up the local file after upload to API (optional, but good practice)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        if success:
            return JsonResponse({'status': 'success', 'message': message})
        else:
            return JsonResponse({'status': 'error', 'message': message}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request or no file provided.'}, status=400)

@csrf_exempt
def chat_view(request):
    """Handles incoming chat messages (text or voice-to-text)."""
    global manager
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_prompt = data.get('prompt')
            
            if not user_prompt:
                return JsonResponse({'status': 'error', 'message': 'No prompt provided.'}, status=400)
            
            response_text = manager.get_response(user_prompt)
            
            return JsonResponse({'status': 'success', 'response': response_text})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON format.'}, status=400)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Server error: {e}'}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=400)

@csrf_exempt
def cleanup_view(request):
    """Deletes the file from the Gemini API."""
    global manager
    manager.cleanup()
    return JsonResponse({'status': 'success', 'message': 'Chat session cleaned up.'})
