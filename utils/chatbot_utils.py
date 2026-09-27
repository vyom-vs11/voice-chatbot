"""
Chatbot Audio and Speech Recognition Utilities
Provides server-side audio file transcription using SpeechRecognition and Google Speech API.
"""

import os
import io
import speech_recognition as sr


def transcribe_audio_file(audio_bytes_or_file) -> dict:
    """
    Transcribes an audio file or bytes using the SpeechRecognition library
    and the Google Web Speech Recognition API.
    Returns:
        dict: {'success': bool, 'text': str, 'error': str}
    """
    recognizer = sr.Recognizer()
    try:
        if isinstance(audio_bytes_or_file, bytes):
            audio_io = io.BytesIO(audio_bytes_or_file)
            with sr.AudioFile(audio_io) as source:
                audio_data = recognizer.record(source)
        elif hasattr(audio_bytes_or_file, "read"):
            with sr.AudioFile(audio_bytes_or_file) as source:
                audio_data = recognizer.record(source)
        elif isinstance(audio_bytes_or_file, str) and os.path.exists(audio_bytes_or_file):
            with sr.AudioFile(audio_bytes_or_file) as source:
                audio_data = recognizer.record(source)
        else:
            return {"success": False, "text": "", "error": "Invalid audio input format."}

        # Recognize speech using Google Speech API
        recognized_text = recognizer.recognize_google(audio_data)
        return {"success": True, "text": recognized_text, "error": None}

    except sr.UnknownValueError:
        return {"success": False, "text": "", "error": "Speech was unintelligible or no words detected."}
    except sr.RequestError as e:
        return {"success": False, "text": "", "error": f"Speech Recognition API service error: {e}"}
    except Exception as e:
        return {"success": False, "text": "", "error": f"Audio processing error: {str(e)}"}
