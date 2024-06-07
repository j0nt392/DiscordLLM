import asyncio 
from v_database import VDB
from youtube_transcript_api import YouTubeTranscriptApi


# Handle shutdown signals
def handle_shutdown(signal, frame):
    vdb.save_faiss_index()
    vdb.save_message_storage()
    print('Saved FAISS index and message storage to disk.')
    loop = asyncio.get_event_loop()
    loop.stop()

# Function to fetch YouTube transcripts
async def fetch_youtube_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([entry['text'] for entry in transcript])
    return transcript_text

# Function to fetch video ID from YouTube URL
def get_video_id(url):
    video_id_match = re.match(r'(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+(?:v=|/)([a-zA-Z0-9_-]{11})', url)
    if video_id_match:
        return video_id_match.group(4)
    return None