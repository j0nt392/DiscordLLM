# invite link 
#https://discord.com/oauth2/authorize?client_id=1240321223418970286&permissions=274877975552&scope=bot
import discord
from discord.ext import commands
import numpy as np
import os 
from dotenv import load_dotenv
from openai import OpenAI
import logging
from v_database import VDB
import signal
import asyncio
from huggingface_hub import hf_hub_download
from youtube_transcript_api import YouTubeTranscriptApi
import re
import boto3 
import json 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
openai_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_key)  # Singleton OpenAI client
TOKEN = os.getenv("TOKEN")
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# Discord bot setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)
boto_client = boto3.client('bedrock-runtime', region_name="eu-west-3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_access_key)
model_id = "mistral.mistral-7b-instruct-v0:2"
vdb = VDB()

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    for guild in bot.guilds:
        for channel in guild.text_channels:
            bot.loop.create_task(vdb.index_channel_history(channel))

# ask using bedrock 
@bot.command()
async def ask(ctx, *, question):
    await ctx.send("Asking Bedrock...")
    try:
        # Define the prompt for the model
        prompt = question

        # Embed the prompt in Mistral's instruction format
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        # Format the request payload using the model's native structure
        native_request = {
            "prompt": formatted_prompt,
            "max_tokens": 512,
            "temperature": 0.5,
        }

        # Convert the native request to JSON
        request = json.dumps(native_request)

        # Invoke the model with the request
        response = boto_client.invoke_model(modelId=model_id, body=request)

        # Decode the response body
        model_response = json.loads(response["body"].read())

        # Extract the response text
        response_text = model_response["outputs"][0]["text"]
        await ctx.send(response_text)
    except Exception as e:
        await ctx.send(f"Error: {e}")

@bot.command()
async def search(ctx, *, query):
    query_vec = vdb.vectorize(query, is_query=True)
    _, indices = vdb.db.search(np.array([query_vec]), k=5)
    results = []
    for idx in indices[0]:
        if idx != -1:
            msg_meta = vdb.message_storage[idx]
            if 'video_title' in msg_meta:
                link = f"https://www.youtube.com/watch?v={msg_meta['message_id']}"
                results.append(f"Video: {msg_meta['video_title']}\nChunk: {msg_meta['content'][:200]}... [Link]({link})")
            else:
                link = f"https://discord.com/channels/{msg_meta['guild_id']}/{msg_meta['channel_id']}/{msg_meta['message_id']}"
                results.append(f"Message: {msg_meta['content']} [Link]({link})")
        else:
            results.append("Message not found")
    response = vdb.get_response(query, results)
    await ctx.send(response)


@bot.command()
async def yt_db(ctx, url):
    await ctx.send("Fetching transcript from YouTube...")
    
    video_id = get_video_id(url)
    if not video_id:
        await ctx.send("Invalid YouTube URL.")
        return
    
    try:
        transcript = fetch_youtube_transcript(video_id)
    except Exception as e:
        await ctx.send(f"Failed to fetch transcript: {e}")
        return
    
    video_title = f"Video ID: {video_id}"  # You can modify this to fetch the actual title if needed
    await ctx.send("Adding transcript to the database...")
    add_transcript_to_db(transcript, video_id, video_title)
    
    await ctx.send("Transcript added to the database.")


@bot.command()
async def summarize(ctx, url):
    await ctx.send("Fetching transcript from YouTube...")
    
    video_id = get_video_id(url)
    if not video_id:
        await ctx.send("Invalid YouTube URL.")
        return
    
    try:
        transcript = await fetch_youtube_transcript(video_id)
    except Exception as e:
        await ctx.send(f"Failed to fetch transcript: {e}")
        return
    
    await ctx.send("Summarizing transcript...")

    summary = await sum_vid(transcript)
    await ctx.send(summary)

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

def chunk_text(text, chunk_size=3000, overlap=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def add_transcript_to_db(transcript, video_id, video_title):
    chunks = chunk_text(transcript)
    for chunk in chunks:
        message = {
            'content': chunk,
            'channel_id': None,
            'message_id': video_id,
            'guild_id': None,
            'video_title': video_title
        }
        vdb.add_vector_transcript(message)

# function to summarize youtube transcript with gpt3.5-instruct model
async def sum_vid(text, chunk_size = 10000):    
    # Split the text into chunks
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Summarize each chunk
    summaries = []
    for chunk in text_chunks:
        prompt = f"Summarize the following text:\n\n{chunk}"
        response = openai_client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        summary = response.choices[0].text.strip()
        summaries.append(summary)
    
    # Combine all summaries
    combined_summary = " ".join(summaries)
    
    # If the combined summary is still too long, summarize it again
    if len(combined_summary) > chunk_size:
        final_prompt = f"Summarize the following text in no more than 350 words:\n\n{combined_summary}"
        final_response = openai_client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=final_prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        final_summary = final_response.choices[0].text.strip()
        return final_summary
    
    return combined_summary
# Function to fetch video ID from YouTube URL
def get_video_id(url):
    video_id_match = re.match(r'(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+(?:v=|/)([a-zA-Z0-9_-]{11})', url)
    if video_id_match:
        return video_id_match.group(4)
    return None
# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

bot.run(TOKEN)


# @client.event
# async def on_ready():
#     logger.info(f'Logged in as {client.user}')
#     for guild in client.guilds:
#         for channel in guild.text_channels:
#             await vdb.index_channel_history(channel)

# @client.event
# async def on_disconnect():
#     vdb.save_faiss_index()
#     vdb.save_message_storage()
#     logger.info('Saved FAISS index and message storage to disk.')

# @client.event
# async def on_message(message):
#     if message.author == client.user:
#         return

#     if message.content.startswith("!search "):
#         query = message.content[len("!search "):]
#         query_vec = vdb.vectorize(query)
#         _, indices = vdb.db.search(np.array([query_vec]), k=5)
#         results = []
#         for idx in indices[0]:
#             if idx != -1:
#                 msg_meta = vdb.message_storage[idx]
#                 link = f"https://discord.com/channels/{msg_meta['guild_id']}/{msg_meta['channel_id']}/{msg_meta['message_id']}"
#                 results.append(f"Message: {msg_meta['content']} [Link]({link})")
#             else:
#                 results.append("Message not found")
#         response = vdb.get_response(query, results)
#         await message.channel.send(response)
#     else:
#         vdb.add_vector(message)

# # Handle shutdown signals
# def handle_shutdown(signal, frame):
#     vdb.save_faiss_index()
#     vdb.save_message_storage()
#     logger.info('Saved FAISS index and message storage to disk.')
#     loop = asyncio.get_event_loop()
#     loop.stop()

# # Register signal handlers
# signal.signal(signal.SIGINT, handle_shutdown)
# signal.signal(signal.SIGTERM, handle_shutdown)

# client.run(TOKEN)
