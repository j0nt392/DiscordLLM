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
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
openai_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_key)  # Singleton OpenAI client
TOKEN = os.getenv("TOKEN")

# Discord bot setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)
vdb = VDB()

bot = commands.Bot(command_prefix="!", intents=intents)

# Track user usage
user_usage = {}
ONE_MINUTE = 60
DAILY_LIMIT = 20

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    for guild in bot.guilds:
        for channel in guild.text_channels:
            if channel.name.lower() == "general":
                await channel.send(f"Indexing {channel.name}...")
                await vdb.index_channel_history(channel)
    for guild in bot.guilds:
        for channel in guild.text_channels:
            if channel.name.lower() == "general":
                await channel.send("All channels indexed. Bot is ready\n\nInstructions: \n!search <query> to do a semantic search on this servers history.\n!summarize <youtubelink> to summarize a youtubevideo. (requires that the video has transcripts enabled).\n-yt_db <youtubelink> to index youtube-transcripts.\n\nAsk me anything and I shall provide you answers that are based on the history of this server as well as youtube-transcripts. For suggestions or feedback, please contact @jonatan. Thank you!")

@bot.event
async def on_message(message):
    logger.info(f"Message received: {message.content}")
    
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Ignore messages starting with !search or !yt_db
    if message.content.startswith('!search') or message.content.startswith('!yt_db') or message.content.startswith('!summarize'):
        await bot.process_commands(message)
        return

    # Add the message to the vector database
    vdb.add_vector(message)
    await bot.process_commands(message)

@bot.command()
@commands.cooldown(1, 5, commands.BucketType.user)  # Cooldown
async def search(ctx, *, query):
    user_id = str(ctx.author.id)
    current_time = time.time()

    if user_id not in user_usage:
        user_usage[user_id] = []

    user_usage[user_id] = [
        t for t in user_usage[user_id] if current_time - t < 86400]

    if len(user_usage[user_id]) >= 5:
        await ctx.send("You have reached your daily quota. Please try again tomorrow.")
        return

    user_usage[user_id].append(current_time)

    query_vec = vdb.vectorize(query, is_query=True)
    _, indices = vdb.db.search(np.array([query_vec]), k=5)
    results = []
    for idx in indices[0]:
        if idx != -1:
            msg_meta = vdb.message_storage[idx]
            if 'video_title' in msg_meta:
                link = f"https://www.youtube.com/watch?v={
                    msg_meta['message_id']}"
                results.append(f"Video: {msg_meta['video_title']}\nChunk: {
                               msg_meta['content'][:200]}... [Link]({link})")
            else:
                link = f"https://discord.com/channels/{msg_meta['guild_id']}/{
                    msg_meta['channel_id']}/{msg_meta['message_id']}"
                results.append(
                    f"Message: {msg_meta['content']} [Link]({link})")
        else:
            results.append("Message not found")
    response = vdb.get_response(query, results)
    await ctx.send(response)


@search.error
async def search_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"Please wait {error.retry_after:.2f} seconds before using this command again.")


@bot.command()
@commands.cooldown(1, 60, commands.BucketType.user)  # Cooldown
async def yt_db(ctx, url):
    user_id = str(ctx.author.id)
    current_time = time.time()

    if user_id not in user_usage:
        user_usage[user_id] = []

    user_usage[user_id] = [
        t for t in user_usage[user_id] if current_time - t < 86400]

    if len(user_usage[user_id]) >= 5:
        await ctx.send("You have reached your daily quota. Please try again tomorrow.")
        return

    user_usage[user_id].append(current_time)

    await ctx.send("Fetching transcript from YouTube...")

    video_id = get_video_id(url)
    if not video_id:
        await ctx.send("Invalid YouTube URL.")
        return

    try:
        transcript = fetch_youtube_transcript(video_id)
        if len(transcript) > 10000:
            await ctx.send("Transcript is too long to process. Please provide a shorter video.")
            return
    except Exception as e:
        await ctx.send(f"Failed to fetch transcript: {e}")
        return

    # You can modify this to fetch the actual title if needed
    video_title = f"Video ID: {video_id}"
    await ctx.send("Adding transcript to the database...")
    add_transcript_to_db(transcript, video_id, video_title)

    await ctx.send("Transcript added to the database.")


@yt_db.error
async def yt_db_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"Please wait {error.retry_after:.2f} seconds before using this command again.")

# function to summarize text using gpt-4


async def sum_vid(text, chunk_size=8000):
    text_chunks = [text[i:i+chunk_size]
                   for i in range(0, len(text), chunk_size)]

    summaries = []
    for chunk in text_chunks:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following youtube-transcript:\n\n{chunk}"}]
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        summaries.append(response.choices[0].message.content.strip())
    combined_summary = " ".join(summaries)

    if len(combined_summary) > chunk_size:
        final_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following youtube-transcript:\n\n{chunk}"}]        
        final_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=final_messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        final_summary = final_response.choices[0].message.content.strip()
        return final_summary
    return combined_summary


@bot.command()
# Cooldown set to 120 seconds
@commands.cooldown(1, 120, commands.BucketType.user)
async def summarize(ctx, url):
    user_id = str(ctx.author.id)
    current_time = time.time()

    if user_id not in user_usage:
        user_usage[user_id] = []

    user_usage[user_id] = [
        t for t in user_usage[user_id] if current_time - t < 86400]

    if len(user_usage[user_id]) >= 5:
        await ctx.send("You have reached your daily quota. Please try again tomorrow.")
        return

    user_usage[user_id].append(current_time)

    await ctx.send("Fetching transcript from YouTube...")

    video_id = get_video_id(url)
    if not video_id:
        await ctx.send("Invalid YouTube URL.")
        return

    try:
        transcript = await fetch_youtube_transcript(video_id)
        if len(transcript) > 1000000:
            await ctx.send("Transcript is too long to process. Please provide a shorter video.")
            return
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


def chunk_text(text, chunk_size=10000, overlap=500):
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

# Function to fetch video ID from YouTube URL


def get_video_id(url):
    video_id_match = re.match(
        r'(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+(?:v=|/)([a-zA-Z0-9_-]{11})', url)
    if video_id_match:
        return video_id_match.group(4)
    return None


# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

bot.run(TOKEN)
