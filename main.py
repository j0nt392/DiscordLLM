# invite link 
#https://discord.com/oauth2/authorize?client_id=1240321223418970286&permissions=274877975552&scope=bot
import discord
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import os 
from dotenv import load_dotenv
import ollama 
import requests 
from openai import OpenAI
import logging
from v_database import VDB
import signal
import asyncio
from huggingface_hub import hf_hub_download


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
openai_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_key)  # Singleton OpenAI client
TOKEN = os.getenv("TOKEN")

# Discord bot setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)
vdb = VDB()

@client.event
async def on_ready():
    logger.info(f'Logged in as {client.user}')
    for guild in client.guilds:
        for channel in guild.text_channels:
            await vdb.index_channel_history(channel)

@client.event
async def on_disconnect():
    vdb.save_faiss_index()
    vdb.save_message_storage()
    logger.info('Saved FAISS index and message storage to disk.')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!search "):
        query = message.content[len("!search "):]
        query_vec = vdb.vectorize(query)
        _, indices = vdb.db.search(np.array([query_vec]), k=5)
        results = []
        for idx in indices[0]:
            if idx != -1:
                msg_meta = vdb.message_storage[idx]
                link = f"https://discord.com/channels/{msg_meta['guild_id']}/{msg_meta['channel_id']}/{msg_meta['message_id']}"
                results.append(f"Message: {msg_meta['content']} [Link]({link})")
            else:
                results.append("Message not found")
        response = vdb.get_response(query, results)
        await message.channel.send(response)
    else:
        vdb.add_vector(message)

# Handle shutdown signals
def handle_shutdown(signal, frame):
    vdb.save_faiss_index()
    vdb.save_message_storage()
    logger.info('Saved FAISS index and message storage to disk.')
    loop = asyncio.get_event_loop()
    loop.stop()

# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

client.run(TOKEN)
