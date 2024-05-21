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
from vdb import vectorize, add_vector, get_response, db, message_storage

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
openai_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_key)  # Singleton OpenAI client
TOKEN = os.getenv("TOKEN")

# Discord bot setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

async def index_channel_history(channel):
    try:
        async for message in channel.history(limit=1000):
            # Check if message starts with '!search' or if the author is a bot
            if not message.content.startswith('!search') and not message.author.bot:
                add_vector(message)
        print(f'indexed messages from {channel.name}')
    except discord.Forbidden:
        print(f'cannot access messages from {channel.name}')
    except Exception as e:
        print(f'error indexing messages from {channel.name}: {e}')


client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    for guild in client.guilds:
        for channel in guild.text_channels:
            await index_channel_history(channel)

# Assuming 'client' is your Discord bot instance
conversation_histories = {}

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Initialize conversation history if not already present
    if message.author.id not in conversation_histories:
        conversation_histories[message.author.id] = []

    if message.content.startswith("!chat "):
        user_input = message.content[len("!chat "):]
        # Append user message to history
        conversation_histories[message.author.id].append({"role": "user", "content": user_input})
        
        # Prepare the payload for Ollama API
        payload = {
            "model": "llama3",
            "messages": conversation_histories[message.author.id],
            "stream": False
        }
        url = "http://localhost:11434/api/chat"
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            print(response_data)
            response_content = response_data['message']['content']
            # Append the assistant's response to history
            conversation_histories[message.author.id].append({"role": "assistant", "content": response_content})
            await message.channel.send(response_content)
        else:
            await message.channel.send("Error processing request")

    if message.content.startswith("!search "):
        query = message.content[len("!search "):]
        query_vec = vectorize(query)
        _, indices = db.search(np.array([query_vec]), k=5)
        results = []
        for idx in indices[0]:
            if idx != -1:
                msg_meta = message_storage[idx]
                link = f"https://discord.com/channels/{msg_meta['guild_id']}/{msg_meta['channel_id']}/{msg_meta['message_id']}"
                results.append(f" Message: {msg_meta['content']}  [Link]({link})")
            else:
                results.append("Message not found")
        response = get_response(query, results)
        await message.channel.send(response)
    else:
        add_vector(message)

client.run(TOKEN)