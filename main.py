# invite link 
#https://discord.com/oauth2/authorize?client_id=1240321223418970286&permissions=274877975552&scope=bot
import discord
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TOKEN = os.getenv("TOKEN")
# Vectorization setup
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def vectorize(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output[0].numpy()

# Faiss and message storage
dimension = 384
db = faiss.IndexFlatL2(dimension)
message_storage = {}  # Dictionary to map indices to messages

def add_vector(text):
    index = len(message_storage)  # Current index for the new message
    vector = vectorize(text.content)
    db.add(np.array([vector]))  # Add the vector to the database
    message_storage[index] = {
        'content': text.content, 
        'channel_id': text.channel.id,
        'message_id': text.id, 
        'guild_id': text.guild.id
    }  # Store the message with its index

# Discord bot setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

async def index_channel_history(channel):
    try:
        async for message in channel.history(limit=1000):
            add_vector(message)
            if not message.author.bot:
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

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!search "):
        query = message.content[len("!search "):]
        query_vec = vectorize(query)
        _, indices = db.search(np.array([query_vec]), k=5)
        results = []
        for idx in indices[0]:
            if idx != -1:
                msg_meta = message_storage[idx]
                link = f"https://discord.com/channels/{msg_meta['guild_id']}/{msg_meta['channel_id']}/{msg_meta['message_id']}"
                results.append(f"Message:\n {msg_meta['content']} \n[Link]({link})")
            else:
                results.append("Message not found")
        await message.channel.send("\n".join(results))
    else:
        add_vector(message)

client.run(TOKEN)