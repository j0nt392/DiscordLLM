import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
import os 
from openai import OpenAI
import numpy as np
import discord
from dotenv import load_dotenv
import pickle
import logging
import ollama

load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class VDB:
    def __init__(self, dimension=1024, index_file='faiss_index.index', storage_file='message_storage.pkl'):
        self.dimension = dimension
        self.index_file = index_file
        self.storage_file = storage_file

        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.openai_client = OpenAI(api_key=self.openai_key)

        # Load tokenizer and model with Hugging Face token for authentication
        self.tokenizer = AutoTokenizer.from_pretrained(
            'intfloat/multilingual-e5-large',
            token=self.huggingface_token
        )
        self.model = AutoModel.from_pretrained(
            'intfloat/multilingual-e5-large',
            token=self.huggingface_token
        )

        # Load or initialize FAISS index and message storage
        if os.path.exists(self.index_file) and os.path.exists(self.storage_file):
            self.db = self.load_faiss_index()
            self.message_storage = self.load_message_storage()
            logger.info('Loaded FAISS index and message storage from disk.')
        else:
            self.db = faiss.IndexFlatL2(self.dimension)
            self.message_storage = {}
            logger.info('Initialized new FAISS index and message storage.')

    def save_faiss_index(self):
        faiss.write_index(self.db, self.index_file)
        logger.info('FAISS index saved to disk.')

    def load_faiss_index(self):
        return faiss.read_index(self.index_file)

    def save_message_storage(self):
        with open(self.storage_file, 'wb') as f:
            pickle.dump(self.message_storage, f)
        logger.info('Message storage saved to disk.')

    def load_message_storage(self):
        with open(self.storage_file, 'rb') as f:
            return pickle.load(f)

    def vectorize(self, text, is_query=True):
        prefix = "query: " if is_query else "passage: "
        input_text = prefix + text
        inputs = self.tokenizer(input_text, return_tensors='pt', max_length=512, padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.squeeze().cpu().numpy()

    def add_vector(self, text):
        index = len(self.message_storage)
        vector = self.vectorize(text.content, is_query=False)
        self.db.add(np.array([vector]))
        self.message_storage[index] = {
            'content': text.content,
            'channel_id': text.channel.id,
            'message_id': text.id,
            'guild_id': text.guild.id
        }

    async def index_channel_history(self, channel):
        try:
            async for message in channel.history(limit=1000):
                if not message.content.startswith('!search') and not message.author.bot:
                    self.add_vector(message)
            logger.info(f'Indexed messages from {channel.name}')
        except discord.Forbidden:
            logger.warning(f'Cannot access messages from {channel.name}')
        except Exception as e:
            logger.error(f'Error indexing messages from {channel.name}: {e}')

    def get_response(self, query, results):
        prompt = f"This is the query from the user: {query}\nAnd these are the search results:\n"
        for i, result in enumerate(results):
            prompt += f"Result {i+1}: {result}\n"
        prompt += "Please filter out any irrelevant or redundant results and provide a coherent response to the user's query. Always include the link to the original message."

        try:
            # response = ollama.chat(model='llama3', messages=[
            #     {
            #     'role': 'user',
            #     'content': prompt,
            #     },
            # ])
            # return response['message']['content']
            response = self.openai_client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7
            )
            response_text = response.choices[0].text.strip()
            return response_text
        except Exception as e:
            logger.error(f"Error during API call: {e}")
            return "An error occurred while generating the response."