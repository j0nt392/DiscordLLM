import faiss
from transformers import AutoTokenizer, AutoModel
import torch 
import os 
from openai import OpenAI
import numpy as np

openai_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_key)  # Singleton OpenAI client
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


# Ollama chatbot 
def get_response(query, results):
    # Construct the prompt string directly using list comprehension within the f-string
    results_str = "\n".join([f"Result {i+1}: {result}" for i, result in enumerate(results)])
    prompt = f"This is the query from the user: {query} \n And these are searchresults: {results_str}. \n The answer to the user query is:"
    print(prompt)
    try:
        # Make the API call to OpenAI
        response = openai_client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7
        )
        # Print the entire response for debugging
        print("API Response:", response)
        response_text = response.choices[0].text.strip()
        print("Extracted Response Text:", response_text)
        return response_text
    except Exception as e:
        print("Error during API call:", e)
        return "An error occurred while generating the response."

    # response = ollama.chat(model='llama3', messages=[
    #     {
    #         'role': 'user',
    #         'content': prompt,
    #     },
    #     ])
    # return response['message']['content']