

# Function to summarize text
def summarize_text(text):
    summaries = summarizer(text, max_length=512, min_length=30, do_sample=False)
    return summaries[0]['summary_text']

