import pandas as pd
import json
import time
import streamlit as st
import google.generativeai as genai

# STEP 1: Setup Gemini API
genai.configure(api_key="AIzaSyD0zckQkzvL_3c2AaALQM1bM_ny2PbnjD4")
gemini = genai.GenerativeModel("gemini-1.5-flash")

# STEP 2: Load and normalize metadata
@st.cache_data
def process_metadata(file_path, mapping_file):
    df = pd.read_csv(file_path)
    with open(mapping_file, 'r') as f:
        column_map = json.load(f)

    df['column_name'] = df['column_name'].apply(lambda x: column_map.get(x, x))

    metadata = {}
    for _, row in df.iterrows():
        schema = row['schema_name']
        table = row['table_name']
        col_data = {"column_name": row['column_name'], "data_type": row['data_type']}
        metadata.setdefault(schema, {}).setdefault(table, []).append(col_data)

    return metadata

# STEP 3: Chunk metadata for retrieval
def chunk_metadata(metadata_dict, max_chars=1000):
    chunks = []
    for schema, tables in metadata_dict.items():
        for table, columns in tables.items():
            chunk_text = f"Schema: {schema}\nTable: {table}\n"
            for col in columns:
                chunk_text += f"- {col['column_name']} ({col['data_type']})\n"
            if len(chunk_text) <= max_chars:
                chunks.append({"schema": schema, "table": table, "text": chunk_text})
    return chunks

# STEP 4: Query Gemini with retry and token limits
def get_answer_from_gemini(query, chunks):
    prompt = (
        "You are a metadata assistant. Below is some technical metadata. "
        "Answer the user's question using the most relevant part.\n\n"
    )
    context = chunks[0]['text'][:1000]
    full_prompt = prompt + "Metadata:\n" + context + "\n\nUser Question: " + query

    for attempt in range(3):
        try:
            response = gemini.generate_content(full_prompt)
            return response.text
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                st.warning("Quota exceeded. Retrying in 60 seconds...")
                time.sleep(60)
            else:
                return f"Error: {e}"
    return "Unable to process the request due to quota limits. Try again later."

# STEP 5: Build UI
def main():
    st.title("Metadata Query System")

    query = st.text_input("Ask your question:")

    if query:
        metadata = process_metadata("sample_metadata.csv", "column_mapping.json")
        chunks = chunk_metadata(metadata)

        with st.spinner("Getting answer..."):
            answer = get_answer_from_gemini(query, chunks)
        st.subheader("Answer:")
        st.write(answer)

# Start app
if __name__ == "__main__":
    main()
