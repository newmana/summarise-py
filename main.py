from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import SQLiteVSS
import os
import sqlite3

# load the document and split it into chunks
loader = TextLoader("modules/state_of_the_union.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

path = 'db/summaries.db'
scriptdir = os.path.dirname(__file__)
db_path = os.path.join(scriptdir, path)
os.makedirs(os.path.dirname(db_path), exist_ok=True)

connection = SQLiteVSS.create_connection(db_file=db_path)

# load it in sqlite-vss in a table named state_union.
# the db_file parameter is the name of the file you want
# as your sqlite database.
db = SQLiteVSS.from_texts(
    texts=texts,
    embedding=embedding_function,
    table="state_union",
    db_file=db_path,
)

# query it
query = "What did the president say about Ketanji Brown Jackson"
data = db.similarity_search(query)

# print results
print(data[0].page_content)

# query it
query = "Who does he bless?"
data = db.similarity_search(query)

# print results
print(data[0].page_content)