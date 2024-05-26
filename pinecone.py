from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

# Converting text document into a list of document objects
loader = TextLoader("transcript.txt")
text_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)

# Converting into embeddings using Hugging Face embedding model
embeddings = HuggingFaceEmbeddings()

# Following is not required:
# content_embeddings = [embeddings.embed_query(doc.page_content) for doc in documents]

# print("1", len(documents))
# print("2", len(content_embeddings[0]))
# print("3", len(content_embeddings[1]))

# Loading embeddings to Pinecone
# Running the following pushes embeddings to Pinecone
index_name = "transcript"
pinecone = PineconeVectorStore.from_documents(
    documents, embeddings, index_name=index_name
)

# Running a similarity search
pc = PineconeVectorStore(embedding=embeddings)
print(pc.similarity_search("What is RAG?")[:3])