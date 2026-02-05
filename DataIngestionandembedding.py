from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader
import os

dir_loader = DirectoryLoader(r'C:\\Users\\BabuR\\Documents\\Notes', glob='**/*.pdf', show_progress=True, loader_cls=PyMuPDFLoader)
documents = dir_loader.load()
print(documents) 


#emddings and vector store
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid     
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings    

class emddingmanager:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)    
        self.load_model()

    def load_model(self):
        try:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise

    ##Initialize the embedding manager
embedding_manager = emddingmanager()
embedding_manager


##vector store
class VectorStore:
    def __init__(self, collection_name: str = 'pdf_documents',persist_directory: str = 'C:\\Users\\BabuR\\Documents\\PythonLearnings-1\\Vectorestore'):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = None
        self.client = None
        self.intialize_store()

    def intialize_store(self):
        try:
            print("Initializing ChromaDB client...")
            os.makedirs('C:\\Users\\BabuR\\Documents\\PythonLearnings-1\\Vectorestore', exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            print(f"Creating or getting collection: {self.collection_name}")
            self.collection = self.client.get_or_create_collection(name=self.collection_name,metadata={"description": "pdf document embeddings"})
            print(f"Vector store initialized successfully. {self.collection_name}")
            #print(f"existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embedding_manager: np.ndarray):
        try:
            if len(documents) != len(embedding_manager):
                raise ValueError("The number of documents and embeddings must match.")
            
            #prepare data for chromadb
            ids = []
            metadatas = []
            documents_texts = []
            embedding_list = []

            for i,(doc, embedding) in enumerate(zip(documents, embedding_manager)):
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(doc_id)

                #metadata
                metadata = dict(doc.metadata)
                metadata['doc_index'] = i
                metadata['content_length'] = len(doc.page_content)
                metadatas.append(metadata)

                #Document content
                documents_texts.append(doc.page_content)

                #Embeddings
                embedding_list.append(embedding.tolist())
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

        try:
            self.collection.add(ids=ids, metadatas=metadatas, documents=documents_texts, embeddings=embedding_list)
            print(f"Added {len(documents)} documents to vector store successfully.")
        except Exception as e:
            print(f"Error adding documents to collection: {e}")
            raise       


vecctor_store = VectorStore()
vecctor_store

#convert documents to embeddings 
texts = [doc.page_content for doc in documents]


embeddings = embedding_manager.get_embeddings(texts)


#store in vector store
vecctor_store.add_documents(documents, embeddings)


#rag retrieval augmented generation
class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: emddingmanager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        

    def retrieve(self, query: str,top_k: int = 5,score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        try:
            print(f"Retrieving documents for query: '{query}' with top_k={top_k} and score_threshold={score_threshold}")
            query_embedding = self.embedding_manager.get_embeddings([query])[0]
            #Vector store query
            results = self.vector_store.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

            #Process Results
            retrived_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] 
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i,(doc, meta, dist, doc_id) in enumerate(zip(documents, metadatas, distances, ids)):
                    similarity_score = 1 - dist  # Convert distance to similarity score

                    if similarity_score >= score_threshold:
                        retrived_docs.append({
                            'id': doc_id,
                            'content': doc,
                            'metadata': meta,
                            'score': similarity_score,
                            'distance': dist,
                            'rank': i + 1
                        })

                print(f"Retrieved {len(retrived_docs)} documents after applying score threshold.")
            else:
                print("No documents retrieved for the given query.")
            
            return retrived_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
        
ragretriever = RAGRetriever(vector_store=vecctor_store, embedding_manager=embedding_manager)
ragretriever.retrieve("What is the aadhar number?")
