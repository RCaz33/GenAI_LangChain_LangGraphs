# Document Loading


* pdf loader { from langchain.document_loaders import PyPDFLoader }
    ->loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
    =>pages = loader.load()
* Youtube loader { from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader }
---->url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
---->save_dir="docs/youtube/"
====> loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
)
====> docs = loader.load()
* load urls { from langchain.document_loaders import WebBaseLoader }
----> loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")

* load fom databases (Notion) { from langchain.document_loaders import NotionDirectoryLoader }
    ->loader = NotionDirectoryLoader("docs/Notion_DB")

# Document Splitting


* choose your splitter : RecusiveCaracterSplitter / NLTKTextSpliter / SpacyTextSplitter / TokenTextSplitter
* split text before putting in vector store { from langchain.text_splitter import ... }
    -> text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)  [.split_documents(pages)]


# Vectorstores and Embeddings

* load several pdf at once { from langchain.document_loaders import PyPDFLoader
}
    ->loaders,docs = [PyPDFLoader("..."),PyPDFLoader("..."),...], []
    =>for loader in loaders:
        docs.extend(loader.load())
* split documents 
    => splits = text_splitter.split_documents(docs)
* persist embed vector store locally { from langchain.vectorstores import Chroma }
    => vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)


# Retrieval

* can explore similarity search with small db of few texts
----> smalldb = Chroma.from_texts(texts, embedding=embedding)
====> smalldb.similarity_search("...", k=2)
* get more diverse info with MMR
====> smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3)
* use llm to extract specific metadata { from langchain.retrievers.self_query.base import SelfQueryRetriever }
----> llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)
====> docs_llm_search = retriever.get_relevant_documents(question)
* compress data with llm (it extract only relevant info from the different contexts) { from langchain.retrievers.document_compressors import LLMChainExtractor }
----> compressor = LLMChainExtractor.from_llm(llm)
* retreive data for context { from langchain.retrievers import ContextualCompressionRetriever
 }
 ----> compression_retriever = ContextualCompressionRetriever( base_compressor=compressor, base_retriever=vectordb.as_retriever() )
 ====> compressed_docs = compression_retriever.get_relevant_documents(question)
* specify  type of extraction (ss / mmr)
----> compression_retriever = ContextualCompressionRetriever( base_compressor=compressor,base_retriever=vectordb.as_retriever(search_type = "mmr") )
* use different type of retreiver (TFIDF / SVM) { from langchain.retrievers import SVMRetriever
from langchain.retrievers import TFIDFRetriever }

# Question Answering

* Loading -> splitting -> storage -> retrieval -> output
* use chain for Q&A { from langchain.chains import RetrievalQA }
----> qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever() )
====> result = qa_chain({"query": question})
* initialize prompt in chain_type_kwargs
----> qa_chain_prompt = """  {context} ... {question} """
====> qa_chain = RetrievalQA.from_chain_type(..., chain_type_kwargs={"prompt": QA_chain_prompt})
* specify chain type (map_reduce, refine, )
----> qa_chain_mr = RetrievalQA.from_chain_type( ..., chain_type= "map_reduce" )
====> result = qa_chain_mr({"query": question})
result["result"]

# Chat

* all previous in One
    -> template = """  {context} ... {question} """
    -> prompt = PromptTemplate(input_variables=["context", "question"],template=template,)
    -> qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt": prompt})
    => result = qa_chain({"query": question})
* set up memory { from langchain.memory import ConversationBufferMemory }
----> memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
* use memory in the chain { from langchain.chains import ConversationalRetrievalChain }
----> qa = ConversationalRetrievalChain.from_llm(llm,
    retriever=vectordb.as_retriever(), memory=memory )
====> result = qa({"question": question})
* Put all together (example in notebook)