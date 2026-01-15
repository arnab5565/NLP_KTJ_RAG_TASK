
import os
from langchain.chat_models import init_chat_model

model1='gemini-2.5-flash-lite'
model2='gemini-2.0-flash'

os.environ["GOOGLE_API_KEY"] = "AIzaSyA-o5rpXf3cgMmqMmUhZRviiGQmCVOJ7Ak"

'''
model = init_chat_model("google_genai:gemini-2.5-flash-lite")
flag=1

while(flag):
    query=input('Ask The chatbot type 1 for exit: ')
    if(query=='1'):
        flag=0
        continue
    resposne=model.invoke(query)
    print('=============AGent Reply===============\n')
    print(resposne.text)
    print('\n')


'''
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from google import genai
from google.genai import types
import numpy as np

gemini_key="AIzaSyCxNGWv8yD5hhBxo37i4dAo2rnypTJ2SzY"

embedd_gemini_model="gemini-embedding-exp-03-07"
embedd_gemini_model2="text-embedding-004"
#euclidean distance algo

def eucledian_distance(list1,list2):

    sum_of_squares=0
    for i in range(len(list1)):

        sum_of_squares+=(list1[i]-list2[i])**2
    return np.sqrt(sum_of_squares)


#pdf parsing

file_path = "./Options_futures_derivatives.pdf"
file_path2="./budget_speech.pdf"
file_path3="./winwin.pdf"
loader = PyPDFLoader(file_path2)

docs = loader.load()

#print(len(docs))
#print(type(docs))
#print(docs[0])
#print(f'{docs[1].page_content}\n')
#print('\n')
'''
for i in range(len(docs)):
    print(f'{docs[i].page_content}\n')
'''



split_data=RecursiveCharacterTextSplitter(
    chunk_size=2000,chunk_overlap=200,add_start_index=True
)

net_splitted_data=split_data.split_documents(docs)
#print(f'{net_splitted_data[2].page_content}\nend of 1st page\n')
#print(f'{net_splitted_data[1].page_content}\nend of 2nd\n')

#print(f'{net_splitted_data[2].metadata}')
#print(len(net_splitted_data))

query_string=[]
for s in net_splitted_data:
    query_string.append(s.page_content)



#embedding and answer generation

client = genai.Client(api_key='AIzaSyA-o5rpXf3cgMmqMmUhZRviiGQmCVOJ7Ak')

result = client.models.embed_content(
        model=embedd_gemini_model2,
        contents=query_string,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)
response = client.models.generate_content(
    model="gemini-2.5-flash-lite", contents=query_string[1]
)

#print(query_string[0])
#print(result.embeddings[0].values[:5])



print(response.text)

