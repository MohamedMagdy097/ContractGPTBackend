from flask import Flask, request, jsonify
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc
from langchaincoexpert.llms import Clarifai

# import csv
import spacy
from pprint import pprint
from fpdf import FPDF

from dropbox_sign import \
    ApiClient, ApiException, Configuration, apis, models

from langchaincoexpert.memory import VectorStoreRetrieverMemory
from langchaincoexpert.chains import ConversationChain
from langchaincoexpert.prompts import PromptTemplate
from langchaincoexpert.vectorstores import SupabaseVectorStore
from langchaincoexpert.embeddings import ClarifaiEmbeddings
from supabase.client import  create_client
# from dotenv import load_dotenv
# from firestore import db
from firebase_admin import firestore
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Clarifai settings
CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")

# Supabase settings
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

# Set up the Clarifai channel
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

# Clarifai settings
USER_ID = 'ahmedz'
APP_ID = 'FINGU'
MODEL_ID = 'GPT-3_5-turbo'

#Drop Box Config
configuration = Configuration(
    # Configure HTTP basic authorization: api_key
    username="ebffdc31428f6518c896b4e7ffe6faadd7c2b614c4271419c3a3cfcfb7369bac",

    # or, configure Bearer (JWT) authorization: oauth2
    # access_token="YOUR_ACCESS_TOKEN",
)


# Initialize Clarifai embeddings
embeddings = ClarifaiEmbeddings(pat=CLARIFAI_PAT, user_id="openai", app_id="embed", model_id="text-embedding-ada")

# Initialize Supabase vector store
# vectordb = SupabaseVectorStore.from_documents({}, embeddings, client=supabase)

# Initialize Clarifai LLM
llm = Clarifai(pat=CLARIFAI_PAT, user_id='openai', app_id='chat-completion', model_id='GPT-4')


# Handle incoming messages
def handle_message(input_text , user_id):
    memory_key = {user_id}
    response = generate_response_llmchain(input_text, user_id)
    return response

def generate_response_llmchain(prompt, conv_id):
    convid = "a" + str(conv_id)
    # filter = {"user_id": userid}
    vectordb = SupabaseVectorStore.from_documents({}, embeddings, client=supabase,user_id=conv_id) # here we use normal userid "for saving memory"

    retriever = vectordb.as_retriever(search_kwargs=dict(k=10,user_id=convid)) # here we use userid with "a" for retreiving memory
    memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key=convid)
    DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI called ContractGPT. 
   ,The Ai is a Contract Creation assitant designed to make Solid Contracts.
   The AI should reply with the contract only without any instructions or explainations Only the Contract. If the question isn't contract related or doesn't output a contract reply with 1.
   

Relevant pieces of previous conversation:
{user_id}
(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
    formatted_template = DEFAULT_TEMPLATE.format(user_id="{"+convid+"}",input = "{input}")

    PROMPT = PromptTemplate(
        input_variables=[convid, "input"], template=formatted_template
    )

    
    conversation_with_summary = ConversationChain(
        llm=llm,
        prompt=PROMPT,
        memory=memory,
        verbose=True
    )

    ans = conversation_with_summary.predict(input=prompt)
    # response = ans
    return ans

def text_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(190, 10, txt=text, align="L")
    pdf_file_path = "output.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path

@app.route('/delete', methods=['DELETE'])
def deleteChat():
    data = request.get_json()
    conversationId = "a" + data['convid']
    
    # Check if conversationId is provided
    if not conversationId:
        return jsonify({"message": "Invalid conversation ID provided"}, status_code=400)

    # Initialize the SupabaseVectorStore instance
    vectordb = SupabaseVectorStore.from_documents({}, embeddings, client=supabase, user_id="")

    # Delete the chat based on conversationId
    try:
        vectordb.delete(ids=[conversationId])
        return jsonify({"message": "Chat deleted successfully"})
    except Exception as e:
        return jsonify({"message": f"Error deleting chat: {str(e)}"}, status_code=500)
    
@app.route('/drop', methods=['POST'])
def drop():
    try:
        # Initialize Dropbox API client
        with ApiClient(configuration) as api_client:
            signature_request_api = apis.SignatureRequestApi(api_client)
          # Parse JSON data from the request body
            request_data = request.get_json()
            text_data = request_data.get("text_data", "Default text for PDF")
            pdf_file_path = text_to_pdf(text_data)

            # Extract signer email addresses from the request data
            signer_1_email = request_data.get("signer_1_email", "jack@example.com")
            signer_2_email = request_data.get("signer_2_email", "jill@example.com")

            # De
            # Define signers and other options
            signer_1 = models.SubSignatureRequestSigner(
                email_address=signer_1_email,
                name=signer_1_email,
                order=0,
            )

            signer_2 = models.SubSignatureRequestSigner(
                email_address=signer_2_email,
                name=signer_2_email,
                order=1,
            )

            signing_options = models.SubSigningOptions(
                draw=True,
                type=True,
                upload=True,
                phone=True,
                default_type="draw",
            )

            field_options = models.SubFieldOptions(
                date_format="DD - MM - YYYY",
            )

            data = models.SignatureRequestSendRequest(
                title="NDA with Acme Co.",
                subject="The NDA we talked about",
                message="Please sign this NDA and then we can discuss more. Let me know if you have any questions.",
                signers=[signer_1, signer_2],
                cc_email_addresses=[
                    "lawyer1@dropboxsign.com",
                    "lawyer2@dropboxsign.com",
                ],
                files=[open(pdf_file_path, "rb")],  # Use the generated PDF
                metadata={
                    "custom_id": 1234,
                    "custom_text": "NDA #9",
                },
                signing_options=signing_options,
                field_options=field_options,
                test_mode=True,
            )

            # Send a signature request
            response = signature_request_api.signature_request_send(data)
            return jsonify({'response': response})

    except ApiException as e:
        print("Exception when calling Dropbox Sign API: %s\n" % e)
        return jsonify({'error': str(e)}, status_code=500)



@app.route('/chat', methods=['POST'])
def api():
    data = request.get_json()
    # input_message is the actual data, the data mime type is specified in type
    input_message = data['prompt']

 
    # ai_id is the id of the ai example GPT4 or GPT3.5 or LLAMA etc 
    conv_id= data["conversationId"]

   
    
    response = handle_message(input_message, conv_id)


    return jsonify({'response': response})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 4000), debug=False)
