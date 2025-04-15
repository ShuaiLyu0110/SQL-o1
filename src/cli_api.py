from llamafactory.chat import ChatModel

# def llm(query):
#     chat_model = ChatModel()
#     messages = []
#     messages.append({"role": "user", "content": query})

#     response = ""
#     for new_text in chat_model.stream_chat(messages):
#         # print(new_text, end="", flush=True)
#         response += new_text
#     return response


# if __name__ == "__main__":
#     query = 'Please type the NER span over options. Output format is "span: type". \nOptions: geographical social political, organization, person, location, facility, vehicle, weapon \nText: At the very least , Lemieux will attract ticket buyers and television viewers to a wintry sport that usually spins its wheels along the road to show - business success . \nOutput: ticket buyers: '
#     res = llm(query)
#     print(res)
#     # 把llm包装成query到res的接口
    
    
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import warnings
from urllib.parse import quote
warnings.filterwarnings("ignore", message="cmap value too big/small")

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
chat_model = ChatModel()
class LLMRequest(BaseModel):
    query: str

@app.post(f"/{os.environ.get('MODEL_NAME', 'llm')}")
async def llm(request:LLMRequest):
    messages = []
    messages.append({"role": "user", "content": request.query})

    response = ""
    for new_text in chat_model.stream_chat(messages):
        # print(new_text, end="", flush=True)
        response += new_text
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=int(os.environ.get("API_PORT", "8000")))
