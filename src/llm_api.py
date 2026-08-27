from llamafactory.hparams import get_infer_args
from llamafactory.chat.hf_engine import HuggingfaceEngine
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import os
from typing import Sequence
from llamafactory.chat import ChatModel

app = FastAPI()

chat_model = ChatModel()
device = chat_model.engine.model.device
max_batch_size = 1

def _generation_inputs(engine: HuggingfaceEngine, model_input: str):
    messages = [{"role": "user", "content": model_input}]
    gen_kwargs, prompt_length = HuggingfaceEngine._process_args(
        engine.model,
        engine.tokenizer,
        engine.processor,
        engine.template,
        engine.generating_args,
        messages,
        None,
        None,
        None,
        {},
    )
    return gen_kwargs, prompt_length


@torch.inference_mode()
def score(engine: HuggingfaceEngine, model_input: str, outputs: Sequence[str]):
    """Return raw sequence log-likelihoods log pi(y | x).

    Reward scaling (beta + alpha * log p) is deliberately handled by the
    search client so the API exposes a model quantity rather than a
    paper-specific reward.
    """
    gen_kwargs, prompt_length = _generation_inputs(engine, model_input)
    prompt_ids = gen_kwargs["input_ids"][0].to(device)
    log_likelihoods = []
    for output in outputs:
        output_ids = engine.tokenizer(
            output, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].to(device)
        if output_ids.numel() == 0:
            log_likelihoods.append(0.0)
            continue
        input_ids = torch.cat((prompt_ids, output_ids), dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        logits = engine.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).logits
        token_log_probs = torch.log_softmax(
            logits[0, prompt_length - 1 : prompt_length + output_ids.numel() - 1, :], dim=-1
        )
        sequence_log_prob = token_log_probs.gather(1, output_ids.unsqueeze(1)).sum()
        log_likelihoods.append(float(sequence_log_prob.cpu()))
    return log_likelihoods

def beam(engine: HuggingfaceEngine, model_input: str, beam_size: int = 5, max_new_tokens: int = 1024):
    gen_kwargs, prompt_length = _generation_inputs(engine, model_input)
    generate_output = engine.model.generate(
        **gen_kwargs,
        num_beams=beam_size,
        num_return_sequences=beam_size,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
    )
    
    response_ids = generate_output.sequences[:, prompt_length:]
    response = engine.tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print(response)
    # return (response, 1)


    scores = generate_output.sequences_scores.cpu().tolist()
    return [
        {"text": text, "beam_score": float(sequence_score)}
        for text, sequence_score in zip(response, scores)
    ]

class LLMRequest(BaseModel):
    input: str
    output: Sequence[str]
    beam_size: int = 5
    max_new_tokens: int = 1024


# @app.post(f"/llm")
@app.post(f"/llm")
async def llm(request:LLMRequest):
    if len(request.output) == 0:
        # messages = []
        # messages.append({"role": "user", "content": request.input})
        # response = ""
        # for new_text in chat_model.stream_chat(messages):
        #     response += new_text
        # print(request.instruction + "\n" + request.input)
        # print(request.input)
        response = {
            "candidates": beam(
                chat_model.engine,
                request.input,
                beam_size=max(1, request.beam_size),
                max_new_tokens=max(1, request.max_new_tokens),
            )
        }
    else:
        response = {"log_likelihoods": score(chat_model.engine, request.input, request.output)}
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=int(os.environ.get("API_PORT", "8000")))
