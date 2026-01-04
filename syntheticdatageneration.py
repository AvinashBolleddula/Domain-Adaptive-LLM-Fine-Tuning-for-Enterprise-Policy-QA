from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore 

import json
from typing import List 
from pydantic import BaseModel
from litellm import completion
from generated_prompt import prompt_template

class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def llm_call(data: str, num_records: int = 5) -> dict:
    stream = completion(
        model="ollama_chat/qwen2.5:7b",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream=True,
        options={"num_predict": 512, "temperature": 0.2},
        format=Response.model_json_schema(),
    )
    data = ""
    for x in stream: 
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None: 
            print(Fore.LIGHTBLUE_EX+ delta + Fore.RESET, end="") 
            data += delta 
    return json.loads(data)

if __name__ == "__main__": 
    converter = DocumentConverter()
    doc = converter.convert("enterprise_leave_benefits_policy_excerpt.pdf").document
    chunker = HybridChunker(max_chunk_size=2000,chunk_overlap=200)
    chunks = chunker.chunk(dl_doc=doc)

    dataset = {}
    for i, chunk in enumerate(chunks): 
            print(Fore.YELLOW + f"Raw Text:\n{chunk.text[:300]}…" + Fore.RESET)
            MAX_CONTEXT_CHARS = 3000
            enriched_text = chunker.contextualize(chunk=chunk)
            enriched_text = enriched_text[:MAX_CONTEXT_CHARS]
            if len(enriched_text) < 300:
                continue
            print(Fore.LIGHTMAGENTA_EX + f"Contextualized Text:\n{enriched_text[:300]}…" + Fore.RESET)

            data = llm_call(
                enriched_text
            )
            dataset[i] = {"generated":data["generated"], "context":enriched_text}
    
    with open('enterprise_leave_benefits_policy_excerpt.json','w') as f: 
        json.dump(dataset, f) 