import os
import sys

import pandas as pd
import ray

import knowledge_engine
from examples.example_llm_inference import ExampleLLMInference
from knowledge_engine.llms.config import LLMMode
from knowledge_engine.llms.vllm.vllm import VLLMFactory
from knowledge_engine.llms.vllm.vllm_engine_wrapper import VLLMTaskType

from knowledge_engine.transforms.extraction.extractor import EnhancedPDFTextExtractor

def main():
    paths = "../pdfs"
    ctx = knowledge_engine.init(exec_mode=knowledge_engine.ExecMode.RAY)

    llm_factory = VLLMFactory(
            model_name="facebook/opt-125m",  # 替换为你的模型路径
            engine_kwargs={
                "tensor_parallel_size": 1,  # 根据GPU数量调整
                "max_num_seqs": 128,
                "max_model_len": 2048,
            },
            max_output_len=2048,
            default_mode=LLMMode.BATCH,
            vllm_task_type=VLLMTaskType.GENERATE,
            default_llm_kwargs={
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 100,
            }
    )

    ds = (
        ctx.read.binary(paths, binary_format="pdf")
        .extract(EnhancedPDFTextExtractor(context=True, type="pymupdf"))
        .llm_inference(ExampleLLMInference(llm_factory=llm_factory))
        .explode()
    )

    ds.write.files("output_docs")

    output = ds.take_all()
    for doc in output:
        print(doc)

if __name__ == "__main__":
    vllm_path = "/opt/anaconda3/envs/python310ms/bin/vllm"
    sys.path.append(os.path.dirname(vllm_path))
    os.environ['VLLM_CPU_KVCACHE_SPACE'] = '10'
    os.environ['VLLM_CPU_OMP_THREADS_BIND'] = 'auto'
    main()