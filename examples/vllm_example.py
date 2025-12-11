import asyncio
import sys
import os

import torch

from knowledge_engine.llms.config import LLMMode
from knowledge_engine.llms.prompts import RenderedMessage, RenderedPrompt
from knowledge_engine.llms.vllm.vllm import VLLM
from knowledge_engine.llms.vllm.vllm_engine_wrapper import VLLMTaskType

vllm_path = "/opt/anaconda3/envs/python310ms/bin/vllm"
sys.path.append(os.path.dirname(vllm_path))
os.environ['VLLM_CPU_KVCACHE_SPACE'] = '10'
os.environ['VLLM_CPU_OMP_THREADS_BIND'] = 'auto'

from vllm import LLM, SamplingParams

# def main():
#     # Create a sampling params object.
#     use_mps = torch.backends.mps.is_available()
#     device_type = "mps" if use_mps else "cpu"
#     print(f"Using device: {device_type}")
#     # Initialize the LLM with a small model
#     llm = LLM(model="facebook/opt-125m",
#               download_dir="./models",
#               enforce_eager=True,
#               tensor_parallel_size=1,
#               trust_remote_code=True,
#               dtype="bfloat16")
#     # Set sampling parameters
#     sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)
#     # Generate text
#     prompt = "Write a short poem about artificial intelligence."
#     outputs = llm.generate([prompt], sampling_params)
#     # Print the result
#     for output in outputs:
#         print(output.outputs[0].text)

def main():
    vllm_model = VLLM(
        model_name="facebook/opt-125m",  # 替换为你的模型路径
        engine_kwargs={
            "tensor_parallel_size": 1,  # 根据GPU数量调整
            "max_num_seqs": 128,
            "max_model_len": 2048,
        },
        max_output_len=2048,
        default_mode=LLMMode.ASYNC,
        vllm_task_type=VLLMTaskType.GENERATE,
        default_llm_kwargs={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 100,
        }
    )

    messages = [RenderedMessage(
        role="user",
        content="Write a short poem about artificial intelligence.",
        images=None  # 如果需要图像输入,可以传入图像列表
    ), RenderedMessage(
        role="user",
        content="Tell me a joke",
        images=None  # 如果需要图像输入,可以传入图像列表
    ), RenderedMessage(
        role="user",
        content="Hello",
        images=None  # 如果需要图像输入,可以传入图像列表
    )]

    prompts = [RenderedPrompt(messages=[messages[0]]), RenderedPrompt(messages=[messages[1]]),
               RenderedPrompt(messages=[messages[2]])]
    result = vllm_model.generate_batch(prompts=prompts)
    for res in result:
        print("=====================")
        print(res)





if __name__ == "__main__":
    main()