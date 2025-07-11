import time

from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.outputs import RequestOutput


class MentalHealthLLM:
    def __init__(
        self,
        model: str = "dzur658/smollm2-mentalhealth-360m",
        seed: int = 42,
        gpu_memory_utilization: float = 0.3,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 8192,
    ):
        self.sys_prompt = "You are an extremely empathetic and helpful AI assistant named SmolHealth designed to listen to the user and provide insight."
        self.sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        self.model = LLM(
            model=model, seed=seed, gpu_memory_utilization=gpu_memory_utilization
        )

    def revoke(self, prompts: list[str]) -> list[RequestOutput]:
        messages: list[list[ChatCompletionMessageParam]] = [
            [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        outputs = self.model.chat(
            messages=messages,
            sampling_params=self.sampling_params,
        )
        return outputs


if __name__ == "__main__":
    prompts = [
        "I am feeling sad and anxious. Can you help me understand why I might be feeling this way?",
        "I have been struggling with my mental health for a while now. I feel overwhelmed and don't know where to start.",
        "I often feel like I'm not good enough and that I don't deserve to be happy.",
        "I have a lot of negative thoughts about myself and I don't know how to change them.",
        "I feel like I'm stuck in a rut and I don't know how to get out of it.",
    ]
    llm = MentalHealthLLM()

    # Generate responses for each prompt
    print("Generating responses in a loop:")
    start = time.time()
    for prompt in prompts:
        output = llm.revoke([prompt])
        print(output[0].outputs[0].text)

    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

    # Generate responses in batch
    print("Generating responses in batch:")
    start = time.time()
    outputs = llm.revoke(prompts)
    for out in outputs:
        print(out.outputs[0].text)

    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
