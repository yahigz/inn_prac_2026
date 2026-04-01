import openai
import os


TOKEN_USAGE_SUMMARY = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "calls": 0,
}


def _record_usage(response, model_name):
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)

    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return

    TOKEN_USAGE_SUMMARY["prompt_tokens"] += prompt_tokens
    TOKEN_USAGE_SUMMARY["completion_tokens"] += completion_tokens
    TOKEN_USAGE_SUMMARY["total_tokens"] += total_tokens
    TOKEN_USAGE_SUMMARY["calls"] += 1

    print(
        "[TokenUsage] "
        f"model={model_name} "
        f"call_prompt={prompt_tokens} "
        f"call_completion={completion_tokens} "
        f"call_total={total_tokens} "
        f"cum_calls={TOKEN_USAGE_SUMMARY['calls']} "
        f"cum_prompt={TOKEN_USAGE_SUMMARY['prompt_tokens']} "
        f"cum_completion={TOKEN_USAGE_SUMMARY['completion_tokens']} "
        f"cum_total={TOKEN_USAGE_SUMMARY['total_tokens']}"
    )


class BaseCallAPI():
    def __init__(self, api_base, api_key, model_name):
        self.api_base = api_base
        self.api_key = api_key
        openai.api_base = self.api_base
        openai.api_key = self.api_key
        self.model_name = model_name

    def load_prompt(self, file_dir):
        with open(file_dir, 'r') as file:
            prompt = file.read()
        return prompt

    def call_api(self, prompt, temperature):
        pass

class GPTCallAPI(BaseCallAPI):
    def __init__(self, api_base, api_key, model_name, stream):
        super(GPTCallAPI, self).__init__(api_base, api_key, model_name)
        self.stream = stream

    def call_api(self, prompt_file, temperature=0.2):
        with open(prompt_file, 'r') as file:
            prompt = file.read()
        response = openai.ChatCompletion.create(
            model=self.model_name,
            # model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a chatbot", },
                {"role": "user", "content": prompt}],
            temperature=temperature,
            stream=self.stream
        )

        if self.stream:
            result = ""
            try:
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        result += chunk.choices[0].delta.content
                        # print(chunk.choices[0].delta.content, end="")
            except:
                pass
        else:
            _record_usage(response, self.model_name)
            result = response["choices"][0]["message"]["content"]

        return result

    def call_api_prompt(self, prompt, temperature=0.2):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a chatbot", },
                {"role": "user", "content": prompt}],
            temperature=temperature,
            stream=self.stream
        )

        if self.stream:
            result = ""
            try:
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        result += chunk.choices[0].delta.content
                        # print(chunk.choices[0].delta.content, end="")
            except:
                pass
        else:
            _record_usage(response, self.model_name)
            result = response["choices"][0]["message"]["content"]

        return result

class LocalCallAPI(BaseCallAPI):
    def __init__(self, api_base, api_key, model_name):
        super(LocalCallAPI, self).__init__(api_base, api_key, model_name)

    def call_api(self, prompt_file,
                 temperature=0.2):
        stop_tokens = ["<|im_end|>"]
        system_prompt = "You are a chatbot"

        with open(prompt_file, 'r') as file:
            prompt = file.read()

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}],
                stop=stop_tokens)
        _record_usage(response, self.model_name)
        return response["choices"][0]["message"]["content"]

    def call_api_prompt(self, prompt,
                 temperature=0.2):
        stop_tokens = ["<|im_end|>"]
        system_prompt = "You are a chatbot"

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}],
                stop=stop_tokens,
                temperature=temperature,)

        _record_usage(response, self.model_name)
        return response["choices"][0]["message"]["content"]


def get_llm_api(args, stream=False):
    model_name = str(getattr(args, 'llm_model', '') or '')
    api_base = str(getattr(args, 'api_base', '') or '').strip()
    api_key = str(getattr(args, 'api_key', '') or '').strip()

    if (not api_base) and model_name.startswith('openai/'):
        api_base = (
            os.getenv('AUTOSAT_API_BASE')
            or os.getenv('DEEPINFRA_API_BASE')
            or 'https://api.deepinfra.com/v1/openai'
        )

    # Prefer external OpenAI-compatible API when endpoint is provided.
    if api_base:
        return GPTCallAPI(api_base=api_base,
                          api_key=api_key,
                          model_name=model_name,
                          stream=stream)

    if model_name.startswith("gpt"):
        return GPTCallAPI(api_base=api_base,
                          api_key=api_key,
                          model_name=model_name,
                          stream=stream)
    elif model_name == 'Qwen':
        return LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                            api_key="sk-",
                            model_name="modelscope/qwen/Qwen-72B-Chat")
    elif model_name == 'llama':
        return LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                            api_key="sk-",
                            model_name="modelscope/modelscope/Llama-2-70b-chat-ms")
    elif model_name == 'deepseek':
        return LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                            api_key="sk-",
                            model_name="modelscope/deepseek-ai/deepseek-coder-33b-instruct")

    raise NotImplementedError(
        "Unsupported llm_model without external API endpoint. "
        "Set api_base/api_key for OpenAI-compatible models or use one of: Qwen, llama, deepseek."
    )

# *----------- Fast llm call --------------------*
def fastllm(prompt , args):
    llm_api = get_llm_api(args, stream=False)
    answer = llm_api.call_api_prompt(prompt=prompt, temperature=args.temperature)
    return answer



if __name__ == '__main__':
    llm_api = LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                          api_key="sk-",
                          model_name="modelscope/modelscope/Llama-2-70b-chat-ms")
    answer = llm_api.call_api(prompt_file='../template/EasySAT/bump_var_function/original_prompt.txt')
    print(answer)