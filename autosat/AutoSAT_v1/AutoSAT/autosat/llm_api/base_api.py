import os
import time

import openai


class BaseCallAPI():
    def __init__(self, api_base, api_key, model_name):
        self.api_base = api_base
        self.api_key = api_key
        openai.api_base = self.api_base
        openai.api_key = self.api_key
        self.model_name = model_name
        self._cum_prompt_tokens = 0
        self._cum_completion_tokens = 0
        self._cum_total_tokens = 0
        self._cum_calls = 0
        self._retry_sleep_seconds = max(1, int(os.getenv("AUTOSAT_API_RETRY_SECONDS", "10")))
        self._max_retries = int(os.getenv("AUTOSAT_API_MAX_RETRIES", "0"))

    def _log_token_usage(self, usage):
        if usage is None:
            return

        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)

        try:
            prompt_tokens = int(prompt_tokens or 0)
            completion_tokens = int(completion_tokens or 0)
            total_tokens = int(total_tokens or 0)
        except Exception:
            return

        self._cum_calls += 1
        self._cum_prompt_tokens += prompt_tokens
        self._cum_completion_tokens += completion_tokens
        self._cum_total_tokens += total_tokens

        print(
            f"[TokenUsage] model={self.model_name} "
            f"call_prompt={prompt_tokens} call_completion={completion_tokens} call_total={total_tokens} "
            f"cum_calls={self._cum_calls} cum_prompt={self._cum_prompt_tokens} "
            f"cum_completion={self._cum_completion_tokens} cum_total={self._cum_total_tokens}"
        , flush=True)

    def load_prompt(self, file_dir):
        with open(file_dir, 'r') as file:
            prompt = file.read()
        return prompt

    def _call_with_retries(self, request_fn, description="API call"):
        attempt = 0
        while True:
            try:
                return request_fn()
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                attempt += 1
                if self._max_retries > 0 and attempt > self._max_retries:
                    print(
                        f"[{description}] failed after {attempt - 1} retries: {exc}",
                        flush=True,
                    )
                    raise

                print(
                    f"[{description}] error on attempt {attempt}: {exc}. "
                    f"Retrying in {self._retry_sleep_seconds} seconds...",
                    flush=True,
                )
                time.sleep(self._retry_sleep_seconds)

    def call_api(self, prompt, temperature):
        pass


class GPTCallAPI(BaseCallAPI):
    def __init__(self, api_base, api_key, model_name):
        super(GPTCallAPI, self).__init__(api_base, api_key, model_name)

    def call_api(self, prompt_file, temperature=0.2):
        with open(prompt_file, 'r') as file:
            prompt = file.read()
        response = self._call_with_retries(
            lambda: openai.ChatCompletion.create(
                model=self.model_name,
                # model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a chatbot", },
                    {"role": "user", "content": prompt}],
                temperature=temperature,
                stream=False
            ),
            description="GPTCallAPI.call_api",
        )
        self._log_token_usage(response.get("usage", None))
        return response["choices"][0]["message"]["content"]


class LocalCallAPI(BaseCallAPI):
    def __init__(self, api_base, api_key, model_name):
        super(LocalCallAPI, self).__init__(api_base, api_key, model_name)

    def call_api(self, prompt_file,
                 temperature=0.2):
        stop_tokens = ["<|im_end|>"]
        system_prompt = "You are a chatbot"

        with open(prompt_file, 'r') as file:
            prompt = file.read()

        response = self._call_with_retries(
            lambda: openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}],
                    stop=stop_tokens),
            description="LocalCallAPI.call_api",
        )
        self._log_token_usage(response.get("usage", None))
        return response["choices"][0]["message"]["content"]


def get_llm_api(args):
    model_name = str(getattr(args, 'llm_model', '') or '')
    api_base = str(getattr(args, 'api_base', '') or '').strip()
    api_key = str(getattr(args, 'api_key', '') or '').strip()

    if (not api_base) and model_name.startswith('openai/'):
        api_base = os.getenv('AUTOSAT_API_BASE') or os.getenv('DEEPINFRA_API_BASE') or ''

    if api_base:
        return GPTCallAPI(api_base=api_base,
                          api_key=api_key,
                          model_name=model_name)

    if model_name in ('gpt-4-1106-preview', 'gpt-3.5-turbo'):
        return GPTCallAPI(api_base=api_base,
                          api_key=api_key,
                          model_name=model_name)
    if model_name == 'Qwen':
        return LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                            api_key="sk-",
                            model_name="modelscope/qwen/Qwen-72B-Chat")
    if model_name == 'llama':
        return LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                            api_key="sk-",
                            model_name="modelscope/modelscope/Llama-2-70b-chat-ms")
    if model_name == 'deepseek':
        return LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                            api_key="sk-",
                            model_name="modelscope/deepseek-ai/deepseek-coder-33b-instruct")

    raise NotImplementedError(
        "Unsupported llm_model without external API endpoint. "
        "Set api_base/api_key for OpenAI-compatible models or use one of: Qwen, llama, deepseek."
    )


if __name__ == '__main__':
    llm_api = LocalCallAPI(api_base="http://172.26.1.16:31251/v1",
                          api_key="sk-",
                          model_name="modelscope/modelscope/Llama-2-70b-chat-ms")
    answer = llm_api.call_api(prompt_file='../template/EasySAT/bump_var_function/original_prompt.txt')
    print(answer)
