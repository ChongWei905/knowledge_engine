import asyncio
import logging
import os
import threading
from typing import Optional, Callable

from transformers import AutoProcessor

from build.lib.knowledge_engine.transforms.base.base_map_transform import BaseMapTransform
from knowledge_engine.data.document import Document
from knowledge_engine.llms.config import LLMMode
from knowledge_engine.llms.llms import LLM, LLMFactory
from knowledge_engine.llms.prompts import PromptProcessor, RenderedPrompt
from knowledge_engine.transforms import Node
from knowledge_engine.utils.thread_local import ThreadLocal, ADD_METADATA_TO_OUTPUT, ThreadLocalAccess

logger = logging.getLogger(__name__)

def _run_new_thread(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()

def run_coros_threadsafe(coros):
    new_loop = asyncio.new_event_loop()
    t = threading.Thread(target=_run_new_thread, args=(new_loop,), daemon=True)
    t.start()

    metadata = []

    async def _gather_coros(coros):
        # Exfiltrate metadata documents from inner thread
        with ThreadLocal(ADD_METADATA_TO_OUTPUT, metadata):
            tasks = [new_loop.create_task(c) for c in coros]
            return await asyncio.gather(*tasks)

    fut = asyncio.run_coroutine_threadsafe(_gather_coros(coros), loop=new_loop)
    results = fut.result()
    new_loop.call_soon_threadsafe(new_loop.stop)
    t.join()
    new_loop.close()
    tls = ThreadLocalAccess(ADD_METADATA_TO_OUTPUT)
    if tls.present():
        tls.get().extend(metadata)
    return results


def _infer_prompts(
    prompts: list[RenderedPrompt],
    llm: LLM,
    llm_mode: LLMMode,
) -> list[str]:
    if llm_mode == LLMMode.SYNC:
        res = []
        for p in prompts:
            if len(p.messages) == 0:
                res.append("")
                continue
            try:
                s = llm.generate(prompt=p)
                res.append(s)
                if all_prompt_dir := os.environ.get("LLM_DEBUG_DIR"):
                    from datetime import datetime
                    from pathlib import Path

                    now = datetime.now().isoformat()
                    path = Path(all_prompt_dir) / f"{now}.txt"
                    logger.info(f"Saving prompt and result to {path}")
                    with open(path, "w") as f:
                        f.write(p.to_human_readable())
                        f.write("\n\n--------------------------------------------\n\n")
                        f.write(s)
            except Exception:
                bad_prompt_path = os.environ.get("BAD_PROMPT_PATH", "/tmp/bad_prompt.txt")
                with open(bad_prompt_path, "w") as f:
                    f.write(p.to_human_readable())
                    logger.error(f"Error generating prompt. Wrote failing prompt to $BAD_PROMPT_PATH:{bad_prompt_path}")
                raise
        return res
    elif llm_mode == LLMMode.ASYNC:
        nonempty = [(i, p) for i, p in enumerate(prompts) if len(p.messages) > 0]
        res = [""] * len(prompts)
        coroutines = [llm.generate_async(prompt=p, llm_kwargs={}) for _, p in nonempty]
        responses = run_coros_threadsafe(coroutines)

        for (i, _), rs in zip(nonempty, responses):
            res[i] = rs
        return res
    elif llm_mode == LLMMode.BATCH:
        return llm.generate_batch(prompts=prompts)
    else:
        raise NotImplementedError("Unknown LLM Mode")


class LLMMap(BaseMapTransform):
    """The LLMMap transform renders each Document in a docset into
    a prompt for an LLM, calls the LLM, and attaches the output to
    the document.

    Args:

        child: Child node in the sycamore execution graph
        prompt: The SycamorePrompt to use to render each document.
            Must implement the ``render_document`` method.
        output_field: The name of the field in doc.properties in which
            to store the llm output
        llm: The llm to use for inference.
        llm_mode: How to call the llm - sync/async/batch. All LLMs do not
            necessarily implement all options.
        iteration_var: Name of the document property to increment with every
            invalid response. Default is None, which means no re-try.
        validate: Function to determine whether an LLM response is valid.
            Default is 'everything is valid'
        max_tries: Hard limit on the number of LLM calls per document. Default
            is 5

    Example:
         .. code-block:: python

            prompt = EntityExtractorZeroShotGuidancePrompt.set(entity="title")

            docset.llm_map(
                prompt=prompt,
                output_field="title",
                llm=OpenAI(OpenAIModels.GPT_4O_MINI)
            )
    """

    def __init__(
        self,
        child: Optional[Node],
        prompt: PromptProcessor,
        output_field: str,
        llm_factory: LLMFactory,
        llm_mode: Optional[LLMMode] = None,
        iteration_var: Optional[str] = None,
        validate: Callable[[Document], bool] = lambda d: True,
        max_tries: int = 5,
        filter: Callable[[Document], bool] = lambda d: True,
        **kwargs,
    ):
        self._prompt = prompt
        self._validate_prompt()
        self._output_field = output_field
        self._llm_factory = llm_factory
        self._llm_mode = llm_mode if llm_mode is not None else llm_factory.get_default_mode()
        self._iteration_var = iteration_var
        self._validate = validate
        self._max_tries = max_tries
        self._filter = filter
        c = LLMMap.llm_map_class(self._prompt, self._output_field, self._llm_mode, self._iteration_var,
                                 self._validate, self._max_tries, self._filter)
        super().__init__(child, f=c, constructor_args=[self._llm_factory], **kwargs)

    @staticmethod
    def llm_map_class(
        prompt: PromptProcessor,
        output_field: str,
        llm_mode: Optional[LLMMode],
        iteration_var: Optional[str],
        validate: Callable[[Document], bool],
        max_tries: int,
        filter: Callable[[Document], bool] = lambda d: True,
    ) -> type:
        class LLMMapClass:
            def __init__(self, llm_factory: LLMFactory):
                self._model_name = llm_factory.get_model_name()
                self._llm = llm_factory.create()

            def __call__(self, documents: list[Document]) -> list[Document]:
                try:
                    auto_processor = AutoProcessor.from_pretrained(self._model_name)
                except Exception as e:
                    auto_processor = None
                    logger.warning(f"Failed to load processor for model: {self._model_name}. Error: {e}")

                if iteration_var is not None:
                    for d in documents:
                        d.properties[iteration_var] = 0

                skips = [not filter(d) for d in documents]
                tries = 0
                while not all(skips) and tries < max_tries:
                    tries += 1
                    rendered_and_index = [
                        (prompt.render_document(d, processor=auto_processor), i) for sk, d, i in zip(skips, documents, range(len(skips))) if
                        not sk
                    ]
                    rendered = []
                    for r, i in rendered_and_index:
                        if len(r.messages) == 0:
                            skips[i] = True
                        else:
                            rendered.append(r)
                    if len(rendered) == 0:
                        break
                    results = _infer_prompts(rendered, self._llm, llm_mode)
                    ri = 0
                    for i in range(len(documents)):
                        if skips[i]:
                            continue
                        documents[i].properties[output_field] = results[ri]
                        skips[i] = validate(documents[i])
                        ri += 1
                        if iteration_var is not None and not skips[i]:
                            documents[i].properties[iteration_var] += 1
                    if iteration_var is None:
                        break

                return documents


        return LLMMapClass

    def _validate_prompt(self):
        doc = Document()
        try:
            _ = self._prompt.render_document(doc)
        except NotImplementedError as e:
            raise e
        except Exception:
            pass


