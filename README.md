*This activity has been created as part of the 42 curriculum by nonottar.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about
the source code and documentation of the **vLLM 0.10.1** repository.

Given a question, the system:

1. retrieves the most relevant passages of the repository with a BM25
   index (no neural retriever, no vector database);
2. returns them as *sources* (`file_path`, `first_character_index`,
   `last_character_index`), the format graded by the moulinette;
3. optionally feeds them to a small language model (`Qwen/Qwen3-0.6B`,
   configurable) which writes a short answer grounded in those passages.

Everything is driven by a single Fire CLI (`uv run python -m src ...`) with
commands to index the corpus, search one query or a whole dataset, answer,
and evaluate recall@k locally.

## Instructions

### Installation

Requirements: Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
make install        # or: uv sync
```

### Data layout

The repository does not contain any data (`data/` is git-ignored).
Put the files in place as follows:

```bash
mkdir -p data/raw data/datasets
unzip vllm-0.10.1.zip -d data/raw          # -> data/raw/vllm-0.10.1/...
cp -r <path>/UnansweredQuestions <path>/AnsweredQuestions data/datasets/
```

Resulting layout (all paths are CLI arguments with these defaults):

```
data/raw/vllm-0.10.1/                                   corpus
data/processed/                                         index (generated)
data/datasets/UnansweredQuestions/dataset_{docs,code}_public.json
data/datasets/AnsweredQuestions/dataset_{docs,code}_public.json
data/output/search_results/<Scope>/                     search results
data/output/search_results_and_answer/<Scope>/          results + answers
```

### Commands

| Command | Purpose |
| --- | --- |
| `index` | chunk the corpus and build the BM25 index in `data/processed` |
| `search` | top-k sources for one query (JSON on stdout) |
| `search_dataset` | top-k sources for every question of a dataset |
| `answer` | retrieve then generate an answer for one query |
| `answer_dataset` | generate answers for a search-results file |
| `evaluate` | recall@1/3/5/10 of a search-results file against a reference |

```bash
uv run python -m src index [--max_chunk_size 2000] [--raw_dir data/raw]
                           [--processed_dir data/processed]
                           [--exclude_dirs tests,benchmarks,...]
uv run python -m src search --query "..." [--k 10]
uv run python -m src search_dataset --dataset_path <file> [--k 10]
                           [--save_directory <dir>]
uv run python -m src answer --query "..." [--k 10] [--model_name Qwen/Qwen3-0.6B]
uv run python -m src answer_dataset --student_search_results_path <file>
                           [--save_directory <dir>] [--raw_dir_root .]
uv run python -m src evaluate --student_search_results_path <file>
                           --dataset_path <AnsweredQuestions file> [--k 10]
```

`--exclude_dirs` (default `tests,benchmarks,examples,csrc,.buildkite,
.github,tools`) lists **top-level** directories of the repository that are
not indexed (they hold almost no documentation or answer material and add
noise). To index everything, pass an empty string:
`uv run python -m src index --exclude_dirs ""`.

### Makefile

`make install`, `make run`, `make debug` (pdb), `make clean`, `make lint`
(flake8 + mypy), `make lint-strict` (flake8 + `mypy --strict`),
`make test` (pytest, no corpus or model needed).

### Evaluating with the moulinette

The moulinette binary is **not** part of this repository (graders bring
their own). With it available as `moulinette`:

```bash
uv run python -m src index
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000
```

The built-in `evaluate` command computes the same metric (a source counts
as a hit when it is in the same file and overlaps a reference span with
IoU >= 0.05) without the moulinette.

### Error handling

No command ever prints a traceback: empty query, non-integer `k`,
`k <= 0` (no sources, with a warning), missing or malformed JSON, schema
mismatch, missing/corrupt index, invalid `--max_chunk_size`, missing raw
directory or unavailable model all produce `Error: ...` on stderr.

## System architecture

```
 data/raw/vllm-0.10.1  (.py .md .txt .yaml ... files)
          |
          v
 [chunking.py]  python (ast) / markdown (headings) / plain-text chunkers
          |  Chunk(file_path, first, last, text)   chars <= max_chunk_size
          v
 [tokenizer.py] identifiers + camelCase/snake_case parts, stop words,
          |     plural stemming, word bigrams (non-.py files)
          v
 [index.py]     BM25 weights precomputed in a scipy sparse matrix
          |     -> data/processed/{bm25_matrix.npz,vocab.json,chunks.json,meta.json}
          v
 [retriever.py] query -> terms -> sparse mat-vec -> top-k chunks
          |
          +--> search / search_dataset  -> MinimalSearchResults / StudentSearchResults
          |
          v
 [generator.py] top-k chunks -> chat prompt -> Qwen3-0.6B -> answer
                 -> answer / answer_dataset -> StudentSearchResultsAndAnswer

 [models.py]     pydantic models exchanged between stages
 [evaluation.py] recall@k (IoU >= 0.05) for local checks
 [cli.py]        Fire entry point, validation and error handling
```

## Chunking strategy

Every chunk records its exact character span in the file, so
`content[first:last]` is the chunk text and `last - first <= max_chunk_size`
(default 2000, checked by invariant tests on many degenerate inputs).

* **Python** (`ast`): the file is cut into top-level statements; classes
  are split at their methods, with the decorators and the leading
  comments attached to the definition. Small neighbours are packed together
  up to the limit, tiny leftovers are merged; oversized pieces are split
  at blank lines, then lines, then spaces. Files with a syntax error fall
  back to the text chunker.
* **Markdown** (`.md`, `.mdx`, `.rst`): sections are delimited by ATX
  headings (fenced code blocks are respected); sections under 400
  characters are merged with the next one; long sections are split at
  paragraph boundaries. The chain of parent headings is added to the
  *indexed* text of each chunk (not to the returned span).
* **Other text** (`.txt`, `.yaml`, `.toml`, `.sh`, `.jinja`, ...):
  paragraph/line-aware splitting.
* Surrounding whitespace is trimmed from chunks, files over 1 MB are
  skipped, and files are decoded as UTF-8 with `errors="replace"`.

## Retrieval method

Okapi **BM25** (k1 = 1.2, b = 0.75, IDF `log(1 + (N-df+0.5)/(df+0.5))`)
implemented from scratch with numpy/scipy. All BM25 term weights are
computed at index time and stored in a sparse chunks x terms matrix; a
query is turned into a bag of terms and scored with one sparse
matrix-vector product, then the top-k are selected with `argpartition`.
Chunks with a zero score are never returned.

Text analysis (shared by indexing and querying):

* identifiers are kept whole *and* split on `_` and camelCase
  (`FusedMoEActivationFormat` -> `fusedmoeactivationformat`, `fused`,
  `moe`, ...), which lets natural-language questions hit code;
* lowercasing, a small stop-word list and trailing-`s` stemming;
* the file path (split into words, repeated twice) is prepended to each
  chunk's indexed text;
* adjacent **word bigrams** are indexed for non-Python files (queries
  always produce bigrams).

## Performance analysis

Official moulinette, k = 10, `--max_context_length 2000`, public datasets,
default `--max_chunk_size 2000`:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- |
| docs | 0.67 | 0.85 | 0.89 | 0.91 |
| code | 0.525 | 0.768 | 0.869 | 0.909 |

Both datasets meet the targets (Recall@5 >= 0.80 for docs, >= 0.50 for
code).

Speed (2 CPU cores): indexing the whole corpus takes about 5 s
(11.5k chunks); searching a 100-question dataset takes about 1 s.

**Tuning history** (docs Recall@5): baseline BM25 0.81 -> excluding
tests/benchmarks/examples/... 0.83 -> markdown heading context 0.85 ->
word bigrams on non-Python files 0.89. Bigrams on `.py` files as well
dropped code Recall@5 to about 0.81, so they are applied to non-Python
files only. BM25 parameters (k1 in 0.9-2.0, b in 0.5-0.9), path
repetition, a larger stop-word list and disabling stemming all moved
results by at most 1-2 points on 100 questions, i.e. within noise, so the
standard defaults were kept.

**Effect of a smaller chunk size** (`--max_chunk_size 1000`, 20.7k chunks,
index in 6.2 s, same moulinette settings):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- |
| docs | 0.73 | 0.87 | 0.90 | 0.93 |
| code | 0.535 | 0.758 | 0.818 | 0.899 |

Smaller chunks improve precision on documentation (Recall@1 0.67 -> 0.73,
Recall@10 0.91 -> 0.93) because a match is more concentrated, but they
split code definitions, which lowers code Recall@5 (0.869 -> 0.818). They
also give the generator less context per source. The default of 2000
was kept because it is the better overall trade-off and the one the
2000-character limit allows.

Generation: `Qwen3-0.6B` is a very small model. It follows a grounded
prompt but can miss details, paraphrase loosely or fail on questions that
need several passages, so answer quality is bounded by the model and by
the 3000-token context budget rather than by retrieval. Answer generation
was not benchmarked here.

## Design decisions

* **BM25 over embeddings**: no extra model or GPU for retrieval, indexing
  in seconds, deterministic, and strong on identifier-heavy technical text.
* **Precomputed sparse matrix**: scoring is one sparse product, so 100
  questions take about 1 s; the index is a handful of plain files.
* **Structure-aware chunking** (`ast`, markdown headings) keeps
  definitions and sections whole, so the returned span matches how
  reference answers are delimited.
* **Identifier splitting** bridges natural-language questions and code
  names; bigrams help prose (docs) but hurt code, hence non-`.py` only.
* **Directory exclusion** is configurable (`--exclude_dirs`); it is tuned
  on the public datasets, whose answers live in `vllm/`, `docs/` and the
  repository root. If a private set cites tests or examples, index with
  `--exclude_dirs ""`.
* **Generation**: the retrieved chunks are numbered and given to the model
  with their path and offsets, within a token budget (the last chunk is
  truncated if needed); thinking mode is disabled, decoding is greedy, and
  any `<think>` block is stripped. The model is loaded lazily, on CUDA,
  MPS or CPU. A failure on one question yields an empty answer and a
  warning instead of aborting the dataset.
* **pydantic** models validate every file exchanged between stages;
  **tqdm** shows progress for indexing and dataset search/answering; all
  files are opened through context managers.
* Tests (`tests/`) use synthetic corpora and a fake model, so they need
  neither the vLLM sources nor network access.

## Challenges faced

* **Recall on documentation** started at 0.81. Excluding noisy directories,
  adding parent-heading context to markdown chunks and word bigrams
  brought it to 0.89; each change was kept only if it helped without
  hurting code recall.
* **Over-tuning risk**: with 100 questions per set, most hyper-parameter
  changes are noise. Only changes with a structural justification were
  kept, and the exclusion of directories was restricted to top-level ones
  (a first version dropped `vllm/benchmarks`, which is real source).
* **camelCase splitting** initially turned `MoE` into `mo` + `e`; the
  regex was adjusted to keep acronyms intact.
* **Exact character offsets**: chunkers must return spans such that the
  text equals the file slice for any input (CRLF, unicode, no newlines,
  syntax errors, 1-character limits); this is enforced by invariant tests.
* **Never crash**: all commands are wrapped so that bad input, corrupt
  indexes or an unavailable model give a clear message, not a traceback.
* **Tiny generator**: a 0.6B model needs a short, strict prompt, a bounded
  context and disabled thinking to stay on topic.

## Example usage

```bash
uv sync
uv run python -m src index
uv run python -m src search --query "How do I enable LoRA adapters?" --k 5
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
uv run python -m src answer --query "How do I enable LoRA adapters?"
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json
```

`search` prints JSON such as:

```json
{
  "question_id": "...",
  "question": "How do I enable LoRA adapters?",
  "retrieved_sources": [
    {"file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
     "first_character_index": 0, "last_character_index": 1890}
  ]
}
```

(the offsets above are illustrative of the format.)

## Resources

* S. Robertson, H. Zaragoza, *The Probabilistic Relevance Framework: BM25
  and Beyond*, Foundations and Trends in Information Retrieval, 2009.
* P. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks*, NeurIPS 2020 (arXiv:2005.11401).
* Qwen3 model card: https://huggingface.co/Qwen/Qwen3-0.6B
* vLLM documentation: https://docs.vllm.ai
* Python `ast` module: https://docs.python.org/3/library/ast.html
* Python Fire: https://github.com/google/python-fire
* tqdm: https://tqdm.github.io/ ; pydantic: https://docs.pydantic.dev/
* Hugging Face Transformers: https://huggingface.co/docs/transformers

### AI usage

Claude (Anthropic) was used to plan, implement, test and document this
project under the student's direction, including the retrieval tuning
experiments (chunking variants, tokenizer options, BM25 parameters) and
this README. The student reviewed the code and is responsible for
understanding it.
