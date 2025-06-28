# WMT25 Model Compression

This repository provides a baseline and a template submission for the WMT25
Model Compression shared task. For task details, visit the [official
page](https://www2.statmt.org/wmt25/model-compression.html).


## 1. Submission requirements

A submission to the shared task requires both a Dockerfile and a Docker image
containing all necessary software and model files for translation, following the
requirements below.

- Include your team’s short name (no spaces) in both the Dockerfile and image
  name, e.g.: `$Team-Dockerfile` and `$Team-dockerimage.tar`

- The image must contain a model directory at `/model/$submission_id` with all
  required files (model, vocabulary, etc.). Please keep `$submission_id` short
  as it will appear in reports.

- You may include additional files, but **do not** use any paths starting with
  `/wmt`—these are reserved for the task evaluation.

- Each model directory must include a `run.sh` script with the following interface:

    ```bash
    /model/$submission_id/run.sh $lang_pair $batch_size < input.txt > output.txt
    ```
    - `$lang_pair`: Language pair in the format `eng-deu`
    - `$batch_size`: Positive integer
    - The script must run without accessing Internet.

To participate, fill the form mentioned in the official website, where you need to
submit a public link to the Docker image online with its sha512sum sum.

### Example Usage

```bash
image_name="$(docker load -i ${image_file_path} | cut -d ' ' -f 3)"
container_id="$(docker run -itd ${opt_memory} --memory-swap=0 ${image_name} bash)"
(time docker exec -i "${container_id}" /model/$submission_id/run.sh $lang_pair $batch_size < input.txt > output.txt 2> stderr.txt)
```

## 2. Baseline

### Setup

1. Installation
```bash
pip install -e .
```

2. (Optional) Hugging Face Login
```bash
huggingface-cli login
```
*Required for gated models (e.g., [aya-expanse-8b](https://huggingface.co/CohereLabs/aya-expanse-8b)).*

3. Download Models & Test Sets

```bash
python -m modelzip.setup
```

This command will create the following
**Directory Structure:**
```
workdir/
├── models/
│   └── aya-expanse-8b-base
└── tests/
  ├── ces-deu/
  └── jpn-zho/
```
*Note: Test sets here are for development only. Official evaluation will use different data.*

---
### Compression Demo

```bash
python -m modelzip.compress
```

This command will create two baselines in addition to the base aya model.
The compressed models are saved in `workdir/models/`.

### Running Baselines

```bash
for m in workdir/models/aya-expanse-8b-*; do
  python -m modelzip.evaluate -m $m
  # python -m modelzip.evaluate -m ./workdir/models/aya-expanse-8b-base/
done

python -m modelzip.report
```

**Sample Output:**
```
wmt19.ces-deu.deu.aya-expanse-8b.base.out.chrf      54.5
wmt19.ces-deu.deu.aya-expanse-8b.bnb-4bit.out.chrf  54.2
wmt19.ces-deu.deu.aya-expanse-8b.bnb-8bit.out.chrf  54.5
wmt24.jpn-zho.zho.aya-expanse-8b.base.out.chrf      24.4
...
```

## 3. Submission preparation

See `Dockerfile` for an example. You can use [modelzip/baseline.py](modelzip/baseline.py)
as a template to create your `run.py` script that uses the compressed model to
translate the input. Specifically, you just need to implement your own version
of the `LLMWrapper` class. Then you can simply add a `run.sh` file containing:

```bash
python run.py $1 $2 <&0 >&1
```

Once everything is ready, you can create a Docker image (in this case we name
it `wmt2025_modelcompression`, but you can chose your preferred name) with:
```bash
docker build -t wmt2025_modelcompression .
```
Then, the image can be stored in a `tar` file with:
```bash
docker save --output $DOCKER_IMAGE_FILENAME.tar wmt2025_modelcompression
```
