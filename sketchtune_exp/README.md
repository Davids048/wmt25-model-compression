# To Run SFT Experiments:
- Copy the SketchTune SFT implementation:
```
git clone https://github.com/Davids048/sketchtune.git
```
- Copy LUTs into the `quantizers` folder.
- Export `sketchtune_exp`'s directory as an env variable `${EXP_DIR}`
- Download data into the `data` folder.
    - `format_json.{py,sh}` converts raw text data into json files for dataset creation.
    - Type `python format_json.py -h` for help.
- Create experiment configs in the `configs` folder.
    - Full config is defined in `utils/config.py`.
- Run the experiments using `train/finetune.sh`.
- Experiment runs will be in `exp_runs` folder. 



