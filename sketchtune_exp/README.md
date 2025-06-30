# To Run SFT Experiments:
1. Copy the SketchTune SFT implementation:
```
git clone https://github.com/Davids048/sketchtune.git
```
2. Download data into the `data` folder.
    - `format_json`.{py,sh} converts raw text data into json files for dataset creation.
3. Create experiment configs in the `configs` folder.
4. Run the experiments using `train/finetune.sh`
    - Need to export `sketchtune_exp`'s directory as an env variable 
    `${EXP_DIR}`
5. Experiment runs will be in `exp_runs` folder. 
