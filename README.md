# Adv-OLM: Generating Textual Adversaries via OLM
This repository will contain the code for our EACL 2021 paper:<br><br>
Adv-OLM: Generating Textual Adversaries via OLM <a href="https://arxiv.org/pdf/2101.08523.pdf"> [Link to Paper]</a>

### Environment setup
- For running in your system:
```bash
conda create -n textattack python=3.6
conda activate textattack
pip install -r requirements.txt
```
- For running on colab:
```bash
pip install -r requirements.txt
```

### Example code run
```bash
python attack_main.py --recipe advolm --batch-size 8 --num-examples 5 --model bert-base-uncased-imdb
```
- For running other attacks like Textfooler, Bae, Pwws, etc., use [TextAttack](https://github.com/QData/TextAttack) 
