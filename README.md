## Attention for Natural Langauge Processing

1. Install
```
conda create -n torchtext python=3.8
conda activate torchtext
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
python -m pip install torchtext==0.13.0
python -m pip install torchdata==0.4.0
python -m pip install spacy

python -m spacy download en
python -m spacy download de
```

2. Dot product attention
```
1) train
python main.py --save_path ./output/dot_prod_model.pth --attn_type dot_prod

2) test
python eval_blue.py --load_path ./output/dot_prod_model.pth --attn_type dot_prod

3) result
blue score : 32.81

ex1.
source : Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen
target : A group of men are loading cotton onto a truck
output : A group of men loading cotton candy on a truck.

ex2.
source : Ein kleines Kind steht allein auf einem zerklüfteten Felsen.
target : A young child is standing alone on some jagged rocks.
output : A small child stands alone on a cliff , in a low cliff.

```

3. Vector product attention
```
1) train
python main.py --save_path ./output/vec_prod_model.pth --attn_type vec_prod

2) test
python eval_blue.py --load_path ./output/vec_prod_model.pth --attn_type vec_prod

3) result(bug)
blue score : 9.58

ex1.
source : Eine Gruppe von Männern lädt Baumwolle auf einen Lastwagen
target : A group of men are loading cotton onto a truck
output : A A group of A group of men.

ex2.
source : Ein kleines Kind steht allein auf einem zerklüfteten Felsen.
target : A young child is standing alone on some jagged rocks.
output : A A young child is standing on a donkey.
```
