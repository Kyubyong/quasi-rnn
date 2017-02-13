# A TensorFlow Implementation of Character Level Neural Machine Translation Using Quasi-RNNs

In [Bradbury et al., 2016](https://arxiv.org/abs/1611.01576) (hereafter, the Paper), the authors introduce a new neural network model which they call the Quasi-RNN. Basically, it tries to benefit from both CNNs and RNNs by combining them. The authors conducted three experiments to evaluate the performance of the Q-RNN. Character level machine translation is one of them. After the Paper was published, some enthusiasts tried to reproduce the experiments as the authors didn't diclose their source codes. Now I'm happy to be one of them. To my best knowledge, this is the first TensorFlow implementation of character level machine translation based on the Paper.

## Requirements
  * numpy >= 1.11.1
  * sugartensor >= 0.0.2.4
  * nltk >= 3.2.2 (only for calculating the bleu score)

## Some notes on implementation

Overall, we tried to follow the instructions in the Paper. Some major differences are as follows.

* The Paper set the maximum sequence length to 300 characters, but we changed it to 150 due to the limitation of our single gpu (GTX 1080 8GB).
* We applied a greedy decoder at the inference phase, not the beam search.
* We didn't reverse source sentences because simply we didn't like the idea :) (We know in some papers it worked well, though.)


## Work Flow

* STEP 1. Download [IWSLT 2016 German–English parallel corpus](https://wit3.fbk.eu/download.php?release=2016-01&type=texts&slang=de&tlang=en) and extract it to `corpora/` folder.
* STEP 2. Run `prepro.py` to make training / test data.
* STEP 3. Run `train.py`.
* STEP 4. Run `eval.py` to get the results for the test sentences.

Or if you'd like to use the pretrained model,

* Download the [output files](https://drive.google.com/open?id=0B0ZXk88koS2KcU5vTjlhcFpwQUk) of STEP 3, then place them to `data/` folder.
* Download the [pre-trained model files](https://drive.google.com/open?id=0B0ZXk88koS2KcUtFblFiai1BM0k), then place them to `asset/train/` folder.
* Run eval.py.

## Evaluation & Results

As shown in the Paper, we trained for 10 epochs with `train.tags.de-en` files and evaluated with `TED.tst2014.de-en` files. We obtained the Bleu Score of 45.9151352081. The details are available in `results.txt`.


	







