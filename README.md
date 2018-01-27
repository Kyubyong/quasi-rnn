# A TensorFlow Implementation of Character Level Neural Machine Translation Using the Quasi-RNN

In [Bradbury et al., 2016](https://arxiv.org/abs/1611.01576) (hereafter, the Paper), the authors introduce a new neural network model which they call the Quasi-RNN. Basically, it tries to benefit from both CNNs and RNNs by combining them. The authors conducted three experiments to evaluate the performance of the Q-RNN. Character level machine translation is one of them. After the Paper was published, some enthusiasts tried to reproduce the experiments as the authors didn't disclose their source codes. Now I'm happy to be one of them. To my best knowledge, this is the first TensorFlow implementation of character level machine translation based on the Paper.

## Requirements
  * numpy >= 1.11.1
  * TensorFlow == 1.0
  * sugartensor >= 1.0.0.2
  * nltk >= 3.2.2 (only for calculating the bleu score)

## Some notes on implementation

Overall, we tried to follow the instructions in the Paper. Some major differences are as follows.

* The Paper set the maximum sequence length to 300 characters, but we changed it to 150 due to the limitation of our single gpu (GTX 1080 8GB).
* We applied a greedy decoder at the inference phase, not the beam search decoder.
* We didn't reverse source sentences.

## Work Flow

* STEP 1. Download [IWSLT 2016 German–English parallel corpus](https://wit3.fbk.eu/download.php?release=2016-01&type=texts&slang=de&tlang=en) and extract it to `corpora/` folder.
* STEP 2. Run `train.py`.
* STEP 3. Run `eval.py` to get the results for the test sentences.

Or if you'd like to use the pretrained model,

* Download the [pre-trained model files](https://www.dropbox.com/s/lhjlz0492xna977/qrnn.tar.gz?dl=0), then extract them to `asset/train/` folder.
* Run eval.py.

## Evaluation & Results

Our best model obtained Bleu Score of 15.145749415. The details are available in `model.ckpt-50604`. 


## Papers that referenced this repo

  * [Machine Translation of Low-Resource Spoken Dialects: Strategies for Normalizing Swiss German](https://arxiv.org/pdf/1710.11035.pdf)







