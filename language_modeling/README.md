# Language modeling

Multi-layer Recurrent Neural Networks (LSTM, RNN) for char and word-level language models in Python using TensorFlow.

Mostly reused code from https://github.com/sherjilozair/char-rnn-tensorflow and https://github.com/hunkim/word-rnn-tensorflow.

# Requirements
- [Tensorflow](http://www.tensorflow.org)

# Basic Usage
Examples of how to use this code are shown in playground.py

To sum it up, this project trains char or word-level language model on provided dataset (default is tinyshakespeare text located in data folder). After the model is trained, it can be used to sample text or determine sequence probability (mostly for ranking hypothesises).

# Sample output

## Word-RNN
```
LEONTES:
Why, my Irish time?
And argue in the lord; the man mad, must be deserved a spirit as drown the warlike Pray him, how seven in.

KING would be made that, methoughts I may married a Lord dishonour
Than thou that be mine kites and sinew for his honour
In reason prettily the sudden night upon all shalt bid him thus again. times than one from mine unaccustom'd sir.

LARTIUS:
O,'tis aediles, fight!
Farewell, it himself have saw.

SLY:
Now gods have their VINCENTIO:
Whipt fearing but first I know you you, hinder truths.

ANGELO:
This are entitle up my dearest state but deliver'd.

DUKE look dissolved: seemeth brands
That He being and
full of toad, they knew me to joy.
```

## Char-RNN

```
ESCALUS:
What is our honours, such a Richard story
Which you mark with bloody been Thilld we'll adverses:
That thou, Aurtructs a greques' great
Jmander may to save it not shif theseen my news
Clisters it take us?
Say the dulterout apy showd. They hance!

AnBESS OF GUCESTER:
Now, glarding far it prick me with this queen.
And if thou met were with revil, sir?

KATHW:
I must not my naturation disery,
And six nor's mighty wind, I fairs, if?

Messenger:
My lank, nobles arms;
```
