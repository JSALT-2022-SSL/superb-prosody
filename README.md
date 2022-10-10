
# SUPERB-prosody: ON THE UTILITY OF SELF-SUPERVISED MODELS FOR PROSODY-RELATED TASKS

```Stay tuned for the update```
## Abstract
Self-Supervised Learning (SSL) from speech data has produced models that have achieved remarkable performance in many tasks, and that are known to implicitly represent many aspects of information latently present in speech signals. However, relatively little is known about the suitability of such models for prosody-related tasks or the extent to which they encode prosodic information. We present a new evaluation framework, ``SUPERB-prosody,'' consisting of three prosody-related downstream tasks and two pseudo tasks. We find that 13 of the 15 SSL models  outperformed the baseline on all the prosody-related tasks. We also show good performance on two pseudo tasks: prosody reconstruction and future prosody prediction. We further analyze the layerwise contributions of the SSL models. Overall we conclude that SSL speech models are highly effective for prosody-related tasks.

## Introduction and Usages

## Installation

1. **Python** >= 3.6
2. Install **sox** on your OS
3. Install s3prl

```sh
pip install -e ./
```

4. Install the specific fairseq

```sh
pip install fairseq@git+https://github.com//pytorch/fairseq.git@f2146bdc7abf293186de9449bfa2272775e39e1d#egg=fairseq
```


