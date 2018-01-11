Speech-To-Text Test
===================

Test and evaluate the accuracy of various speech-to-text systems.

Mozilla Deep Speech
-------------------

https://github.com/mozilla/DeepSpeech

./test_deepspeech.py

    2018-1-8
    
        time: 40s
        diff accuracy: 0.728972718858
        abs accuracy: 0.433333333333

Google Speech
-------------

https://pypi.python.org/pypi/SpeechRecognition/

./test_googlespeech.py

    2018-1-10

        time: 19s
        diff accuracy: 0.726952838341
        abs accuracy: 0.6

PocketSphinx
------------

https://pypi.python.org/pypi/SpeechRecognition/
https://github.com/bambocher/pocketsphinx-python

./test_pocketsphinx.py

    2018-1-10
    
        time: 25s
        diff accuracy: 0.650675023999
        abs accuracy: 0.3

Houndify
--------

https://pypi.python.org/pypi/SpeechRecognition/
https://www.houndify.com

./test_houndify.py

    2018-1-10
    
        time: 35s
        diff accuracy: 0.853484911794
        abs accuracy: 0.6

Bing
----

./test_bingspeech.py

    2018-1-10
    
        time: 2m13s
        diff accuracy: 0.837140933267
        abs accuracy: 0.666666666667

Ensemble
--------

./test_ensemble.py

    2018-1-10
    
        time: 2m25s
        diff accuracy: 0.922982141044
        abs accuracy: 0.733333333333
