Speech-To-Text Test
===================

Test and evaluate the accuracy of various speech-to-text systems.

Audio samples are standardized on mono-channel with a rate of 16k.

Mozilla Deep Speech
-------------------

https://github.com/mozilla/DeepSpeech

./test_deepspeech.py

    2018-1-8
    
        version: 0.1.0
        diff accuracy: 0.728972718858
        abs accuracy: 0.433333333333
        real: 40s

    2018-7-12

        version: 0.1.0
        diff accuracy: 0.774034944463
        abs accuracy: 0.253521126761
        real: 4m28.3034749031s
        samples: 142

    2018-10-8

        version: 0.2.0
        diff accuracy: 0.7217131145921839
        abs accuracy: 0.028169014084507043
        real: 4m28.3034749031s
        samples: 142

Google Speech
-------------

https://pypi.python.org/pypi/SpeechRecognition/

./test_googlespeech.py

    2018-1-10

        diff accuracy: 0.726952838341
        abs accuracy: 0.6
        real: 19s

    2018-7-12

        diff accuracy: 0.819401803119
        abs accuracy: 0.394366197183
        real: 1m36.2234201431s
        samples: 142

    2018-10-8
    
        diff accuracy: 0.819401803119
        abs accuracy: 0.394366197183
        real: 1m33.689s
        samples: 142

PocketSphinx
------------

https://pypi.python.org/pypi/SpeechRecognition/
https://github.com/bambocher/pocketsphinx-python

./test_pocketsphinx.py

    2018-1-10
    
        diff accuracy: 0.650675023999
        abs accuracy: 0.3
        real: 25s

    2018-7-12

        diff accuracy: 0.675711847027
        abs accuracy: 0.140845070423
        real: 2m4.07494902611s
        samples: 142

    2018-10-8
    
        diff accuracy: 0.6639683027589178
        abs accuracy: 0.16901408450704225
        real: 1m29.295s
        samples: 142

Houndify
--------

https://pypi.python.org/pypi/SpeechRecognition/
https://www.houndify.com

./test_houndify.py

    2018-1-10
    
        diff accuracy: 0.853484911794
        abs accuracy: 0.6
        real: 35s

    2018-7-12

        diff accuracy: 0.859605148697
        abs accuracy: 0.394366197183
        real: 2m57.0215849876s
        samples: 142

    2018-10-8

        diff accuracy: 0.860153219863991
        abs accuracy: 0.4084507042253521
        real: 3m16.033s
        samples: 142

Bing
----

./test_bingspeech.py

    2018-1-10
    
        time: 2m13s
        diff accuracy: 0.837140933267
        abs accuracy: 0.666666666667

    2018-7-12

        ?

Ensemble
--------

./test_ensemble.py

    2018-1-10
    
        time: 2m25s
        diff accuracy: 0.922982141044
        abs accuracy: 0.733333333333

    2018-7-12

        diff accuracy: 0.86423527877
        abs accuracy: 0.43661971831
        real: 10m7.54498410225s
        samples: 142
