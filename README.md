Speech-To-Text Test
===================

Test and evaluate the accuracy of various speech-to-text systems.

Audio samples are standardized on mono-channel with a rate of 16k.

# Setup

To evaluate each system, run the corresponding setup_* script and then test_* script in the src directory.

For some, you might have to check the project's homepage and update the script to refer to the latest model file.

# Results

Mozilla Deep Speech
-------------------

https://github.com/mozilla/DeepSpeech

./setup_deepspeech.sh

Google Speech
-------------

https://pypi.python.org/pypi/SpeechRecognition/

./setup_googlespeech.sh

PocketSphinx
------------

https://pypi.python.org/pypi/SpeechRecognition/
https://github.com/bambocher/pocketsphinx-python

./setup_pocketsphinx.sh

Houndify
--------

https://pypi.python.org/pypi/SpeechRecognition/
https://www.houndify.com

./setup_houndify.sh

Bing
----

Microsoft ended their public API. No further tests are possible.

./setup_bingspeech.sh

IBM
---

./setup_ibmspeech.sh

AssemblyAI
----------

./setup_assemblyai.sh

Vosk
----

https://alphacephei.com/vosk

./setup_vosk.sh

Ensemble
--------

A weighted average of all of the above.

./setup_ensemble.sh
