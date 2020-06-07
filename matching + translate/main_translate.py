from __future__ import division
import os
#import serv

globalTags = ""
globalTranslate = ""

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../DigitalLabTest-3659f2cf8dcb.json"

#irumata
import pygame
from textrazor import TextRazor
textrazor_api = "2fded875a6ccf370201599cc593d1ff67d45703f94448e7facce3e87"

def transcribe_streaming(stream_file):
    """Streams transcription of the given audio file."""
    import io
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    with io.open(stream_file, 'rb') as audio_file:
        content = audio_file.read()

    # In practice, stream should be a generator yielding chunks of audio data.
    stream = [content]
    requests = (types.StreamingRecognizeRequest(audio_content=chunk)
                for chunk in stream)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US')
    streaming_config = types.StreamingRecognitionConfig(config=config)


    # streaming_recognize returns a generator.
    responses = client.streaming_recognize(streaming_config, requests)

    for response in responses:
        # Once the transcription has settled, the first result will contain the
        # is_final result. The other results will be for subsequent portions of
        # the audio.
        for result in response.results:
            print('Finished: {}'.format(result.is_final))
            print('Stability: {}'.format(result.stability))
            alternatives = result.alternatives
            # The alternatives are ordered from most likely to least.
            for alternative in alternatives:
                print('Confidence: {}'.format(alternative.confidence))
                print(u'Transcript: {}'.format(alternative.transcript))

from googletrans import Translator
from gtts import gTTS

from rutermextract import TermExtractor
 

import time




import re
import sys

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue
from tempfile import NamedTemporaryFile
            
from io import BytesIO

# Audio recording parameters
RATE = 8000
CHUNK = int(RATE / 10)  # 100ms


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


def listen_print_loop(responses, previous, client):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
  #  client_TTS = texttospeech.TextToSpeechClient()
    trans = Translator()
    speech_is_on=True
    client_entity = TextRazor(textrazor_api, extractors=["entities"])
    tags = []
    prev_len = 0
    current_main_phrase = ""
    start_time = time.time()
    start_time_chank = time.time()
    already_readed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))
        if not result.is_final and time.time()-start_time_chank<10 :
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            actual_transcript_chank = transcript[already_readed:]
            previous+=actual_transcript_chank
            if not result.is_final:
                already_readed = len(transcript)
            else:
                already_readed = 0
            start_time_chank = time.time()
        
            if len(previous) - prev_len > 100:
                
                response = client_entity.analyze(previous)
                res_dict={}
                for entity in response.entities():
                    print("ent", entity.matched_text)
                    if len(entity.matched_text) > 4:
                        res_dict[entity.matched_text[-2].lower()]=entity.matched_text.lower()
                    else:
                        if not entity.matched_text.lower().isnumeric():
                            res_dict[entity.matched_text.lower()]=entity.matched_text.lower()
                for topic in response.topics():
                    print("top",topic.label, topic.score)
                    res_dict[topic.label]=topic.label
                for topic in response.categories():
                    print("top",topic.label, topic.score)
                    res_dict[topic.label]=topic.label
                                   
                tags = list(res_dict.values())
                #globalTags = tags
                prev_len = len(previous)

#             term_extractor = TermExtractor()
#             terms = list()
#             for term in term_extractor(previous):
#                 terms.append(term.normalized)
            #print(tags)
            
            #print(actual_transcript_chank)
            globalTags = actual_transcript_chank
            #print(previous + overwrite_chars)
            src_l = trans.detect(actual_transcript_chank).lang
            if src_l == "en":
                dst_l = "ru"
            else:
                dst_l = "en"
            if src_l != "en":
                trans_text = trans.translate(text=actual_transcript_chank,dest=dst_l, src=src_l).text
            else:
                trans_text = " speak on english"
            #print(trans_text)
            
            # Instantiates a client
            if not speech_is_on:

                fp = BytesIO()

                f = NamedTemporaryFile()

                gTTS(trans_text).write_to_fp(fp)

                fp.seek(0)
                pygame.mixer.init()
                pygame.mixer.music.load(fp)
                pygame.mixer.music.play()
             #   while pygame.mixer.music.get_busy():
             #       pygame.time.Clock().tick(10)

            
            
        #  print ("lang ",src_l, "trans to ",dst_l, trans_text)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit|заканчиваем запись)\b', transcript, re.I):
                print('Exiting..')
                break
            if re.search(r'\b(стоп голос)\b', transcript, re.I):
                speech_is_on = False
            if re.search(r'\b(включи голос)\b', transcript, re.I):
                speech_is_on = True
            if re.search(r'\b(важно|запоминаем)\b', transcript, re.I):
                current_main_phrase=transcript.replace("важно","").replace("запоминаем","")
                start_time = time.time()
            if (time.time() - start_time) > 30:
                current_main_phrase = ""
            print(" Важное: ",current_main_phrase )
            num_chars_printed = 0

#time.time() - start_time

"23g4"

def main():
    pass
    '''
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'ru-RU'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code
    )
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        print("send req")
        requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)
        print("get resp")

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        print("print resp")

        listen_print_loop(responses, "", client)
        '''

def translate():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'ru-RU'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code
    )
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        print("send req")
        requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)
        print("get resp")

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        print("print resp")

        listen_print_loop(responses, "", client)


if __name__ == "__main__":
    main()