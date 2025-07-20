

# for speech-to-text
import speech_recognition as sr

# for text-to-speech
from gtts import gTTS

# for language model
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os

import time

# for data
import os
import datetime
import numpy as np


# Building the AI
class ChatBot():
    def __init__(self, name):
        print("----- Starting up", name, "-----")
        self.name = name

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("Listening...")
            audio = recognizer.listen(mic)
            self.text="ERROR"
        try:
            self.text = recognizer.recognize_google(audio)
            print("Me  --> ", self.text)
        except:
            print("Me  -->  ERROR")
            

    @staticmethod
    def text_to_speech(text):
        print("Dev --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)

        speaker.save("res.mp3")
        statbuf = os.stat("res.mp3")
        mbytes = statbuf.st_size / 1024
        duration = mbytes / 200
        os.system('start res.mp3')  #if you are using mac->afplay or else for windows->start
        # os.system("close res.mp3")
        time.sleep(int(50*duration))
        os.remove("res.mp3")
        
        

    def wake_up(self, text):
        return True if self.name in text.lower() else False

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')


# Running the AI
if __name__ == "__main__":
    
    ai = ChatBot(name="dev")
    
    # Load the model and tokenizer for text generation
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set environment variable for tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    ex=True
    while ex:
        ai.speech_to_text()

        ## wake up
        if ai.wake_up(ai.text) is True:
            res = "Hello I am Dave the AI, what can I do for you?"
        
        ## action time
        elif "time" in ai.text:
            res = ai.action_time()
        
        ## respond politely
        elif any(i in ai.text for i in ["thank","thanks"]):
            res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","mention not"])
        
        elif any(i in ai.text for i in ["exit","close"]):
            res = np.random.choice(["Tata","Have a good day","Bye","Goodbye","Hope to meet soon","peace out!"])
            
            ex=False
        ## conversation
        else:   
            if ai.text=="ERROR":
                res="Sorry, come again?"
            else:
                # Encode the input text
                input_ids = tokenizer.encode(ai.text + tokenizer.eos_token, return_tensors='pt')
                
                # Generate response
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_length=1000,
                        pad_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=3,
                        do_sample=True,
                        top_k=100,
                        top_p=0.7,
                        temperature=0.8
                    )
                
                # Decode the response
                res = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
                
                # Clean up the response
                if not res.strip():
                    res = "I'm not sure how to respond to that."

        ai.text_to_speech(res)
    print("----- Closing down Dev -----")
    