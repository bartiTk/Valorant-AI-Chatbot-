#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications
"""
import csv
import io

import pandas
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import pyttsx3
import spacy
import cv2
import azure.cognitiveservices.speech as speechsdk
from nltk.inference import ResolutionProver
from nltk.sem import Expression
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

#######################################################
#  Initialise AIML agent
#######################################################
import aiml

# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="mybot-basic.xml")

#######################################################
# Loading trained Dataset
#######################################################
model = keras.models.load_model("valo_agent.h5")

#######################################################
# Welcome user
#######################################################
print("Welcome to this chat bot. Please feel free to ask questions about Valorant from me")

#######################################################
# create global variables
#######################################################
questionsCSV = []
answerCSV = []
kb = []
engine = pyttsx3.init()  # using pyttsx3 because it can run offline and is easy to setup

# ---------------------------------------------------------------------------------------------------#
#######################################################
# Turn CSV file questions into an array
#######################################################
nlp = spacy.load('en_core_web_sm')

with open("mybot-basicExtra.csv") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for line in csv_reader:
        questionsCSV.append(line[0])

with open("mybot-basicExtra.csv") as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for line in csv_reader:
        answerCSV.append(line[1])

#######################################################
# Lemmatizing the array
#######################################################
lemQuestions = []
for i in questionsCSV:
    doc = nlp(i)
    lemSent = " ".join([token.lemma_ for token in doc])
    lemQuestions.append(lemSent)

#######################################################
# Vectorizing the questions
#######################################################
vectorizer = TfidfVectorizer()
vecQuestions = vectorizer.fit_transform(lemQuestions)


# ---------------------------------------------------------------------------------------------------#

#######################################################
# Azure Voice Recognition
#######################################################
def voice_from_microphone():
    voice_config = speechsdk.SpeechConfig(subscription="c1e5aae958f94e298f90805becd4c12d", region="uksouth")
    voice_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    voice_recognizer = speechsdk.SpeechRecognizer(speech_config=voice_config, audio_config=audio_config)

    print("Speak into the microphone")
    voice_recognition_result = voice_recognizer.recognize_once_async().get()

    if voice_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(voice_recognition_result.text))
        return voice_recognition_result.text
    elif voice_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(voice_recognition_result.no_match_details))
    elif voice_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = voice_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")


# ---------------------------------------------------------------------------------------------------#

#######################################################
# read_expr
#######################################################
read_expr = Expression.fromstring

data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
#######################################################
# Main loop
#######################################################
while True:
    # get user input
    try:
        userInput = input("> ")
        if userInput.lower() == "voice":
            userInput = voice_from_microphone()
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        engine.say("Bye")
        break
    # pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    # activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    # post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.askopenfilename(
                filetypes=[("Image Files", "*.jpg;*.jpeg")]
            )
            image = Image.open(filepath)
            # Converting into Bytes
            image_byte_arr = io.BytesIO()
            image.save(image_byte_arr, "jpeg")
            image_bytes = image_byte_arr.getvalue()
            # change to grayscale
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            # Make sure the image is the same as the model layout
            img = cv2.resize(img, (64, 64))
            img = np.expand_dims(img, axis=0)

            predictions = model.predict(img)
            predicted_class = np.argmax(predictions, axis=1)[0]
            # Map the Agent name to what the model thinks it is
            agent_names = ["Astra", "Breach", "Brimstone", "Chamber", "Cypher", "Fade", "Harbor", "Jett", "KayO",
                           "Killjoy", "Neon", "Omen", "Phoenix", "Raze", "Reyna", "Sage",
                           "Skye", "Sova", "Viper", "Yoru"]
            agent_name = agent_names[predicted_class]
            # Show the agent name to the User
            print(agent_name)
            engine.say(agent_name)
            engine.runAndWait()


        elif cmd == 31:  # if input pattern is "I know that * is *"
            object, subject = params[1].split(' is ')
            originalExpr = read_expr(subject + '(' + object + ')')
            # >>> ADD SOME CODES HERE to make sure expr does not contradict
            # with the KB before appending, otherwise show an error message.
            contradict = True
            #Checks the subject of Agent and if it contradicts with any ability or role already in the system
            if subject == 'agent':
                expr = read_expr('role' + '(' + object + ')')
                answer = ResolutionProver().prove(expr, kb, verbose=False)
                if answer:
                    print('Sorry this contradicts with what I know!')
                else:
                    expr = read_expr('ability' + '(' + object + ')')
                    answer = ResolutionProver().prove(expr, kb, verbose=False)
                    if answer:
                        print('Sorry this contradicts with what I know!')
                    else:
                        contradict = False

            #Checks the subject of role and if it contradicts with any already know agents or abilities
            elif subject == 'role':
                expr = read_expr('agent' + '(' + object + ')')
                answer = ResolutionProver().prove(expr, kb, verbose=False)
                if answer:
                    print('Sorry this contradicts with what I know!')
                else:
                    expr = read_expr('ability' + '(' + object + ')')
                    answer = ResolutionProver().prove(expr, kb, verbose=False)
                    if answer:
                        print('Sorry this contradicts with what I know!')
                    else:
                        contradict = False

            #Checks the subject of ability and if it contradicts with any agents or roles that are already known
            elif subject == 'ability':
                expr = read_expr('agent' + '(' + object + ')')
                answer = ResolutionProver().prove(expr, kb, verbose=False)
                if answer:
                    print('Sorry this contradicts with what I know!')
                else:
                    expr = read_expr('role' + '(' + object + ')')
                    answer = ResolutionProver().prove(expr, kb, verbose=False)
                    if answer:
                        print('Sorry this contradicts with what I know!')
                    else:
                        contradict = False

            #Passed Validation. Store the relationship in the kb.csv
            if not contradict:
                kb.append(originalExpr)
                print('OK, I will remember that', object, 'is', subject)




        elif cmd == 32:  # if the input pattern is "check that * is *"
            object, subject = params[1].split(' is ')
            expr = read_expr(subject + '(' + object + ')')
            answer = ResolutionProver().prove(expr, kb, verbose=False)
            if answer:
                print('Correct.')
            else:
                print('It may not be true... let me check...')
                # >> This is not an ideal answer.
                # >> ADD SOME CODES HERE to find if expr is false, then give a
                # definite response: either "Incorrect" or "Sorry I don't know."

                #Checks subject agent to any role or ability
                if subject == 'agent':
                    expr = read_expr('role' + '(' + object + ')')
                    answer = ResolutionProver().prove(expr, kb, verbose=False)
                    if answer:
                        print('Incorrect.')
                    else:
                        expr = read_expr('ability' + '(' + object + ')')
                        answer = ResolutionProver().prove(expr, kb, verbose=False)
                        if answer:
                            print('Incorrect.')
                        else:
                            print('Sorry I don\'t know.')

                #Checks subject role to any agent or ability
                elif subject == 'role':
                    expr = read_expr('agent' + '(' + object + ')')
                    answer = ResolutionProver().prove(expr, kb, verbose=False)
                    if answer:
                        print('Incorrect.')
                    else:
                        expr = read_expr('ability' + '(' + object + ')')
                        answer = ResolutionProver().prove(expr, kb, verbose=False)
                        if answer:
                            print('Incorrect.')
                        else:
                            print('Sorry I don\'t know.')

                #Checks subject ability to any agent or role
                elif subject == 'ability':
                    expr = read_expr('agent' + '(' + object + ')')
                    answer = ResolutionProver().prove(expr, kb, verbose=False)
                    if answer:
                        print('Incorrect.')
                    else:
                        expr = read_expr('role' + '(' + object + ')')
                        answer = ResolutionProver().prove(expr, kb, verbose=False)
                        if answer:
                            print('Incorrect.')

                        #If cant find any match the program cannot be 100% sure that the statement is incorrect beacuse it doesnt have the data
                        else:
                            print('Sorry I don\'t know.')

        elif cmd == 99:
            # IF not a AIML answer, will look for the closest answer to question in CSV
            ########## TASK 1 ##########
            lemInput = " ".join(
                [token.lemma_.lower() for token in nlp(userInput) if not token.is_stop and token.is_alpha])
            vecInput = vectorizer.transform([lemInput])
            cosSim = cosine_similarity(vecQuestions, vecInput).flatten()
            maxIndex = cosSim.argmax()

            if cosSim[maxIndex] != 0:
                print(answerCSV[maxIndex])
                engine.say(answerCSV[maxIndex])
                engine.runAndWait()
            elif cosSim[maxIndex] == 0:
                print("Sorry i do not know that. Be more Specific")
                engine.say("Sorry i do not know that. Be more Specific")
                engine.runAndWait()
            else:
                print("I did not get that, please try again.")
                print(params[1])
                expr = read_expr(params[1])

                answer = ResolutionProver().prove(expr, kb, verbose=False)
                print(answer)

    else:
        print(answer)
        engine.say(answer)
        engine.runAndWait()
