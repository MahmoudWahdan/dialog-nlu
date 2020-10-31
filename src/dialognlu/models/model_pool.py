# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from .joint_trans_bert import TfliteJointTransBertModel
from .joint_trans_distilbert import TfliteJointTransDistilBertModel
from .joint_trans_albert import TfliteJointTransAlbertModel
from .joint_trans_roberta import TfliteJointTransRobertaModel
import os
import multiprocessing


LOAD_TFLITE_CLASS_NAME_2_MODEL = {
    'JointTransDistilBertModel': TfliteJointTransDistilBertModel,
    'JointTransBertModel': TfliteJointTransBertModel,
    'JointTransAlbertModel': TfliteJointTransAlbertModel,
    'JointTransRobertaModel': TfliteJointTransRobertaModel
}


"""
To solve the problem that Pool initializer don't give us access to the variables created in initializer, we used global variable
The idea of using global variable is:
    Each worker is in a separate process. Thus, you can use an ordinary global variable.
Source: https://stackoverflow.com/questions/10117073/how-to-use-initializer-to-set-up-my-multiprocess-pool/10118250#10118250

I though about another idea that may seem be the same but a little better than gloabl variable.
The idea is to use a class with static methods. And because the process will have only one class,
then it is safe.
"""
class WorkerProcessor:

    @staticmethod
    def load_model(clazz_name, load_folder_path):
        if clazz_name not in LOAD_TFLITE_CLASS_NAME_2_MODEL:
            raise Exception('%s has no supported tflite model')
        model = LOAD_TFLITE_CLASS_NAME_2_MODEL[clazz_name].load(load_folder_path)
        WorkerProcessor.model = model
        print("Model Loaded, process id: %d" % os.getpid())

    @staticmethod
    def predict_slots_intent(x):
        return WorkerProcessor.model.predict_slots_intent(x[0], x[1], x[2], x[3], x[4])


class NluModelPool:

    def __init__(self, clzz, path, num_process=2):
        self.pool = multiprocessing.Pool(initializer=WorkerProcessor.load_model, initargs=(clzz, path,), processes=num_process)

    def predict_slots_intent(self, X, slots_vectorizer, intent_vectorizer, remove_start_end=True,
                             include_intent_prob=False):
        parameters = []
        for i in range(len(X["valid_positions"])):
            parameters.append(({k:v[[i]] for k,v in X.items()}, slots_vectorizer, intent_vectorizer, remove_start_end, include_intent_prob,))
        output = self.pool.map(WorkerProcessor.predict_slots_intent, parameters)
        slots = []
        intents = []
        for i in output:
            slots.append(i[0])
            intents.append(i[1])
        return slots, intents