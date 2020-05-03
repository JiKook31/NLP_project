import tensorflow as tf
from data import Data
from seq2seq import Seq2seq

class Predict:
    def __init__(self, checkpoint='checkpoint', directory='coco'):
        self.data  = Data(directory + '/train_source.txt',
                          directory + '/train_target.txt',
                          directory + '/train_vocab.txt')
        model = Seq2seq(self.data.vocab_size)
        estimator = tf.estimator.Estimator(model_fn=model.make_graph, model_dir=checkpoint)
        def input_fn():
            inp = tf.placeholder(tf.int64, shape=[None, None], name='input')
            output = tf.placeholder(tf.int64, shape=[None, None], name='output')
            tf.identity(inp[0], 'source')
            tf.identity(output[0], 'target')
            dict =  { 'input': inp, 'output': output}
            return tf.estimator.export.ServingInputReceiver(dict, dict)
        self.predictor = tf.contrib.predictor.from_estimator(estimator, input_fn)

    def infer(self, sentence):
        input = self.data.prepare(sentence)
        predictor_prediction = self.predictor({"input": input, "output":input})
        words = [self.data.rev_vocab.get(i, '<UNK>') for i in predictor_prediction['output'][0] if i > 2]
        return ' '.join(words)