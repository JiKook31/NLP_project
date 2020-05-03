from paraphrase.predict import Predict
import torch

def paraphrase(sent, paraphraser, num):
    para_list = [paraphraser.infer(sent)]
    for i in range(1, num):
        new_para = paraphraser.infer(para_list[i-1])
        para_list.append(new_para)
    return para_list

def translate(sent, translator):
    return translator.translate(sent)

def para_translate(sentence, num_paras=3):
    translator, paraphraser = loadmodel()
    return para_translate_preloaded(translator, paraphraser, sentence, num_paras=num_paras)


def loadmodel():
    translator = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')
    paraphraser = Predict(checkpoint='checkpoint_coco', directory='paraphrase/coco')
    return translator, paraphraser

def para_translate_preloaded(translator, paraphraser, sentence, num_paras=3):
    sent_trans = translate(sentence, translator)
    sent_paras = [sent_trans]
    sent_paras.extend(paraphrase(sent_trans, paraphraser, num_paras))

    return sent_paras

print(para_translate("Сегодня хорошая погода"))