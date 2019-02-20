#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : DSSM_first.py
# @Author: WangYe
# @Date  : 2019/1/18
# @Software: PyCharm
import keras
import  matchzoo as mz

train_pack = mz.datasets.wiki_qa.load_data('train', task='ranking')
valid_pack = mz.datasets.wiki_qa.load_data('dev', task='ranking')
predict_pack = mz.datasets.wiki_qa.load_data('test', task='ranking')

preprocessor = mz.preprocessors.DSSMPreprocessor()
train_pack_processed = preprocessor.fit_transform(train_pack)
valid_pack_processed = preprocessor.transform(valid_pack)
predict_pack_processed = preprocessor.transform(predict_pack)


ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]

model = mz.models.DSSM()
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = ranking_task
model.params['mlp_num_layers'] = 3
model.params['mlp_num_units'] = 300
model.params['mlp_num_fan_out'] = 128
model.params['mlp_activation_func'] = 'relu'
model.guess_and_fill_missing_params()
model.build()
model.compile()


train_generator = mz.PairDataGenerator(train_pack_processed, num_dup=1, num_neg=4, batch_size=64, shuffle=True)

pred_x, pred_y = predict_pack_processed[:].unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_x))

history = model.fit_generator(train_generator, epochs=20, callbacks=[evaluate], workers=5, use_multiprocessing=False)