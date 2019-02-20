#!D:/workplace/python
# -*- coding: utf-8 -*-
# @File  : dist_train_ps.py
# @Author: WangYe
# @Date  : 2019/1/11
# @Software: PyCharm

import os
import paddle
import paddle.fluid as fluid

# train reader
BATCH_SIZE = 20
EPOCH_NUM = 30
BATCH_SIZE = 8

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE)

def train():
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    loss = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(loss)
    opt = fluid.optimizer.SGD(learning_rate=0.001)
    opt.minimize(avg_loss)

    place = fluid.CPUPlace()
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe = fluid.Executor(place)

    # fetch distributed training environment setting
    training_role = os.getenv("PADDLE_TRAINING_ROLE", "PSERVER")
    port = os.getenv("PADDLE_PSERVER_PORT", "6174")
    pserver_ips = os.getenv("PADDLE_PSERVER_IPS", "localhost")
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    eplist = []
    for ip in pserver_ips.split(","):
        eplist.append(':'.join([ip, port]))
    pserver_endpoints = ",".join(eplist)
    trainers = int(os.getenv("PADDLE_TRAINERS","1"))
    current_endpoint = os.getenv("PADDLE_CURRENT_IP", "localhost") + ":" + port

    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id = trainer_id,
        pservers = pserver_endpoints,
        trainers = trainers)

    if training_role == "PSERVER":
        pserver_prog = t.get_pserver_program(current_endpoint)
        startup_prog = t.get_startup_program(current_endpoint, pserver_prog)
        exe.run(startup_prog)
        exe.run(pserver_prog)
    elif training_role == "TRAINER":
        trainer_prog = t.get_trainer_program()
        exe.run(fluid.default_startup_program())

        for epoch in range(EPOCH_NUM):
            for batch_id, batch_data in enumerate(train_reader()):
                avg_loss_value, = exe.run(trainer_prog,
                                      feed=feeder.feed(batch_data),
                                      fetch_list=[avg_loss])
                if (batch_id + 1) % 10 == 0:
                    print("Epoch: {0}, Batch: {1}, loss: {2}".format(
                        epoch, batch_id, avg_loss_value[0]))
        # destory the resource of current trainer node in pserver server node
        exe.close()
    else:
        raise AssertionError("PADDLE_TRAINING_ROLE should be one of [TRAINER, PSERVER]")

train()