import skimage.io as io

import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np

def readDatatrain1():
    matfn=r'./label.mat'
    # matfn=r'E:\声音小组\LQPHP\audiotrack\matlab_code\handwriting\20181210_wxy_812_ultraGesture_pauseRecordDemo\label.mat'
    data = sio.loadmat(matfn)

    # 批量读取灰度图片
    data_dir=r'./data'
    # data_dir=r'E:\声音小组\LQPHP\audiotrack\matlab_code\handwriting\20181210_wxy_812_ultraGesture_pauseRecordDemo\gray_picture_1024_512_1024'
    str1=data_dir + '/*.jpg'
    coll = io.ImageCollection(str1)
    print(len(coll))

    label = data['label']
    # label = np.transpose(label)
    img=[]
    colll=[]
    for n in range(0,len(coll)):
        #print('n:\n',n)
        # im = Image.fromarray(coll[n])
        # w,h = im.size
        # im1 = im.thumbnail((w//2,h//2))
        # img1 = np.array(im)
        # img = img1.reshape(-1,img1.shape[0],img1.shape[1],1)
        img = coll[n].reshape(-1,coll[n].shape[0],coll[n].shape[1],1).astype('float32')
        colll.append(img)
    # img = coll.reshape(-1,1,Width,Height)
    collll = np.concatenate(colll,axis=0)
    # label = np.array(y)
    # colll=np.array(coll)
    print('collll.shape:',collll.shape)
    print('label.shape',label.shape)


    train_X, test_X, train_y, test_y = train_test_split(collll,
                                                        label,
                                                        test_size=0.4,
                                                        random_state=0)
    return train_X,test_X,train_y,test_y,label


#
#
# def readDatatrain2():
#     matfn = './label.mat'
#     # matfn=r'E:\声音小组\LQPHP\audiotrack\matlab_code\handwriting\20181210_wxy_812_ultraGesture_pauseRecordDemo\label.mat'
#     data = sio.loadmat(matfn)
#     label = data['label']
#
#     # 批量读取灰度图片
#     data_dir = './data'
#     # str1=data_dir + '/*.jpg'
#     # coll = io.ImageCollection(str1)
#     # print(len(coll))
#     img = []
#     colll = []
#     for n in range(1, 781):
#         img = []
#         img1 = []
#         str1 = []
#         str1 = data_dir + '/' + str(n) + '.jpg'
#         print('str1: ', str1)
#         img = io.imread(str1)
#         img1 = img.reshape(-1, img.shape[0], img.shape[1], 1).astype('float32')
#         print('img.shape: ', img.shape)
#         colll.append(img1)
#     collll = np.concatenate(colll, axis=0)
#     print('collll.shape:', collll.shape)
#     print('label.shape', label.shape)
#
#     sfliename = ['zmh_20181214_gray_picture_1024_512_1024', 'wxy_20181210_gray_picture_1024_512_1024',
#                  'twl_20181211_gray_picture_1024_512_1024',
#                  'lx_20181218_gray_picture_1024_512_1024', 'lqp_20181210_gray_picture_1024_512_1024',
#                  'lp_20181214_gray_picture_1024_512_1024',
#                  'ftc_20190125_gray_picture_1024_512_1024', 'yhy_20190115_gray_picture_1024_512_1024',
#                  'ty_20190307_gray_picture_1024_512_1024',
#                  'data', 'xyk_20190308_gray_picture_1024_512_1024',
#                  'qx_20190308_gray_picture_1024_512_1024',
#                  'hf_20190309_gray_picture_1024_512_1024']
#     label1 = []
#     colll_a = []
#     print('length(sfilename): ', len(sfliename))
#     for ik in range(0, len(sfliename)):
#         for ij in range(1, 781):
#             data_dir1 = sfliename[ik]
#             str1 = []
#             str1 = data_dir1 + '/' + str(ij) + '.jpg'
#             img = []
#             img1 = []
#             img = io.imread(str1)
#
#             img1 = img.reshape(-1, img.shape[0], img.shape[1], 1).astype('float32')
#             print('img.shape: ', img.shape)
#             colll_a.append(img1)
#             print('文件名：', str1)
#         label1 = np.append(label1, label)
#
#     collll_a = np.concatenate(colll_a, axis=0)
#     # print('colll_a.shape:',colll_a.shape)
#     print('collll_a.shape: ', collll_a.shape)
#     # collll = np.concatenate(colll,axis=0)
#     # print('collll.shape:',collll.shape)
#     # print('label.shape',label.shape)
#
#
#
#
#     # train_X, test_X, train_y, test_y = train_test_split(collll_a,
#     #                                                     label1,
#     #                                                     test_size=0.4,
#     #                                                     random_state=0)
#     # del collll
#     # print('label:',label)
#
#     train_X = collll_a
#     test_X = collll
#     train_y = label1
#     test_y = label
#     print('label1.shape:', label1.shape)
#
#     print('train_X.shape: ', train_X.shape)
#     print('test_X.shape: ', test_X.shape)
#     print('train_y.shape: ', train_y.shape)
#     print('test_y.shape: ', test_y.shape)
#     return train_X, test_X, train_y, test_y
#
#
#
#
# def model1(data_X,train_Y,N):
#     model = Sequential()
#     model.add(Conv2D(32, (8, 8), strides=(2, 2), input_shape=(data_X.shape[1], data_X.shape[2], 1), padding='valid',
#                      activation='tanh', kernel_initializer='uniform'))
#     model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
#     model.add(Conv2D(64, (8, 8), strides=(1, 1), padding='same', activation='tanh', kernel_initializer='uniform'))
#     model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
#     model.add(Dropout(0.4))
#     model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='tanh', kernel_initializer='uniform'))
#     model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='tanh',
#                      kernel_initializer='uniform'))  # 没有这一层是89%
#     model.add(Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='tanh', kernel_initializer='uniform'))
#     model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))
#     model.add(Dropout(0.4))
#     model.add(Flatten())
#     model.add(Dense(256, activation='tanh'))
#     model.add(Dropout(0.75))
#     model.add(Dense(128, activation='tanh'))
#     model.add(Dropout(0.75))
#     model.add(Dense(N, activation='softmax'))
#     optimizer_Adamax = keras.optimizers.Adamax(lr=0.0006, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer_Adamax, metrics=[
#         'accuracy'])  ##optimizer='Adamax'60%,,          'Nadam','adam','SGD','RMSprop','Adadelta','TFOptimizer',      'Adagrad',
#     model.summary()
#     # history = model.fit(data_X, train_Y, epochs=3, batch_size=20, validation_split=0.2)
#     return model
#
# def model6(data_X,train_Y,N):
#     model = Sequential()
#     model.add(Conv2D(32, (9, 9), strides=(2, 2), input_shape=(data_X.shape[1], data_X.shape[2], 1), padding='valid',
#                      activation='tanh', kernel_initializer='uniform'))
#     model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#     # model.add(BatchNormalization())
#     model.add(Dropout(0.3))
#     model.add(Conv2D(32, (9, 9), strides=(1, 1), padding='same', activation='tanh', kernel_initializer='uniform'))
#     model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#     # model.add(BatchNormalization())
#     model.add(Dropout(0.3))
#     model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='tanh', kernel_initializer='uniform'))
#     model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='tanh',
#                      kernel_initializer='uniform'))  # 没有这一层是89%
#     model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='tanh', kernel_initializer='uniform'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.4))
#     model.add(Flatten())
#     model.add(Dense(512, activation='tanh'))
#     model.add(Dropout(0.6))
#     model.add(Dense(256, activation='tanh'))
#     model.add(Dropout(0.6))
#     model.add(Dense(N, activation='softmax'))
#     # model.add((lambda x:K.tf.nn.softmax(x)))
#     # model.add(Activation(tf.nn.softmax(dim = N )))
#     # model.add(Activation(tf.nn.softmax(dim=axis)))
#     optimizer_Adamax = keras.optimizers.Adamax(lr=0.0006, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
#     model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=[
#         'accuracy'])  ##optimizer='Adamax'60%,,          'Nadam','adam','SGD','RMSprop','Adadelta','TFOptimizer',      'Adagrad',
#     model.summary()
#
#     return model
#
#
#
#
# def classprint(clf,X_train_std, train_y,X_test_std, test_y,clfName):
#     clf.fit(X_train_std, train_y)
#     print(str(clfName))
#     print('Training accuracy:', clf.score(X_train_std, train_y))
#     print('Test accyracy:', clf.score(X_test_std, test_y))
#     acc=clf.score(X_test_std, test_y)
#     prob = clf.predict_proba(X_test_std)  # 输出分类概率
#     cr=classification_report(test_y, clf.predict(X_test_std))
#     cm=confusion_matrix(test_y, clf.predict(X_test_std))
#     # re.append(classification_report(test_y, lr.predict(X_test_std)))
#     # ma.append(confusion_matrix(test_y, lr.predict(X_test_std)))
#     # cr = LogisticRegression(multi_class='multinomial', penalty='l2', solver='sag', C=100)
#     # # cr.fit(X_train_reduced, train_y)
#     # # accuracy = cross_val_score(cr, X_test_std, test_y.ravel(), cv=10)
#     # # accTrain = cross_val_score(cr, X_train_std, train_y.ravel(), cv=10)
#     # # print('Test accyracy:{}\n{}', np.mean(accuracy), accuracy)
#     # # print("Train accuracy:{}\n{}", np.mean(accTrain), accTrain)
#     predictlabel = clf.predict(X_test_std)
#     return acc,prob,cr,cm,predictlabel
#
#
# def modelUG(data_X,train_Y,N):
#     model = Sequential()
#     model.add(Conv2D(32, (10, 10),activation='relu'))
#     model.add(MaxPooling2D(pool_size=(5, 5)))
#     model.add(Conv2D(32, (5, 5),activation='relu'))
#     model.add(MaxPooling2D(pool_size=(5, 5)))
#     model.add(Conv2D(32, (5, 5),activation='relu'))
#     model.add(MaxPooling2D(pool_size=(5, 5)))
#     model.add(Conv2D(32, (5, 5),activation='relu'))
#     model.add(MaxPooling2D(pool_size=(5, 5)))
#     model.add(Conv2D(32, (3, 3),activation='relu'))
#     model.add(MaxPooling2D(pool_size=(5, 5)))
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.6))
#     model.add(Dense(N, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=[
#         'accuracy'])  ##optimizer='Adamax'60%,,          'Nadam','adam','SGD','RMSprop','Adadelta','TFOptimizer',      'Adagrad',
#     model.summary()
#
#     return model