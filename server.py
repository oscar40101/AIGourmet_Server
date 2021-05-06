#Building a simple Keras + deep learning REST API
#https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

#from keras.applications import ResNet50
#from keras.preprocessing.image import img_to_array
#from keras.applications import imagenet_utils
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from tensorflow import keras
from PIL import Image
import flask
from flask import render_template
from flask import request
from flask import Response
import requests
import argparse
import imutils
import pickle
import numpy as np
from cv2 import cv2
from flask_cors import CORS
import io
import os
from pymongo import MongoClient
import json
import base64
import heapq
from imutils.video import VideoStream
import time
import argparse
from yolowebcam import *
import threading


# 初始化Flask物件，並貼上app標籤方便使用該模組功能
app = flask.Flask(__name__)
# 使用Cors，避免瀏覽器間要求服務被擋
CORS(app)
# 先將model這個變數清空
model = None
# 先定義全域變數變清空值
outputFrame = None

lock = threading.Lock()

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app

def detect_motion():
    
    global outputFrame# 將輸出照片設為全域變數
    frame = yolowebcam() # yolowebcam function 透過 yield 回傳值為generator(每個時間點的圖片) 
    for i in frame:  # generator 需透過next或for迴圈取出實際值(圖片)，此利用for迴圈 因為有N個值
        # newframe = next(i) # 利用next只會將下一時間點的圖片取出，會沒辦法達到streaming(串流)的效果
        outputFrame = i.copy() # 將每一張照片(nd.array) 利用copy() 複製到變數 outputFrame
    # print(type(frame)) # nd.array

def food_result():
    yoloresult()
	
def generate(): #透過此function持續輸出給前端新的照片，以達到streaming串流的效果
    	# grab global references to the output frame and lock variables
	global outputFrame
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		# with lock:
		# check if the output frame is available, otherwise skip
		# the iteration of the loop
		if outputFrame is None:
			continue
		# encode the frame in JPEG format
		(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
		# ensure the frame was successfully encoded
		if not flag:
			continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image, mode="caffe")

    # return the processed image
    return image


@app.route('/')
def hello():
    return render_template(r"homepage.html")

@app.route('/about')
def about():
    return render_template(r"about.html")

@app.route('/upload')
def upload():
    return render_template(r"upload.html")

@app.route('/webcam')
def webcam():
    return render_template(r"webcam.html")


@app.route('/yolo')
def yolo():
    detect_motion()
    return flask.jsonify("webcam stop!")

@app.route('/stop')
def stop():
    return flask.jsonify(yolostop()) # yolostop()回傳為分析結果

@app.route('/result')
def result():
    return render_template(r"result.html")

@app.route("/video_feed")
def video_feed(): #透過此api 將generate生成的圖片，以固定格式回傳給前端，其端網頁只需將img標籤的src設為此URL即可
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/search', methods=['POST'])
def search():
    if flask.request.method == "POST":
        data = request.get_json(force=True)#獲取json資料
        # print(data) # {'0': 'eggplant', '1': 'potato'}
        list1 = []
        list1.append(data['one']) # eggplant
        list1.append(data['two']) # potato
        list1 = sorted(list1)
        print(f"server get list {list1}")

        # vaule=request.form['food']#表單method爲post方法,獲取name爲name的輸入框值
        # value=request.args.get('value')#通過鏈接方式獲取值
        # value=request.values.get('radio')#表單method爲post方法，獲取單選框值
        # list1 = request.getParameterValues("food")
        # list1 = request.form.getlist('food')#表單method爲post方法，獲取複選框值
        # print(f"server get : {list1}")

        # 模糊查詢

        client = MongoClient('mongodb+srv://manbox37:manbox37@cluster0.mpgic.mongodb.net/food3?retryWrites=true&w=majority')# 如果你只想連本機端的server你可以忽略，遠端的url填入: mongodb://<user_name>:<user_password>@ds<xxxxxx>.mlab.com:<xxxxx>/<database_name>，請務必既的把腳括號的內容代換成自己的資料。
        db = client["fooddb1"]
        collection = db["fooddb1"]

        if (len(list1)==1):
            fuzzygoogle= collection.find({"$and":[{'Ingredients':{"$regex":list1[0]}}]})
            dog = [d for d in fuzzygoogle]
            print(dog)

        elif (len(list1)==2):
            fuzzygoogle= collection.find({"$and":[{'Ingredients':{"$regex":list1[0]}},{'Ingredients':{"$regex":list1[1]}}]}).limit(3)
            dog = {}
            idx = 0
            names_dic = {} # recipe's name
            urls_dic = {} # recipe's url

            for d in fuzzygoogle:
                names = f"name_{idx}"
                urls = f"url_{idx}"
                # 紀錄菜餚名
                names_dic[names] = d['name']
                # 紀錄菜餚網址
                urls_dic[urls] = d['recipe']
                #存入照片
                img = base64.b64decode(d['image'])
                img_name = f"{idx}.jpg"
                dir_name = './static/menu'
                img_path = os.path.join(dir_name,img_name)
                with open(img_path, 'wb') as f:
                    f.write(img)
                idx += 1

        else:
            fuzzygoogle= collection.find({"$and":[{'Ingredients':{"$regex":list1[0]}},{'Ingredients':{"$regex":list1[1]}},{'Ingredients':{"$regex":list1[2]}}]})
            dog = [d for d in fuzzygoogle]
            print(dog)
        
        # print(names_dic)
        # print(urls_dic)
        
        total = {} # 裝name跟url回去前端
        total['names'] =names_dic
        total['urls'] = urls_dic
        # print(total)

        return flask.jsonify(total)

@app.route('/detect', methods=['POST'])
def detect():
#====================================================
    #分析食物

    #判斷是否資料有收到
    data = {"success": False}
    
    # modelname = "C:\\Users\\Student\\Desktop\\foodClassify\\foodClassify_0122_3\\foodClassify_0122_3.model"
    # labelbin = "C:\\Users\\Student\\Desktop\\foodClassify\\foodClassify_0122_3\\foodClassify_0122_3.pickle"

    #若Request為POST
    if flask.request.method == "POST":
        img = flask.request.files["image"].read()
        img_stream = io.BytesIO(img)
        image = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), 1)
        
        # pre-process the image for classification
        image = cv2.resize(image, (96, 96)) #train 需要288 原本為96
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # load the trained convolutional neural network and the label
        # binarizer
        print("[INFO] loading network...")
        model = load_model("foodClassify_0128_2.model")
        lb = pickle.loads(open("foodClassify_0128_2.pickle", "rb").read())
        print(lb.classes_) # ['apple' 'cauliflower' 'guava' 'kiwi' 'onion' 'orange']
        # classify the input image
        print("[INFO] classifying image...")
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        # label = lb.classes_[idx]# 第一名index
        proba_list = proba.tolist() 

        # 最大的3个数的索引
        max_index_list = map(proba_list.index, heapq.nlargest(3, proba_list))
        proba_max_index_list = list(max_index_list)
        first_index = proba_max_index_list[0] # 第一名index
        second_index = proba_max_index_list[1] # 第二名index
        third_index = proba_max_index_list[2] # 第三名index

        first_label = lb.classes_[first_index] # 第一名label
        second_label = lb.classes_[second_index] # 第二名label
        third_label = lb.classes_[third_index] # 第三名label
        print(first_label) #apple
        print(second_label) #onion
        print(third_label) #kiwi
        label_dic = {}
        label_list = []
        label_dic["first_label"] = first_label
        label_dic["second_label"] = second_label
        label_dic["third_label"] = third_label
        label_list.append(first_label)
        label_list.append(second_label)
        label_list.append(third_label)
        # print(label_dic)

        # #====================================================

        #傳入DB
        # connection


        client = MongoClient('mongodb+srv://manbox37:manbox37@cluster0.mpgic.mongodb.net/food3?retryWrites=true&w=majority')# 如果你只想連本機端的server你可以忽略，遠端的url填入: mongodb://<user_name>:<user_password>@ds<xxxxxx>.mlab.com:<xxxxx>/<database_name>，請務必既的把腳括號的內容代換成自己的資料。
        db = client["fooddb1"]
        collection = db["fooddb1"]

        # test if connection success
        collection.stats  # 如果沒有error，你就連線成功了。

        names_total = {}
        urls_total = {}

        print("database")

        for label in label_list:

            #尋找6筆資料
            cursor = collection.find({'title': label}).limit(6)

            idx = 0
            names_dic = {} # recipe's name
            urls_dic = {} # recipe's url
            

            for data in cursor:
                names = f"{label}_name_{idx}"
                urls = f"{label}_url_{idx}"
                # 紀錄菜餚名
                names_dic[names] = data['name']
                # print(names_dic)
                # 紀錄菜餚網址
                urls_dic[urls] = data['recipe']
                # print(urls_dic)
                #存入照片
                img = base64.b64decode(data['image'])
                img_name = f"{label}_{idx}.jpg"
                dir_name = './static/images'
                img_path = os.path.join(dir_name,img_name)
                with open(img_path, 'wb') as f:
                    f.write(img)
                idx+=1

            # 將菜餚名和網址丟回Server資料夾
            # with open('./recipe_name.json','w', encoding='utf-8') as f:
            #     json.dump(names,f)
            # with open('./recipe_url.json','w', encoding='utf-8') as f:
            #     json.dump(urls,f)
            _names = f"{label}_name"
            names_total[_names] = names_dic
            _urls = f"{label}_url"
            urls_total[_urls] = urls_dic
            idx = 0
            names_dic = {}
            urls_dic = {}
            
        label_dic["names"] = names_total
        label_dic["urls"] = urls_total
        print(label_dic)

            
        return flask.jsonify(label_dic)

if __name__ == '__main__':
	import os

	HOST = os.environ.get('SERVER_HOST', 'localhost')

	# # start the flask app
	app.run(host=HOST, port=5555, debug=True, use_reloader=False)
