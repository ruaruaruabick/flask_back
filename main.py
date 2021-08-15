from flask import Flask,request
from recoggraph import recogbase64
from predictbaseline import predictbaseline
from predictKOBE import perdictkobe
app = Flask(__name__)
baselinemodel = predictbaseline()
kobemodel = perdictkobe()
@app.route('/')
def hello_world():
   return 'Hello World'

@app.route('/getbaseline')
def getbaseline():
    style = request.args.get("style")
    return baselinemodel.predict(style)

@app.route('/upbase64')
def upbase64():
    base64data = request.args.get("base64data")
    return recogbase64(base64data)

@app.route('/getkobe')
def getkobe():
    title = request.args.get("t")
    user = request.args.get("u")
    
    return [kobemodel.predict(title,user,"a"),kobemodel.predict(title,user,"b"),kobemodel.predict(title,user,"c")]

if __name__ == '__main__':
   app.run(port=8848,debug=True)