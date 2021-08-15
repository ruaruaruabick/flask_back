#encoding:utf-8

import requests
import re 
'''
图像识别组合API
'''
def recogbase64(tempbase64):
    # result = re.search("base64,",tempbase64).span()
    request_url = "https://aip.baidubce.com/api/v1/solution/direct/imagerecognition/combination"
    params = "{\"image\":\"%s\",\"scenes\":[\"ingredient\",\"plant\",\"animal\",\"advanced_general\", \"logo_search\",\"multi_object_detect\"]}" %tempbase64.replace(" ","+")
    access_token = '24.423afa94502b83da9f8c121620a94456.2592000.1630477562.282335-24637773'
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/json'}
    response = requests.post(request_url, data=params, headers=headers)
    print(response.json())
    return response.json()
"""
recogbase64()
"""