# -*- coding: utf-8 -*-
import os
import pymongo
import gridfs

directory = "dataset/modi/test/positive"
data = "test"
tag = "positive"
category = "modi"


client = pymongo.MongoClient()
db = client.precog_dataset
fs = gridfs.GridFS(db)
for file in os.listdir(directory):
	filename=os.fsdecode(file)
	with open(directory+"/"+filename,'rb') as image:
		p = fs.put(image, filename=filename, tag=tag, dataset=data, category=category)
		print (filename, p)