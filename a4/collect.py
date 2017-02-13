"""
collect.py
""" 
from collections import  deque
import pandas as pd
import configparser
from facepy import GraphAPI
import time
import gzip


import io
import tarfile
import matplotlib.pyplot as plt
import networkx as nx

def collectdata():
    '''
    Collect Comments From Data and in the 1st iteration collect the pages the page likes
    '''
    q=deque()
    creategraph=4
    columns = ['Id', 'message', 'createdTime','name','statusType']
    config = configparser.ConfigParser()
    config.read("configure.txt")
    graph = GraphAPI(config.get("facebook","USER_TOKEN"))
    arrayresponse=dict()
    commentid=set()
    timestamp = int(time.time())
    df = pd.DataFrame(columns=columns) 
    val=0
    for i in range(0,15):
        response=dict()
        untilvalue=timestamp-2579200
        response = graph.get(config.get("facebook","Trump")+ "?fields=feed.limit(100).since("+str(untilvalue)+").until("+str(timestamp)+"){name,message,from,created_time,status_type}", page=False, retry=3)
       
        #fields=posts{comments.limit(20)}
        
        #response = graph.get(config.get("facebook","Trump")+ "?fields=posts.since("+str(untilvalue)+").until("+str(timestamp)+"){comments.limit(70),name,message,from,created_time,status_type}", page=False, retry=3)
        arrayresponse[i]=response
        timestamp=untilvalue
        creategraph=creategraph-1
        if(creategraph==1):
            cluster_create_graph(config.get("facebook","Trump"),graph)
            creategraph=0
        for respoonseval in arrayresponse.items():
       #  if("posts"  in respoonseval[1]):
            for items in (respoonseval[1])["feed"]["data"]:
                if(items['id'] not in commentid and "message"  in items):
                    commentid.add(items['id'])
                    temp=items["from"]
                    df.loc[val]=[items['id'],
                                 items["message"],
                                  items["created_time"],
                                  temp["id"],items["status_type"]]
                    #datalist.append(df2)
                    val=val+1  
                    
        df.to_pickle("Trump.csv")       
    
   
    print ('Data Set Size=',len(df))
    return df
import pickle
def cluster_create_graph(parentid,graph):
    utf = io.open('edges.txt', 'w', encoding='utf8')
    sumutf=io.open('summary.txt', 'w', encoding='utf8')
    sumutf.write("FaceBook Data from official Page of  Donald Trump is used")
    sumutf.close()
    utf.close()
    utf = io.open('edges.txt', 'a', encoding='utf8')
    q=deque()
    G=nx.Graph()
    jumps=9
    config = configparser.ConfigParser()
    config.read("configure.txt")
    q.append(config.get("facebook","Trump"))
    pagename=dict()
    pageid=set()
    graph = GraphAPI(config.get("facebook","USER_TOKEN"))
    FbpageDetails = graph.get(config.get("facebook","Trump"), page=False, retry=3)
    pagename[config.get("facebook","Trump")]=FbpageDetails["name"]
    flag=0
    lasthopid=""
    values=""
    while(len(q)>0):
        pageids=q.popleft()
        jumps=jumps-1
       # print (jumps)
        responselikes = graph.get(pageids+"?fields=likes", page=False, retry=3)
        if(pageids==lasthopid): 
            jumps=12
        if(pageids not in  pageid): 
            pageid.add(pagename[pageids])
        if(len(responselikes)>1):
            i=0
            totalusers=len(responselikes["likes"]["data"])
            for response in responselikes["likes"]["data"]:
                foreignchar=0
                if flag==0:
                    i=i+1
                namelist=[]
                if(response["id"] not in pageid):
                    pagename[response["id"]]= response["name"]
                if(jumps>=0):
                    q.append(response["id"])
                values=response["name"]
                
                pageid.add(response["id"])
             #   for vals in values:
              #      if(vals!="'" or vals!="\""):
               #         if(ord(vals)>132 or ord(vals)<0):
                #                values=values.replace(vals,"")
                 #               foreignchar=1
                  #              break
                if(foreignchar!=1):
                    namelist.append(values)
                #f.writelines(+"\t"+values+"\n")
                    utf.write(pagename[pageids] + '\t' + values + '\n')
                if(i>=totalusers and flag==0):
                        flag=1
                        lasthopid=response["id"]
    utf.close()
   
def main():
  
   df=collectdata()  
    #df = pd.read_pickle("laptop.csv")
   # print("Total FB  Comments from Page of Donald J. Trump",str(len(df)))
    
    
if __name__ == '__main__':
    main()
