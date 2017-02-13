"""
sumarize.py
""" 
from facepy import GraphAPI
import urllib.request
import string
import pandas as pd
import io




def main():
    utf=io.open('summary.txt', 'a', encoding='utf8')
    readfile=io.open("cluster.txt",'r',encoding='utf-8')
    for lines in readfile:
            utf.write(lines)
    readfile.close()
    readfile=io.open("classify.txt",'r',encoding='utf-8')
    for lines in readfile:
            utf.write(lines)
    utf.close()
if  __name__ == '__main__':
    main()
