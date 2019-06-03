import os
from win32com.client import Dispatch

pwd = os.getcwd()

wordApp = Dispatch('word.Application')
wordApp.Visible = True
myDoc = wordApp.Documents.Add()
myRange = myDoc.Range(0, 0)
myRange.InsertBefore('hello python word doc!')
myDoc.SaveAs(pwd + '\\python_word_demo.docx')
myDoc.Close()
wordApp.Quit()