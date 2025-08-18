from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.amazon.in/Lenovo-Smartchoice-I7-13620H-Keyboard-83K100CJIN/dp/B0F2162VGQ/ref=sr_1_1_sspa?crid=3SCPDVO5WI3B7&dib=eyJ2IjoiMSJ9.6_UXmQ-aHGhq-zmSwemB2tzoe46vuuXAnxw9WWyrRbWynbEjKNng9UgKB3kJ8eGrRztWpPNrMrAwJPpYp2XcCH9IbhVCZlqqc6SD6VSMZcKLXsZAmSOalEUT2R9fUhrbKXDKLXkihdEkTkqpFRWHul6q1MDBgtH9jRLPLtTpWZR-fIAGXuKL3hwHZRBL8GJxsNEHYDAtUOUer0rDtbcN0AparJx1JZIqdnHwKfn-JY8.P47WqeAKYuffRnyUX57hDaP9qU_8FbX6-sXdM2udl68&dib_tag=se&keywords=ideapad&nsdOptOutParam=true&qid=1755429522&sprefix=ideapa%2Caps%2C290&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1'

loader = WebBaseLoader(url) 

docs = loader.load()

print(len(docs))

print(docs[0].page_content)