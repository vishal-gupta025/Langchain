from pydantic import BaseModel , EmailStr , Field
from typing import Optional

class student(BaseModel):
    name: str
    age : Optional[int] = None
    email: EmailStr
    cgpa : float = Field(gt=0 , lt =10)

new_stu = {'name': 'vishal gupta' , 'email': 'abc@gmail.com' , 'cgpa': 9}

student = student(**new_stu)
print(student)