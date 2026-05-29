from pydantic import BaseModel, Field, EmailStr
from typing import Annotated, List, Optional

__all__ = ['Email', 'Email_Buckets', 'Email_List']

class Email(BaseModel):
    sender : Annotated[EmailStr, Field(description="email of sender")]
    subject : Annotated[str, Field(description="Subject of mail")]
    body : Annotated[Optional[str], Field(description="a summary of content of mail")]

class Email_Buckets(BaseModel):
    urgent : Annotated[Optional[List[Email]], Field(description="urgent mails requiring immediate attention")]
    noble : Annotated[Optional[List[Email]], Field(description="mails requiring attention but not immeditely")]
    common : Annotated[Optional[List[Email]], Field(description="mails not requied to be read")] 

class Email_List(BaseModel):
    emails : Annotated[List[Email], Field(description="list of emails for user")]