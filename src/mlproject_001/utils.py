import os
import sys
import numpy as np
import pandas as pd

from src.mlproject_001.exception import CustomException
from src.mlproject_001.logger import logging

from dotenv import load_dotenv
import pymysql
from sqlalchemy import create_engine


load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        # conn_analytics = pymysql.connect(
        #     host = host,
        #     user = user,
        #     password = password,
        #     db = db,
        #     charset='utf8mb4'
        # )
        conn_analytics = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db}")

        logging.info("Connection established: %s", conn_analytics)
        
        df = pd.read_sql("""
                        select 
                            la.Name loan_name,
                            la.Id loan_id, 
                            la.loan__Disbursal_Date__c as disbursal_date,
                            aps.Id as aps_id,
                            aps.Name,
                            aps.CreatedDate, 
                            deact.CreatedDate as deactivated_date, 
                            aps.loan__Active__c,
                            aps.loan__CL_Contract__c,
                            aps.loan__Debit_Date__c,
                            aps.loan__Recurring_ACH_Start_Date__c, 
                            aps.loan__Recurring_ACH_End_Date__c,
                            aps.loan__Type__c loan__type__c,
                            aps.Source__c
                        from 
                                cl_import.loan__Automated_Payment_Setup__c as aps 
                        right join 
                                cl_import.loan__Loan_Account__c la 
                                on aps.loan__CL_Contract__c = la.Id 
                        left join 
                                (select 
                                        ParentID,
                                        CreatedDate as CreatedDate 
                                from 
                                        cl_import.loan__Automated_Payment_Setup__History 
                                where 
                                        Field='loan__Active__c' and 
                                        OldValue='true' and 
                                        NewValue='false' 
                                        and IsDeleted=0) deact 
                                on aps.Id = deact.ParentID
                        where  
                            (date(deact.CreatedDate) >= date('2024-07-01') 
                            or (deact.CreatedDate is null and date(aps.CreatedDate) >= date('2024-07-01')))
                        order by 
                                aps.CreatedDate
                        limit 3000;""", conn_analytics)
        print(df.head())
        return df
    
    except Exception as e:
        raise CustomException(e, sys)