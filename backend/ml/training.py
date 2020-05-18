##machine learning model and classifier
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
#Prevent posible package not found error
from ComedianIdentifier.dbmanager import DatabaseManager
from ComedianIdentifier import cleaners


def get_data_from_db(db_manager: DatabaseManager, table_name: str):
    """gets data from @code(table_name) from database
    
    Arguments:
        db_manager {DatabaseManager} -- a DatabaseManager instance that 
                                        has connection to desired db
        table_name {str} -- name of the table
    
    Returns:
        [list] -- list of tuple containing row element
    """
    command = f'''SELECT * FROM {table_name}'''
    db_manager.execute(command)
    data = db_manager.fetch_all()
    return data

# db_manager = DatabaseManager()
# get_data_from_db(db_manager, 'JOKES')
