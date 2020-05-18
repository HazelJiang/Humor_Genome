#this can fill db with data from local file 
#or restore the data in the database in case someone messes up the db
import csv
from ComedianIdentifier.dbmanager import DatabaseManager

name = 'textfiles/Final CSV  - Sheet1.csv'

#parse data from csv
def populate(filename):
    data = []
    with open(filename) as file:
        reader = csv.reader(file, delimiter=',')
        line = 0
        for row in reader:
            if line != 0:                           #excluding column title
                data.append(row)
            line += 1
    return data

if __name__ == '__main__':
    temp_data = populate(name)
    db_manager = DatabaseManager()
    for i in temp_data:
        command = f'''INSERT INTO JOKES VALUES (
            "{i[0]}",
            "{i[3].replace('"', "'")}",
            {i[1]},
            {i[2]}
        )
        '''
        db_manager.execute(command)
        db_manager.commit()
    db_manager.close()