import os
import sys
import sqlite3


class SQLLiteConnection:
    def __init__(self, dbname="./ftxdb.sqlite"):
        self.dbName = dbname
        print(f'Connecting to  {self.dbName}')
        self.conn = sqlite3.connect(self.dbName)
        conn = self.conn
        # check DB
        self.add_table('signals')

    def add_table(self, name):
        curs = self.conn.cursor()
        curs.execute(f"""create table if not exists {name} ('values' TEXT)""")
        self.conn.commit()

    def search(self, value, table='signals'):
        conn = sqlite3.connect(self.dbName)
        curs = conn.cursor()
        value = str(value)
        curs.execute(f"SELECT * FROM {table} WHERE (?)", (value,))
        conn.commit()
        conn.close()

    def append(self, value, table='signals'):
        conn = sqlite3.connect(self.dbName)
        curs = conn.cursor()
        value = str(value)
        curs.execute(f"INSERT INTO {table} VALUES(?)", (value, ))
        conn.commit()
        conn.close()

    def get_list(self, table='signals'):
        conn = sqlite3.connect(self.dbName)
        curs = conn.cursor()
        curs.execute(f"SELECT * from {table}" )
        rows = curs.fetchall()
        res = []
        for row in rows:
            res.append(row[0])
        conn.commit()
        conn.close()
        return res

    def remove(self, item, table='signals'):
        conn = sqlite3.connect(self.dbName)
        curs = conn.cursor()
        item = str(item)

        curs.execute(f"delete from  {table} where `values`=?", (item, ))
        conn.commit()
        conn.close()

    def clear(self, table='signals'):
        conn = sqlite3.connect(self.dbName)
        curs = conn.cursor()
        curs.execute(f"delete * from {table}")
        conn.commit()
        conn.close()

    def disconnect(self):
        self.disconnect()