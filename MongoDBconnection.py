
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import BulkWriteError
import pandas as pd
import json
uri = "mongodb+srv://normanisapplying_db_user:DSCI560@redditdb.4g6rz8z.mongodb.net/?appName=redditDB"

# Create a new client and connect to the server



class MongoDBConnection:
    def __init__(self, uri=uri, db_name="reddit_db", collection_name="posts"):
        """
        initialize the database and set 
        """
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(f"Connection failed: {e}")

        # using url as unique index
        self.collection.create_index("url", unique=True)
        print(f"Index created/verified for field 'url' in collection '{collection_name}'")

    def insert_data(self, data):
        """
        insert data
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
            
        if not data:
            print("No data provided to insert.")
            return

        try:
            
            result = self.collection.insert_many(data, ordered=False)
            print(f"Successfully inserted {len(result.inserted_ids)} new documents.")
            
        except BulkWriteError as e:
            
            inserted_count = e.details['nInserted']
            duplicates = len(data) - inserted_count
            print(f"Inserted {inserted_count} new documents. Skipped {duplicates} duplicates.")


    def upload_json_file(self, file_path):
        """
         for local json data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # transfer to dict 
            if isinstance(data, dict):
                data = [data]
            
            # ensure it is list
            if not isinstance(data, list):
                print(f"Error: The JSON file content must be a list or a dict. Got {type(data)}.")
                return

            print(f"Successfully read {len(data)} records from '{file_path}'.")

            
            self.insert_data(data)

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from '{file_path}'. Please check the format.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_data(self, query=None):
        """
        fetch data from database
        """
        if query is None:
            query = {}
            
       
        cursor = self.collection.find(query, {"_id": 0})
        df = pd.DataFrame(list(cursor))
        print(f"Retrieved {len(df)} documents from database.")
        return df

    def close(self):
        """
        close connection
        """
        self.client.close()
        print("MongoDB connection closed.")

   
if __name__ == "__main__":
    # initialize
    mongo = MongoDBConnection()

    mongo.upload_json_file("posts.json")

    print(f"currently has: {mongo.collection.count_documents({})} instances of data")
    mongo.close()