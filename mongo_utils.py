from pymongo import MongoClient
import pandas as pd
connection_string = 'mongodb+srv://officialreca0:eswxP6699ruM1jWT@cluster0.yhis61q.mongodb.net/?retryWrites=true&w=majority'
client = MongoClient(connection_string)
def load_data_from_mongo(db_name, collection_name):
    db = client[db_name]
    collection = db[collection_name]
    data = list(collection.find())
    df = pd.DataFrame(data)
    return df
def get_product_data(db_name='test', collection_name='products'):
    return load_data_from_mongo(db_name, collection_name)
def get_category_mapping(db_name='test', collection_name='categories'):
    db = client[db_name]
    categories_collection = db[collection_name]
    categories = list(categories_collection.find({}, {'_id': 1, 'name': 1}))
    category_map = {str(category['_id']): category['name'] for category in categories}
    return category_map