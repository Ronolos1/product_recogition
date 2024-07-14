from pymongo import MongoClient
import os
import requests
import pandas as pd
import shutil

# MongoDB connection setup
connection_string = 'mongodb+srv://officialreca0:eswxP6699ruM1jWT@cluster0.yhis61q.mongodb.net/?retryWrites=true&w=majority'
client = MongoClient(connection_string)

# Function to load data from MongoDB into a DataFrame
def load_data_from_mongo(db_name, collection_name):
    db = client[db_name]
    collection = db[collection_name]
    data = list(collection.find())
    df = pd.DataFrame(data)
    return df

# Function to get products data
def get_products_data(db_name, collection_name='products'):
    return load_data_from_mongo(db_name, collection_name)

# Function to get category mapping from ID to name
def get_category_mapping():
    db = client['test']  # Assuming database name is 'test'
    categories_collection = db['categories']  # Assuming collection name is 'categories'
    categories = list(categories_collection.find({}, {'_id': 1, 'name': 1}))
    category_map = {str(category['_id']): category['name'] for category in categories}
    return category_map

# Function to download images for products
def download_product_images(db_name, train_path='./product_images/train', val_path='./product_images/val', recon_path='./product_images/recon'):
    products_df = get_products_data(db_name)
    category_map = get_category_mapping()

    for path in [train_path, val_path, recon_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    print("Starting image download process...")

    for index, product in products_df.iterrows():
        product_id = str(product['_id'])  # Convert ObjectId to string
        title = product['title']
        category_id = str(product['category'])  # Convert ObjectId to string for category lookup
        images = product['images']  # List of image URLs

        if category_id in category_map:
            category_name = category_map[category_id]
        else:
            category_name = 'Unknown'  # Fallback if category ID doesn't match any category in the map

        recon_category_path = os.path.join(recon_path, category_name)
        train_category_path = os.path.join(train_path, category_name)
        val_category_path = os.path.join(val_path, category_name)

        for path in [recon_category_path, train_category_path, val_category_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        print(f"Downloading images for product '{title}' in category '{category_name}'")

        for i, image_url in enumerate(images):
            image_name = f"{product_id}_{i+1}.jpg"  # Naming convention: productID_1.jpg, productID_2.jpg, ...
            recon_image_path = os.path.join(recon_category_path, image_name)

            # Check if image file already exists in recon
            if os.path.exists(recon_image_path):
                print(f"Skipping download for {image_name}. Image already exists.")
                continue

            # Download image using requests
            try:
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    with open(recon_image_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=1024):
                            file.write(chunk)
                    print(f"Downloaded: {image_name}")
                else:
                    print(f"Failed to download image {image_url}. Status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading image {image_url}: {e}")

            # Copy image to train or val directory based on index
            if i == len(images) - 1:
                val_image_path = os.path.join(val_category_path, image_name)
                if not os.path.exists(val_image_path):
                    shutil.copy(recon_image_path, val_image_path)
                    print(f"Copied to val: {image_name}")
            else:
                train_image_path = os.path.join(train_category_path, image_name)
                if not os.path.exists(train_image_path):
                    shutil.copy(recon_image_path, train_image_path)
                    print(f"Copied to train: {image_name}")

    print("Image download process completed.")

if __name__ == "__main__":
    # Example usage to download images
    download_product_images('test', './product_images/train', './product_images/val', './product_images/recon')
